"""
src/kg/build_kg.py — Session 11
Build a knowledge graph of module facts by STRUCTURED extraction from the handbook PDF.

Why structured extraction (not spaCy NER)? The handbook has a fixed schema — every
module lists the same labelled fields. Parsing that schema directly gives exact,
unambiguous (entity, attribute, value) triples, which is precisely what fixes the
entity-attribute binding errors the RAG evaluation revealed. spaCy NER becomes the
right tool later, for unstructured documents with no fixed schema.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import PROJECT_ROOT, KG_DIR, KG_JSON

import pymupdf  # PyMuPDF

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Field labels that appear on their own line, each followed by its value line
LABELS = {
    "Module Number", "Module Manager", "E-Mail", "Credits", "Workload Class",
    "Workload Self-Study", "Offered", "Modul Type", "Language", "Use in other SG",
    "Module Duration", "Participation Requirements:", "Module Objectives",
}

FIELD_MAP = [
    ("Module Manager", "manager"),
    ("E-Mail", "email"),
    ("Credits", "credits"),
    ("Offered", "offered"),
    ("Modul Type", "type"),
    ("Language", "language"),
]


def read_pdf_lines(pdf_path: Path) -> list[str]:
    """Extract horizontal text lines, dropping rotated watermark and Git footer."""
    doc = pymupdf.open(pdf_path)
    lines: list[str] = []
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                if abs(line["dir"][1]) > 0.01:          # rotated watermark
                    continue
                t = "".join(s["text"] for s in line["spans"]).strip()
                if not t or re.match(r"^Git:\s*[0-9a-f]{6,}", t):
                    continue
                lines.append(t)
    doc.close()
    return lines


def build(pdf_path: Path) -> list[dict]:
    lines = read_pdf_lines(pdf_path)

    def value_after(i: int) -> str:
        """The value is the next line — unless that line is itself a label (empty field)."""
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        return "" if nxt in LABELS else nxt

    # Each module starts at a 'Module Number' label whose value is the module code
    anchors = [i for i, l in enumerate(lines)
               if l == "Module Number" and re.match(r"^\d{3,5}$", value_after(i))]

    modules = []
    for idx, a in enumerate(anchors):
        end = anchors[idx + 1] if idx + 1 < len(anchors) else len(lines)
        block = range(a, end)
        number = value_after(a)
        name = lines[a - 2].strip() if a - 2 >= 0 else "?"  # name sits 2 lines above

        rec = {"number": number, "name": name}
        for label, key in FIELD_MAP:
            rec[key] = ""
            for i in block:
                if lines[i] == label:
                    rec[key] = value_after(i)
                    break

        # Exam / final grade: label and value may be on the same OR the next line
        rec["exam"] = ""
        for i in block:
            m = re.match(r"^Final grade:\s*(.*)$", lines[i])
            if m:
                rec["exam"] = m.group(1).strip() or (lines[i + 1] if i + 1 < len(lines) else "")
                break

        modules.append(rec)
    return modules


def to_triples(modules: list[dict]) -> list[tuple[str, str, str]]:
    """Flatten into (entity, attribute, value) triples — the graph view."""
    triples = []
    for m in modules:
        for attr in ("number", "manager", "email", "credits", "offered", "type", "language", "exam"):
            if m.get(attr):
                triples.append((m["name"], attr, m[attr]))
    return triples


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF found in {RAW_DIR}")
        return
    pdf_path = pdfs[0]
    print(f"Building KG from {pdf_path.name} ...")

    modules = build(pdf_path)
    KG_DIR.mkdir(parents=True, exist_ok=True)
    KG_JSON.write_text(json.dumps(modules, ensure_ascii=False, indent=2), encoding="utf-8")

    triples = to_triples(modules)
    print(f"\nExtracted {len(modules)} modules, {len(triples)} facts (triples).\n")
    for m in modules:
        print(f"  {m['number']:>5}  {m['name'][:34]:34}  mgr={m['manager'][:20]:20}  cr={m['credits']:>2}  {m['type']}")
    print(f"\nSaved -> {KG_JSON}")
    print("\nSample triples:")
    for t in triples[:6]:
        print(f"  ({t[0]}) --{t[1]}--> ({t[2]})")


if __name__ == "__main__":
    main()