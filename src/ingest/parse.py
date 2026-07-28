"""
src/ingest/parse.py
Session 1 — Document parser for the RAG assistant.

Walks data/raw/, extracts clean text from PDF / HTML / TXT files, and writes:
  - data/processed/<name>.txt        one clean text file per source (for eyeballing)
  - data/processed/documents.jsonl   one JSON record per source (for the next stage)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import trafilatura

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST = PROCESSED_DIR / "documents.jsonl"

SUPPORTED = {".pdf", ".html", ".htm", ".txt"}

# Lines matching these are page noise and get dropped (add patterns as you find them)
NOISE_PATTERNS = [
    re.compile(r"^Git:\s*[0-9a-f]{6,}", re.IGNORECASE),  # LaTeX build footer
]


def is_noise(line: str) -> bool:
    return any(p.match(line) for p in NOISE_PATTERNS)


# --- Text cleaning -----------------------------------------------------
def clean_text(text: str) -> str:
    """Normalise whitespace and fix common PDF extraction artefacts."""
    # Join words hyphenated across a line break: "Aufent-\nhaltstitel" -> "Aufenthaltstitel"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Turn single newlines inside paragraphs into spaces, keep blank lines as breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse runs of spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into a single blank line
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# --- Extractors --------------------------------------------------------
def extract_pdf(path: Path) -> str:
    """Extract horizontal text only (drops rotated watermarks) and skip noise lines."""
    doc = fitz.open(path)
    lines_out: list[str] = []
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                # line["dir"] = (cos, sin) of the writing direction.
                # Horizontal text has sin ~ 0; a watermark is rotated, so skip it.
                if abs(line["dir"][1]) > 0.01:
                    continue
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text or is_noise(text):
                    continue
                lines_out.append(text)
    doc.close()
    return "\n".join(lines_out)


def extract_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return trafilatura.extract(raw) or ""


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    if suffix == ".txt":
        return extract_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


# --- Main --------------------------------------------------------------
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No documents found in {RAW_DIR}. Drop some PDF/HTML files in there first.")
        return

    records = []
    for path in files:
        print(f"Parsing {path.name} ...", end=" ")
        try:
            text = clean_text(extract(path))
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        if len(text) < 50:
            print("skipped (almost no text — likely a scanned PDF, needs OCR)")
            continue

        doc_id = path.stem
        records.append({
            "id": doc_id,
            "source": path.name,
            "type": path.suffix.lower().lstrip("."),
            "n_chars": len(text),
            "text": text,
        })
        (PROCESSED_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        print(f"ok ({len(text):,} chars)")

    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(records)} document(s) -> {PROCESSED_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()