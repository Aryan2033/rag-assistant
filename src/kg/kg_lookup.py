"""
src/kg/kg_lookup.py — Session 12
Answer a question directly from the knowledge graph, when possible.
Parses the question into (entity=module, attribute=field), then looks up the triple.
Returns None if the question isn't a clean single-fact KG query — so the caller
can fall back to RAG.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import KG_JSON

# --- attribute detection: question keywords -> KG field ---
ATTRIBUTE_KEYWORDS = {
    "manager":  ["who teaches", "module manager", "manager", "lecturer", "professor",
                 "who is responsible", "taught by", "modulverantwortliche", "wer ist"],
    "email":    ["email", "e-mail", "contact", "mail address"],
    "credits":  ["credits", "credit", "ects", "cp", "how many credits"],
    "offered":  ["offered", "which term", "what term", "semester is", "summer or winter"],
    "type":     ["mandatory or elective", "elective or mandatory", "module type",
                 "is it mandatory", "is it elective", "compulsory"],
    "language": ["language", "taught in", "which language"],
    "number":   ["module number", "module code", "module id"],
    "exam":     ["exam format", "exam", "final grade", "grade format", "examination", "assessment"],
}

# a few aliases so short forms resolve to full module names
ALIASES = {
    "nlp": "Natural Language Processing",
    "ml and deep learning": "Machine Learning and Deep Learning",
    "ml": "Machine Learning and Deep Learning",
    "ai": "Artificial Intelligence",
    "big data": "Big Data & Data Mining",
    "data mining": "Big Data & Data Mining",
    "thesis": "Masterarbeit",
    "master's thesis": "Masterarbeit",
    "master thesis": "Masterarbeit",
    "mostflexipl": "Advanced Programming with MOSTflexiPL",
}

_kg = None
_by_name = None


def _load():
    global _kg, _by_name
    if _kg is None:
        _kg = json.loads(KG_JSON.read_text(encoding="utf-8"))
        _by_name = {m["name"].lower(): m for m in _kg}
    return _kg, _by_name


# High-signal keywords that should win regardless of length (checked first, in order)
PRIORITY = [
    ("email",   ["email", "e-mail", "mail address"]),
    ("credits", ["how many credits", "credits", "ects"]),
    ("exam",    ["exam format", "exam", "final grade", "examination", "assessment"]),
    ("offered", ["which term", "what term", "offered", "summer or winter"]),
    ("type",    ["mandatory or elective", "elective or mandatory", "module type", "compulsory"]),
    ("number",  ["module number", "module code", "module id"]),
    ("language",["taught in", "which language", "what language"]),
]


def detect_attribute(question: str) -> str | None:
    q = question.lower()
    # 1) high-signal attributes win first, so "email of the manager" -> email
    for attr, kws in PRIORITY:
        if any(kw in q for kw in kws):
            return attr
    # 2) manager is the fallback (many phrasings: who teaches, lecturer, professor, ...)
    for kw in ["who teaches", "module manager", "manager", "lecturer", "professor",
               "who is responsible", "taught by", "modulverantwortliche", "wer ist"]:
        if kw in q:
            return "manager"
    return None


def detect_module(question: str) -> dict | None:
    kg, by_name = _load()
    q = question.lower()

    # 1) direct full-name match (longest name first, so 'Data Analytics' beats nothing partial)
    matches = [m for m in kg if m["name"].lower() in q]
    if matches:
        return max(matches, key=lambda m: len(m["name"]))

    # 2) alias match
    for alias, full in ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return by_name.get(full.lower())

    return None


def kg_lookup(question: str) -> dict | None:
    """Return {'answer','entity','attribute'} if the KG can answer, else None."""
    q = question.lower()

    # Guard 1: requirement/prerequisite questions are NOT plain attribute lookups
    if any(p in q for p in ["required to start", "required to begin", "prerequisite",
                            "needed to start", "needed to begin", "to be admitted",
                            "requirement to", "how many cp are required", "credits are required"]):
        return None

    # Guard 2: the KG has no dates/times — let RAG handle (and correctly refuse)
    if any(p in q for p in ["date", "when exactly", "what time", "deadline", "calendar"]):
        return None

    module = detect_module(question)

    # Remove the module's name from the question so its words (e.g. "Language"
    # in "Natural Language Processing") can't be mistaken for an attribute.
    q_wo_entity = question
    if module:
        q_wo_entity = re.sub(re.escape(module["name"]), " ", question, flags=re.IGNORECASE)

    attribute = detect_attribute(q_wo_entity)
    if not module or not attribute:
        return None
    value = module.get(attribute, "")
    if not value:
        return None
    return {"answer": value, "entity": module["name"], "attribute": attribute}

if __name__ == "__main__":
    tests = [
        "Who is the module manager for Data Analytics?",         # q07
        "What is the module type of Data Analytics?",            # q25
        "What is the email address of the NLP module manager?",  # q28
        "How many credits is Projekt?",
        "Wer ist der Modulverantwortliche für Natural Language Processing?",
        "How do I register my Anmeldung?",                       # should be None -> RAG
    ]
    for q in tests:
        r = kg_lookup(q)
        if r:
            print(f"KG  ✓  {q}")
            print(f"        {r['entity']} . {r['attribute']} = {r['answer']}")
        else:
            print(f"KG  –  {q}   (no KG hit -> would fall back to RAG)")