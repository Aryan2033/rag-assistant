"""
src/eval/gold_set.py — Session 16
Loads and validates the gold evaluation set.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import GOLD_SET

REQUIRED = {"id", "question", "ground_truth", "answerable", "expected_keywords", "category"}


def load_gold() -> list[dict]:
    if not GOLD_SET.exists():
        raise FileNotFoundError(f"Missing {GOLD_SET}")
    rows = [json.loads(l) for l in GOLD_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = set()
    for r in rows:
        missing = REQUIRED - r.keys()
        if missing:
            raise ValueError(f"{r.get('id', '?')} missing fields: {missing}")
        if r["id"] in ids:
            raise ValueError(f"Duplicate id: {r['id']}")
        ids.add(r["id"])
    return rows


if __name__ == "__main__":
    rows = load_gold()
    print(f"Loaded {len(rows)} gold questions.\n")
    ans = Counter(r["answerable"] for r in rows)
    print(f"Answerable: {ans[True]}   Unanswerable (refusal tests): {ans[False]}\n")
    print("By category:")
    for cat, n in sorted(Counter(r['category'] for r in rows).items()):
        print(f"  {cat:16s} {n}")