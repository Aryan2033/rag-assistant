"""
src/eval/run_eval.py — Session 17
Runs the full gold set through the RAG system and scores it:
  - answerable questions: keyword-match (does the answer contain the expected fact?)
  - unanswerable questions: refusal detection (did it correctly decline?)
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.gold_set import load_gold
from src.generation.answer import answer

# Phrases that indicate the model correctly declined to answer
REFUSAL_MARKERS = [
    "don't have", "do not have", "not covered", "not mentioned", "not specified",
    "no information", "not in the", "cannot find", "can't find", "not available",
    "isn't in", "is not in", "not provided", "there is no", "no such",
    "cannot be found", "can't be found", "could not find", "couldn't find",
    "does not mention", "doesn't mention", "not mention", "does not contain",
]


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def keywords_present(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return all(k.lower() in low for k in keywords)


def score_one(row: dict, reply: str) -> bool:
    if row["answerable"]:
        # Correct = contains the expected fact AND didn't wrongly refuse
        return keywords_present(reply, row["expected_keywords"]) and not is_refusal(reply)
    else:
        # Correct = correctly refused
        return is_refusal(reply)


def main() -> None:
    gold = load_gold()
    print(f"Running {len(gold)} questions through the system...\n")

    results = []
    t0 = time.time()
    for i, row in enumerate(gold, 1):
        reply, _ = answer(row["question"])
        correct = score_one(row, reply)
        results.append({"row": row, "reply": reply, "correct": correct})
        mark = "✓" if correct else "✗"
        print(f"{mark} [{row['id']}] {row['question'][:55]}")
        if not correct:
            print(f"     expected: {row['ground_truth']}")
            print(f"     got:      {reply[:90]}")

    elapsed = time.time() - t0

    # --- Scorecard ---
    total = len(results)
    correct = sum(r["correct"] for r in results)

    ans = [r for r in results if r["row"]["answerable"]]
    una = [r for r in results if not r["row"]["answerable"]]
    ans_correct = sum(r["correct"] for r in ans)
    una_correct = sum(r["correct"] for r in una)

    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        c = r["row"]["category"]
        by_cat[c][1] += 1
        by_cat[c][0] += int(r["correct"])

    print("\n" + "=" * 52)
    print("SCORECARD")
    print("=" * 52)
    print(f"Overall accuracy:        {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"Answerable (found fact): {ans_correct}/{len(ans)}  ({100*ans_correct/len(ans):.1f}%)")
    print(f"Unanswerable (refused):  {una_correct}/{len(una)}  ({100*una_correct/len(una):.1f}%)")
    print(f"\nBy category:")
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"  {cat:16s} {c}/{n}")
    print(f"\nTotal time: {elapsed:.0f}s  ({elapsed/total:.1f}s per question)")


if __name__ == "__main__":
    main()