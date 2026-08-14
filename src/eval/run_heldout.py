"""
src/eval/run_heldout.py — Held-out generalization test.
Same scoring as run_eval.py, but on questions NOT used to build/tune the system.
The score here is the honest generalization estimate.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EVAL_DIR
from src.eval.run_eval import score_one
from src.generation.answer import answer

HELD_OUT = EVAL_DIR / "held_out.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in HELD_OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Running {len(rows)} HELD-OUT questions (never tuned against)...\n")

    results = []
    t0 = time.time()
    for row in rows:
        reply, _ = answer(row["question"])
        correct = score_one(row, reply)
        results.append({"row": row, "reply": reply, "correct": correct})
        mark = "✓" if correct else "✗"
        print(f"{mark} [{row['id']}] {row['question'][:60]}")
        if not correct:
            print(f"     expected: {row['ground_truth']}")
            print(f"     got:      {reply[:100]}")
    elapsed = time.time() - t0

    total = len(results)
    correct = sum(r["correct"] for r in results)
    print("\n" + "=" * 50)
    print(f"HELD-OUT ACCURACY:  {correct}/{total}  ({100*correct/total:.1f}%)")
    print("=" * 50)
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        c = r["row"]["category"]
        by_cat[c][1] += 1
        by_cat[c][0] += int(r["correct"])
    print("By category:")
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"  {cat:16s} {c}/{n}")
    print(f"\nTime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()