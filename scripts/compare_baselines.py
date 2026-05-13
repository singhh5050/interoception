"""Compare baseline sweeps across models. Reports honest correctness counts,
timeout rates, acknowledgment rates, and the model-specific failure modes
seen in the transcripts.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


# Strict acknowledgment regex (same one used in earlier turn-by-turn analyses)
ACK = re.compile(
    r"\b(elapsed|seconds?\s+(left|remaining|so\s+far)|running\s+out\s+of\s+time|"
    r"time\s+(is|left)|budget|hurry|deadline|%\s+(left|remaining|gone)|"
    r"\d+s\s+(left|remaining))\b",
    re.IGNORECASE,
)

# Detect "gave up" answers (model bails instead of guessing)
GAVE_UP = re.compile(
    r"^(?:not\s+possible|none|n/?a|impossible|no\s+solution|unsolvable|cannot)",
    re.IGNORECASE,
)


def analyze(label: str, runs_dir: Path) -> dict:
    summary_csv = runs_dir / "summary.csv"
    if not summary_csv.exists():
        return {"label": label, "missing": True}
    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"] == "True")
    n_wrong = sum(1 for r in rows if r["correct"] == "False")
    n_none = sum(1 for r in rows if r["correct"] == "")
    n_timeout = sum(1 for r in rows if r["timed_out"] == "True")
    n_gave_up = sum(1 for r in rows if r["answer"] and GAVE_UP.search(r["answer"]))
    tokens = [int(r["total_output_tokens"]) for r in rows]
    elapsed = [float(r["elapsed_s"]) for r in rows]

    # acknowledgment scan over transcripts
    n_ack = 0
    for fp in sorted(runs_dir.glob("p*.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        for t in data.get("turns", []):
            if t["role"] != "assistant":
                continue
            if ACK.search(t.get("content", "")):
                n_ack += 1
                break

    return {
        "label": label,
        "n": n,
        "correct_pct": 100 * n_correct / n,
        "wrong_pct": 100 * n_wrong / n,
        "timeout_pct": 100 * n_timeout / n,
        "fmt_break_pct": 100 * (n_none - n_timeout) / n,
        "gave_up_pct": 100 * n_gave_up / n,
        "ack_pct": 100 * n_ack / n,
        "avg_tokens": sum(tokens) / n,
        "avg_elapsed_s": sum(elapsed) / n,
    }


def main():
    sweeps = [
        ("Qwen2.5-7B",        Path("runs/eval_baseline_gh200")),
        ("Qwen2.5-14B",       Path("runs/eval_baseline_qwen14b")),
        ("Qwen2.5-Math-7B",   Path("runs/eval_baseline_math7b")),
        ("Qwen2.5-3B",        Path("runs/eval_baseline_qwen3b")),
        ("Llama-3.2-3B",      Path("runs/eval_baseline_llama3b")),
        ("Phi-4-mini",        Path("runs/eval_baseline_phi4mini")),
        ("Falcon3-7B",        Path("runs/eval_baseline_falcon7b")),
    ]
    results = [analyze(label, d) for label, d in sweeps]

    print(f"\n{'model':>18}  {'n':>4}  {'corr':>5}  {'wrong':>5}  {'TO':>5}  "
          f"{'fmt':>5}  {'gave_up':>7}  {'ack':>5}  {'tokens':>7}  {'elapsed_s':>9}")
    for r in results:
        if r.get("missing"):
            print(f"  {r['label']:>16}  (missing)")
            continue
        print(f"  {r['label']:>16}  {r['n']:>4}  "
              f"{r['correct_pct']:>4.0f}%  {r['wrong_pct']:>4.0f}%  {r['timeout_pct']:>4.0f}%  "
              f"{r['fmt_break_pct']:>4.0f}%  {r['gave_up_pct']:>6.0f}%  {r['ack_pct']:>4.0f}%  "
              f"{r['avg_tokens']:>7.0f}  {r['avg_elapsed_s']:>9.1f}")


if __name__ == "__main__":
    main()
