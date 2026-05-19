"""Reproduce one baseline cell (Qwen3-4B @ T=30s, n=50) via the new env.

Compare bucket frequencies to Track A's traj_*_qwen3_4b (T=30, n=10):
  sim:  20% correct, 80% none (mostly fmt-break / quit)
  real: 30% correct, 70% none

A successful reproduction lands in the same range. Exact match is not expected
(verifiers framework controls sampling, batching, and template subtly differently
from the bespoke rollout.py), but the rough distribution should match.
"""
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

from interoception_countdown import load_environment
from openai import AsyncOpenAI
import verifiers as vf

# Same _bucket logic as in the env, repeated here so we can apply it to results without
# importing private helpers.
from _solver import validate_solution


def bucket(state, answer):
    if not state.get("answer_emitted"):
        return "timeout"
    parsed = state.get("parsed_answer")
    ok = validate_solution(parsed, answer["nums"], answer["target"])
    if ok is True: return "correct"
    if ok is False: return "wrong"
    return "quit"


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    model = "Qwen/Qwen3-4B-Instruct-2507"
    T = 30.0

    env = load_environment(
        problems_jsonl="data/eval.jsonl",
        target_s_min=T, target_s_max=T,
        max_turns=16,
    )
    print(f"env loaded — running n={n} at T={T}s on {model}")

    raw = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    client = vf.OpenAIChatCompletionsClient(raw)
    sampling = {"max_completion_tokens": 256, "temperature": 0.7}

    out_dir = Path("runs/env_repro_qwen3_4b_T30")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"correct": 0, "wrong": 0, "quit": 0, "timeout": 0}
    rows = []
    t0 = time.time()

    for i in range(n):
        row = env.dataset[i]
        state = await env.rollout(input=row, client=client, model=model, sampling_args=sampling)
        await env.rubric.score_rollout(state)
        b = bucket(state, row["answer"])
        counts[b] += 1
        rows.append({
            "idx": i,
            "nums": "-".join(str(x) for x in row["answer"]["nums"]),
            "target": row["answer"]["target"],
            "bucket": b,
            "n_turns": len(state["trajectory"]),
            "elapsed_s": round(state.get("elapsed_s", 0.0), 2),
            "answer": state.get("parsed_answer"),
            "reward": round(float(state.get("reward") or 0.0), 3),
        })
        if (i + 1) % 5 == 0:
            wall = time.time() - t0
            print(f"  [{i+1}/{n}] {wall:.0f}s elapsed  buckets={counts}")

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"buckets: {counts}")
    print(f"correct%  = {100*counts['correct']/n:.0f}")
    print(f"wrong%    = {100*counts['wrong']/n:.0f}")
    print(f"quit%     = {100*counts['quit']/n:.0f}")
    print(f"timeout%  = {100*counts['timeout']/n:.0f}")

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir / 'summary.csv'}")

    # Compare to Track A Qwen3-4B T=30 cells.
    print("\n--- comparison to Track A (Qwen3-4B, T=30, n=10) ---")
    print(f"  this run (new env, n={n}):  correct={counts['correct']}/{n} ({100*counts['correct']/n:.0f}%)  "
          f"wrong={counts['wrong']} quit={counts['quit']} timeout={counts['timeout']}")
    print(f"  traj_sim_qwen3_4b T=30:    correct=2/10 (20%)  wrong=0 fmt=8 timeout=7")
    print(f"  traj_real_qwen3_4b T=30:   correct=3/10 (30%)  wrong=0 fmt=7 timeout=3")


if __name__ == "__main__":
    asyncio.run(main())
