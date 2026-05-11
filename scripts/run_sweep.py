"""Run a sweep of rollouts over (problem, target_s) and dump the results.

Loads the vLLM model once and reuses it across rollouts. Each rollout is
written to its own JSON under --out-dir. A summary CSV is also written.

Usage:

    python scripts/run_sweep.py \
        --hf-model Qwen/Qwen2.5-7B-Instruct \
        --sim-model Qwen2.5-7B \
        --hardware H100_SXM \
        --target-s 15 30 60 120 \
        --problems 0 1 2 3 \
        --seeds 0 1 \
        --out-dir runs/sweep_qwen7b_h100
"""
from __future__ import annotations

import argparse
import csv
import json
import itertools
from pathlib import Path

from vllm import LLM

from interoception.rollout import run_rollout
from interoception.sim_wallclock import WallclockEstimator
from interoception.tasks import PROBLEMS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--sim-model", default="Qwen2.5-7B")
    p.add_argument("--hardware", default="H100_SXM")
    p.add_argument("--target-s", type=float, nargs="+", default=[15.0, 30.0, 60.0, 120.0])
    p.add_argument("--problems", type=int, nargs="+", default=list(range(len(PROBLEMS))))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--max-turns", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def check_answer(problem, answer: str | None) -> bool | None:
    """Return True/False if we can evaluate, None if we can't parse."""
    if answer is None:
        return None
    expr = answer.strip().strip("`").strip()
    allowed = set("0123456789+-*/(). ")
    if not expr or any(ch not in allowed for ch in expr):
        return None
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(value, (int, float)):
        return None
    return abs(value - problem.target) < 1e-6


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading vLLM model: {args.hf_model}")
    llm = LLM(
        model=args.hf_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_model_len=8192,
    )
    estimator = WallclockEstimator(hardware=args.hardware, sim_model=args.sim_model)

    summary_path = args.out_dir / "summary.csv"
    rows: list[dict] = []
    cells = list(itertools.product(args.problems, args.target_s, args.seeds))

    print(f"\nrunning {len(cells)} rollouts")
    print(f"sim_model = {args.sim_model}, hardware = {args.hardware}\n")

    for i, (prob_idx, target_s, seed) in enumerate(cells, 1):
        problem = PROBLEMS[prob_idx]
        question = problem.to_prompt()

        print(f"[{i}/{len(cells)}] problem={prob_idx} T={target_s:.0f}s seed={seed} ...", end=" ", flush=True)

        result = run_rollout(
            llm=llm,
            question=question,
            target_s=target_s,
            estimator=estimator,
            chunk_tokens=args.chunk_tokens,
            max_turns=args.max_turns,
            temperature=args.temperature,
            seed=seed,
        )

        correct = check_answer(problem, result.answer)
        print(
            f"emitted={result.answer_emitted} timeout={result.timed_out} "
            f"elapsed={result.elapsed_s:.1f}s tokens={result.total_output_tokens} "
            f"answer={result.answer!r} correct={correct}"
        )

        run_path = args.out_dir / f"p{prob_idx}_T{int(target_s)}_s{seed}.json"
        run_path.write_text(json.dumps({
            "args": vars(args) | {"out_dir": str(args.out_dir)},
            "problem_idx": prob_idx,
            "problem": {"numbers": list(problem.numbers), "target": problem.target},
            "target_s": target_s,
            "seed": seed,
            "answer": result.answer,
            "answer_emitted": result.answer_emitted,
            "timed_out": result.timed_out,
            "elapsed_s": result.elapsed_s,
            "total_output_tokens": result.total_output_tokens,
            "correct": correct,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "output_tokens": t.output_tokens,
                    "elapsed_s_at_end": t.elapsed_s_at_end,
                }
                for t in result.turns
            ],
        }, indent=2, default=str))

        rows.append({
            "problem_idx": prob_idx,
            "numbers": "-".join(str(n) for n in problem.numbers),
            "target": problem.target,
            "target_s": target_s,
            "seed": seed,
            "answer": result.answer,
            "answer_emitted": result.answer_emitted,
            "timed_out": result.timed_out,
            "elapsed_s": round(result.elapsed_s, 3),
            "total_output_tokens": result.total_output_tokens,
            "n_assistant_turns": sum(1 for t in result.turns if t.role == "assistant"),
            "correct": correct,
            "run_file": run_path.name,
        })

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rollouts to {args.out_dir}")
    print(f"summary: {summary_path}")
    print_summary(rows)


def print_summary(rows: list[dict]) -> None:
    print("\n=== summary ===")
    by_T: dict[float, list[dict]] = {}
    for r in rows:
        by_T.setdefault(r["target_s"], []).append(r)
    print(f"{'T (s)':>6} {'n':>3} {'emit%':>6} {'to%':>6} {'corr%':>6} {'avg_elapsed':>12} {'avg_tokens':>11}")
    for T in sorted(by_T):
        rs = by_T[T]
        n = len(rs)
        emit_pct = 100 * sum(1 for r in rs if r["answer_emitted"]) / n
        to_pct = 100 * sum(1 for r in rs if r["timed_out"]) / n
        scored = [r for r in rs if r["correct"] is not None]
        corr_pct = (100 * sum(1 for r in scored if r["correct"]) / len(scored)) if scored else float("nan")
        avg_el = sum(r["elapsed_s"] for r in rs) / n
        avg_tok = sum(r["total_output_tokens"] for r in rs) / n
        print(f"{T:>6.0f} {n:>3} {emit_pct:>5.0f}% {to_pct:>5.0f}% {corr_pct:>5.0f}% {avg_el:>12.2f} {avg_tok:>11.0f}")


if __name__ == "__main__":
    main()
