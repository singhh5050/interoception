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
import random
from pathlib import Path

from vllm import LLM

from interoception.rollout import run_rollout
from interoception.sim_wallclock import WallclockEstimator
from interoception.solver import validate_solution
from interoception.tasks import PROBLEMS, load_problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--sim-model", default="Qwen2.5-14B")
    p.add_argument("--hardware", default="H100_SXM")
    p.add_argument("--target-s", type=float, nargs="+", default=[15.0, 30.0, 60.0, 120.0])
    # Two ways to pick problems:
    #   (a) --problems-file path: load a JSONL written by scripts/build_dataset.py,
    #       then randomly sample --num-problems of them with --sample-seed.
    #   (b) --problems i j k ...: indices into the hardcoded `PROBLEMS` list
    #       (used by the original exploration sweeps).
    p.add_argument("--problems-file", type=Path, default=None,
                   help="JSONL file produced by scripts/build_dataset.py")
    p.add_argument("--num-problems", type=int, default=50,
                   help="how many to sample from --problems-file (ignored if --problems given)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--problems", type=int, nargs="+", default=None,
                   help="indices into the hardcoded PROBLEMS list (alternative to --problems-file)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--max-turns", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--prompt-style", choices=["default", "medium", "strong"], default="default")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="vLLM max context. Default: let vLLM use the model's max_position_embeddings.")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def resolve_problems(args) -> list:
    """Resolve --problems-file / --problems into a list of (idx, CountdownProblem).
    idx is a stable identifier used to name output files (the row index within
    the JSONL, or the index into PROBLEMS for the hardcoded path)."""
    if args.problems_file is not None:
        all_problems = load_problems(args.problems_file)
        rng = random.Random(args.sample_seed)
        idxs = list(range(len(all_problems)))
        rng.shuffle(idxs)
        idxs = idxs[: args.num_problems]
        return [(i, all_problems[i]) for i in idxs]
    if args.problems is not None:
        return [(i, PROBLEMS[i]) for i in args.problems]
    # Default: use all hardcoded problems
    return [(i, PROBLEMS[i]) for i in range(len(PROBLEMS))]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    problems = resolve_problems(args)
    print(f"loading vLLM model: {args.hf_model}")
    llm_kwargs: dict = dict(
        model=args.hf_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    estimator = WallclockEstimator(hardware=args.hardware, sim_model=args.sim_model)

    summary_path = args.out_dir / "summary.csv"
    rows: list[dict] = []
    cells = list(itertools.product(problems, args.target_s, args.seeds))

    print(f"\nrunning {len(cells)} rollouts ({len(problems)} problems × {len(args.target_s)} budgets × {len(args.seeds)} seeds)")
    print(f"sim_model = {args.sim_model}, hardware = {args.hardware}\n")

    for i, ((prob_idx, problem), target_s, seed) in enumerate(cells, 1):
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
            prompt_style=args.prompt_style,
        )

        correct = validate_solution(result.answer, problem.numbers, problem.target)
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
            "solution_count": problem.solution_count,
            "difficulty_band": _difficulty_band(problem.solution_count),
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


def _difficulty_band(solution_count: int) -> str:
    # Mirrors scripts/build_dataset.py's bucket_key. Kept inline so a sweep can
    # be re-summarized later without re-importing the builder.
    if solution_count == 0:
        return "n/a"
    if solution_count <= 3:
        return "rare"
    if solution_count <= 12:
        return "med"
    return "common"


def _print_group(label: str, rs: list[dict]) -> None:
    n = len(rs)
    if n == 0:
        return
    emit_pct = 100 * sum(1 for r in rs if r["answer_emitted"]) / n
    to_pct = 100 * sum(1 for r in rs if r["timed_out"]) / n
    scored = [r for r in rs if r["correct"] is not None]
    corr_pct = (100 * sum(1 for r in scored if r["correct"]) / len(scored)) if scored else float("nan")
    avg_el = sum(r["elapsed_s"] for r in rs) / n
    avg_tok = sum(r["total_output_tokens"] for r in rs) / n
    print(f"  {label:>14} {n:>4} {emit_pct:>5.0f}% {to_pct:>5.0f}% {corr_pct:>5.0f}% {avg_el:>12.2f} {avg_tok:>11.0f}")


def print_summary(rows: list[dict]) -> None:
    header = f"  {'group':>14} {'n':>4} {'emit%':>6} {'to%':>6} {'corr%':>6} {'avg_elapsed':>12} {'avg_tokens':>11}"
    print("\n=== summary by T ===")
    print(header)
    by_T: dict[float, list[dict]] = {}
    for r in rows:
        by_T.setdefault(r["target_s"], []).append(r)
    for T in sorted(by_T):
        _print_group(f"T={int(T)}s", by_T[T])

    has_difficulty = any(r["solution_count"] > 0 for r in rows)
    if has_difficulty:
        print("\n=== summary by difficulty band ===")
        print(header)
        for band in ("rare", "med", "common"):
            _print_group(band, [r for r in rows if r["difficulty_band"] == band])

        print("\n=== T × difficulty ===")
        print(header)
        for T in sorted(by_T):
            for band in ("rare", "med", "common"):
                cell = [r for r in by_T[T] if r["difficulty_band"] == band]
                _print_group(f"T={int(T)}|{band}", cell)


if __name__ == "__main__":
    main()
