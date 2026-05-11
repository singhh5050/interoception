"""Run a single multi-turn rollout with simulated wallclock injection.

Usage (on a GPU box with vLLM installed):

    python scripts/run_one.py \
        --hf-model Qwen/Qwen2.5-7B-Instruct \
        --sim-model Qwen2.5-7B \
        --hardware H100_SXM \
        --target-s 30 \
        --problem 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm import LLM

from interoception.rollout import run_rollout
from interoception.sim_wallclock import WallclockEstimator
from interoception.tasks import PROBLEMS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-Math-7B-Instruct",
                   help="HuggingFace checkpoint loaded by vLLM")
    p.add_argument("--sim-model", default="Qwen2.5-7B",
                   help="Model name in the hwprop catalog (must be calibrated)")
    p.add_argument("--hardware", default="H100_SXM",
                   help="Hardware name in the hwprop catalog")
    p.add_argument("--target-s", type=float, default=30.0,
                   help="Wallclock budget T in seconds")
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--max-turns", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--problem", type=int, default=0,
                   help="Index into interoception.tasks.PROBLEMS")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to dump the rollout as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    problem = PROBLEMS[args.problem]
    question = problem.to_prompt()

    print(f"=== problem: {problem.numbers} -> {problem.target} ===")
    print(f"target_s = {args.target_s}  |  hw = {args.hardware}  |  sim_model = {args.sim_model}")
    print(f"hf_model = {args.hf_model}\n")

    llm = LLM(
        model=args.hf_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    estimator = WallclockEstimator(hardware=args.hardware, sim_model=args.sim_model)

    result = run_rollout(
        llm=llm,
        question=question,
        target_s=args.target_s,
        estimator=estimator,
        chunk_tokens=args.chunk_tokens,
        max_turns=args.max_turns,
        temperature=args.temperature,
        seed=args.seed,
    )

    print("=== transcript ===")
    for t in result.turns:
        header = f"[{t.role}]"
        if t.role == "assistant":
            header += f"  ({t.output_tokens} tok, elapsed_after={t.elapsed_s_at_end:.2f}s)"
        print(header)
        print(t.content)
        print()

    print("=== summary ===")
    print(f"answer_emitted = {result.answer_emitted}")
    print(f"timed_out      = {result.timed_out}")
    print(f"elapsed_s      = {result.elapsed_s:.2f}  (budget {result.target_s:.1f})")
    print(f"output_tokens  = {result.total_output_tokens}")
    print(f"answer         = {result.answer!r}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "args": vars(args) | {"out": str(args.out)},
            "problem": {"numbers": list(problem.numbers), "target": problem.target},
            "answer": result.answer,
            "answer_emitted": result.answer_emitted,
            "timed_out": result.timed_out,
            "elapsed_s": result.elapsed_s,
            "target_s": result.target_s,
            "total_output_tokens": result.total_output_tokens,
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
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
