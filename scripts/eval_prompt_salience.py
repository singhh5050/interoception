"""Prompt-salience eval — does an aggressive budget prompt unlock T-conditioning?

Self-contained eval that runs on any host with a GPU + vLLM. Loads a base model
(optionally with a LoRA adapter), runs rollouts through the interoception env
under one or more prompt variants, and writes per-rollout JSONL for later
analysis with probe_long_T_conditioning.py.

The experiment matrix is (model, prompt_variant). Typical invocation:

  # Base model under both prompts (~10 min on A100):
  python scripts/eval_prompt_salience.py \\
      --base-model Qwen/Qwen3-4B-Instruct-2507 \\
      --variants base remaining_budget \\
      --num-examples 498 \\
      --output-dir analysis/eval_rollouts/prompt_salience

  # Trained model (LoRA on top of base) under both prompts:
  python scripts/eval_prompt_salience.py \\
      --base-model Qwen/Qwen3-4B-Instruct-2507 \\
      --adapter-path /path/to/step_500/adapter \\
      --adapter-name long-500 \\
      --variants base remaining_budget \\
      --num-examples 498

Output: <output-dir>/<run-label>_<variant>.jsonl with per-rollout records
(example_id, target_s, completion, reward, is_correct).

Each run-label corresponds to a unique (model, adapter) combo. Re-runs of an
already-completed (label, variant) skip — delete the JSONL to force a re-eval.

Dependencies on the host:
  - vllm (pip install vllm — pulls torch)
  - hwprop (pip install git+https://github.com/singhh5050/hardware-proprioception.git)
  - the interoception_countdown env package (pip install -e environments/interoception_countdown[sim])
  - the data/ and configs/ trees from this repo
"""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from typing import Any

# Ensure the env package is importable from the repo layout.
ENV_PKG = pathlib.Path(__file__).resolve().parent.parent / "environments" / "interoception_countdown"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))

import interoception_countdown as env_mod  # noqa: E402  — module exposes load_environment + InteroceptionConfig
from _solver import validate_solution      # noqa: E402  — used for in-process scoring (see run_one_rollout)

ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL | re.IGNORECASE)

DEFAULT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 128,        # per-turn chunk; matches training config
}


def make_env(args, variant: str):
    """Construct the verifiers env with the prompt-salience variant applied."""
    kwargs = dict(
        problems_jsonl=str((pathlib.Path(__file__).resolve().parent.parent / args.problems_jsonl)),
        target_s_min=args.target_s_min, target_s_max=args.target_s_max,
        target_s_dist="uniform", dataset_seed=args.dataset_seed,
        timing_source=args.timing_source, hardware=args.hardware, sim_model=args.sim_model,
        max_turns=args.max_turns, reward_shape="hyperbolic",
        attempt_bonus=0.0, enforce_max_time=False,
        prompt_variant=variant,
    )
    return env_mod.load_environment(**kwargs)


def chat_to_text(messages):
    """Render the message list back to the same form the wandb tables store
    completions in. Lets probe_long_T_conditioning.py parse it identically."""
    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks)


async def run_one_rollout(env, llm, sampling, lora_request, row, example_id: int):
    """Single rollout: setup_state, then loop env_response ↔ generate until commit
    or max_turns. Mirrors what prime-rl's orchestrator does for an eval rollout."""
    state = {
        "prompt": list(row["prompt"]),
        "info": dict(row["info"]),
        "answer": dict(row["answer"]),
        "example_id": example_id,
    }
    await env.setup_state(state)
    messages = list(state["prompt"])
    trajectory: list[dict] = []

    for turn in range(env.cfg.max_turns):
        # vLLM chat call
        kwargs = {}
        if lora_request is not None:
            kwargs["lora_request"] = lora_request
        outputs = llm.chat([messages], sampling, use_tqdm=False, **kwargs)
        text = outputs[0].outputs[0].text
        usage = type("U", (), {"prompt_tokens": outputs[0].prompt_token_ids and len(outputs[0].prompt_token_ids) or 0,
                                "completion_tokens": len(outputs[0].outputs[0].token_ids)})

        assistant_msg = {"role": "assistant", "content": text}
        messages.append(assistant_msg)

        # Build a trajectory step shaped like prime-rl's (env reads response.usage
        # for completion_tokens; we fake a tiny object with that attribute).
        step = {
            "completion": [assistant_msg],
            "response": type("R", (), {"usage": usage})(),
        }
        trajectory.append(step)
        state["trajectory"] = trajectory

        env_msgs = await env.env_response(messages, state)
        if env_msgs:
            messages.extend(env_msgs)

        if state.get("is_completed"):
            break

    # Score directly: the env's rubric funcs are wrapped with prime-rl's
    # @vf.reward decorator, which doesn't compose cleanly when invoked outside
    # the prime-rl orchestrator runtime — calls return 0 silently. We replicate
    # the env's hyperbolic c·f reward inline using the underlying solver.
    parsed = state.get("parsed_answer")
    if parsed is None:
        # env_response only captures the parsed answer mid-rollout; if the model
        # committed on the LAST turn (or single-turn rollouts), parsed_answer is
        # never set. Scan the final assistant turn so those don't get mis-scored
        # as timeouts.
        last_text = ""
        if trajectory:
            last = trajectory[-1].get("completion") or []
            if last:
                last_text = last[-1].get("content", "") or ""
        m = ANSWER_RE.search(last_text)
        if m:
            parsed = m.group(1).strip()
        elif "<answer>" in last_text:  # opened but cut off pre-</answer>
            parsed = last_text.split("<answer>", 1)[1].strip()

    is_correct = 0
    reward = 0.0
    if parsed is not None:
        ok = validate_solution(parsed, row["answer"]["nums"], row["answer"]["target"])
        if ok is True:
            is_correct = 1
            t = state.get("elapsed_s", 0.0)
            T = state.get("target_s", 1.0)
            # Hyperbolic c·f reward (matches env's correctness_with_time for
            # shape='hyperbolic' with enforce_max_time=False): f = min(1, T/t).
            f_term = 1.0 if (t <= T or t == 0) else T / t
            reward = float(f_term)  # c=1, so reward = c * f = f

    return {
        "example_id": example_id,
        "target_s": state["target_s"],
        "completion": chat_to_text(messages),
        "reward": reward,
        "is_correct": is_correct,
        "parsed_answer": parsed,
        "elapsed_s": state.get("elapsed_s", 0.0),
        "answer_emitted": bool(state.get("answer_emitted", False)),
    }


async def run_variant(args, llm, lora_request, variant: str, out_path: pathlib.Path):
    print(f"\n=== {variant!r}  →  {out_path} ===", flush=True)
    if out_path.exists() and not args.force:
        print(f"  (already exists — skipping; pass --force to re-run)")
        return

    env = make_env(args, variant)
    ds = env.dataset
    n = min(args.num_examples, len(ds))

    sampling = _sampling_params(args)
    rows = []
    t0 = time.time()
    for i in range(n):
        row = ds[i]
        rec = await run_one_rollout(env, llm, sampling, lora_request, row, example_id=i)
        rows.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == n:
            acc = sum(1 for r in rows if r["reward"] > 0) / len(rows)
            print(f"  [{i+1}/{n}]  acc={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  wrote {out_path}  ({len(rows)} rollouts)")


def _sampling_params(args):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_completion_tokens,
        # stop on the env's commit tag so we don't waste tokens past it
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="HF id or path of the base model")
    ap.add_argument("--adapter-path", default=None, help="Optional LoRA adapter dir")
    ap.add_argument("--adapter-name", default="adapter", help="Label for the LoRA in vLLM")
    ap.add_argument("--run-label", default=None, help="Filename label (default: 'base' or adapter-name)")
    ap.add_argument("--variants", nargs="+", default=["base", "remaining_budget"])
    ap.add_argument("--num-examples", type=int, default=498)
    ap.add_argument("--output-dir", default="analysis/eval_rollouts/prompt_salience")
    ap.add_argument("--problems-jsonl", default="data/eval.jsonl")
    ap.add_argument("--dataset-seed", type=int, default=777)
    ap.add_argument("--target-s-min", type=float, default=1.0)
    ap.add_argument("--target-s-max", type=float, default=40.0)
    ap.add_argument("--target-s-list", default=None,
                    help="Comma-separated list of fixed budget T values (e.g. '2,5,10,20,30,40'). "
                         "If set, runs one probe per budget per variant with target_s_min=max=T. "
                         "Output filename includes the budget: <label>_T<XX>_<variant>.jsonl. "
                         "vLLM is loaded ONCE and reused across all budget runs (saves cold-start).")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=128)
    ap.add_argument("--timing-source", default="sim", choices=["sim", "real"])
    ap.add_argument("--hardware", default="A100_80GB")
    ap.add_argument("--sim-model", default="Qwen3-4B")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.85)
    # Training used seq_len=2048 but accumulated turns (system + problem + ~10
    # `[Xs elapsed]` injections + per-turn completions) can run past that; the
    # in-training orchestrator truncates, but our simple rollout loop doesn't.
    # 4096 leaves comfortable headroom on the A100 40GB.
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = ap.parse_args()

    run_label = args.run_label or (args.adapter_name if args.adapter_path else "base")
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import vLLM — it's heavy and only needed once we know we're running.
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    enable_lora = args.adapter_path is not None
    print(f"loading vLLM  base={args.base_model}  lora={'on' if enable_lora else 'off'}")
    llm = LLM(
        model=args.base_model,
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank if enable_lora else 16,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        dtype="bfloat16",
    )
    lora_request = None
    if enable_lora:
        lora_request = LoRARequest(args.adapter_name, 1, args.adapter_path)
        print(f"  LoRA: {args.adapter_path}  (lora_int_id=1)")

    # If target-s-list is set, sweep over fixed budgets (vLLM stays loaded across
    # all of them — this is the big wallclock win, since cold-start dominates).
    if args.target_s_list:
        budgets = [float(b.strip()) for b in args.target_s_list.split(",") if b.strip()]
        print(f"\nbudget sweep: {len(budgets)} budgets × {len(args.variants)} variants "
              f"= {len(budgets) * len(args.variants)} runs (vLLM reused throughout)")
        for budget in budgets:
            # Snapshot original args, override min/max so the env's per-problem
            # rng.uniform(min, max) collapses to a constant.
            args.target_s_min = budget
            args.target_s_max = budget
            budget_tag = f"T{int(round(budget)):02d}"
            for variant in args.variants:
                out_path = out_dir / f"{run_label}_{budget_tag}_{variant}.jsonl"
                asyncio.run(run_variant(args, llm, lora_request, variant, out_path))
    else:
        # Original behavior: one variant per run, random T from [min, max].
        for variant in args.variants:
            out_path = out_dir / f"{run_label}_{variant}.jsonl"
            asyncio.run(run_variant(args, llm, lora_request, variant, out_path))


if __name__ == "__main__":
    main()
