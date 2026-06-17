"""Accuracy-vs-budget sweep (within-problem).

For a grid of FIXED budgets T, evaluate the same test problems at each T and report
accuracy. Unlike eval_prompt_salience.py (one rollout at a time), this batches all
active problems through one vLLM call per turn, so a whole budget runs in seconds.

Loads vLLM once, loops the budget grid (model resident across budgets), and prints a
final `RESULT_JSON {...}` line with per-budget accuracy/commit-rate/mean-elapsed.
"""
from __future__ import annotations
import argparse, asyncio, json, pathlib, re, sys, time

ENV_PKG = pathlib.Path(__file__).resolve().parent.parent.parent / "environments" / "interoception_countdown"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))
import interoception_countdown as env_mod          # noqa: E402
from _solver import validate_solution               # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL | re.IGNORECASE)


def make_env(args, budget, variant):
    return env_mod.load_environment(
        problems_jsonl=args.problems_jsonl,
        target_s_min=budget, target_s_max=budget, target_s_dist="uniform",
        dataset_seed=args.dataset_seed, timing_source="sim", hardware="A100_80GB",
        sim_model="Qwen3-4B", max_turns=args.max_turns, reward_shape="hyperbolic",
        attempt_bonus=0.0, enforce_max_time=False, prompt_variant=variant,
    )


def _committed(state):
    """Replicate eval_prompt_salience scoring: prefer the live-captured answer,
    else scan the final assistant turn (last-turn / single-turn commits)."""
    parsed = state.get("parsed_answer")
    if parsed is not None:
        return parsed
    traj = state.get("trajectory") or []
    last_text = (traj[-1].get("completion") or [{}])[-1].get("content", "") if traj else ""
    m = ANSWER_RE.search(last_text)
    if m:
        return m.group(1).strip()
    if "<answer>" in last_text:
        return last_text.split("<answer>", 1)[1].strip()
    return None


async def run_budget(args, llm, lora_request, sampling, budget, variant):
    env = make_env(args, budget, variant)
    ds = env.dataset
    n = min(args.num_examples, len(ds))
    rows = [ds[i] for i in range(n)]
    states = []
    for i in range(n):
        s = {"prompt": list(rows[i]["prompt"]), "info": dict(rows[i]["info"]),
             "answer": dict(rows[i]["answer"]), "example_id": i}
        await env.setup_state(s)
        states.append(s)
    messages = [list(s["prompt"]) for s in states]
    trajectories = [[] for _ in range(n)]
    active = list(range(n))

    for _turn in range(args.max_turns):
        if not active:
            break
        batch = [messages[i] for i in active]
        kw = {"lora_request": lora_request} if lora_request is not None else {}
        outputs = llm.chat(batch, sampling, use_tqdm=False, **kw)
        still = []
        for out, i in zip(outputs, active):
            text = out.outputs[0].text
            usage = type("U", (), {
                "prompt_tokens": (len(out.prompt_token_ids) if out.prompt_token_ids else 0),
                "completion_tokens": len(out.outputs[0].token_ids)})
            amsg = {"role": "assistant", "content": text}
            messages[i].append(amsg)
            trajectories[i].append({"completion": [amsg],
                                    "response": type("R", (), {"usage": usage})()})
            states[i]["trajectory"] = trajectories[i]
            env_msgs = await env.env_response(messages[i], states[i])
            if env_msgs:
                messages[i].extend(env_msgs)
            if not states[i].get("is_completed"):
                still.append(i)
        active = still

    correct = committed = 0
    elapsed_sum = 0.0
    recs = []
    for i in range(n):
        parsed = _committed(states[i])
        ic = 0
        if parsed is not None and validate_solution(
                parsed, rows[i]["answer"]["nums"], rows[i]["answer"]["target"]) is True:
            ic = 1
        emitted = bool(states[i].get("answer_emitted"))
        el = float(states[i].get("elapsed_s", 0.0))
        correct += ic; committed += int(emitted); elapsed_sum += el
        recs.append({"example_id": i, "target_s": states[i]["target_s"], "is_correct": ic,
                     "elapsed_s": el, "answer_emitted": emitted, "parsed_answer": parsed})
    return ({"budget": budget, "n": n, "accuracy": correct / n,
             "commit_rate": committed / n, "mean_elapsed": elapsed_sum / n}, recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--adapter-name", default="adapter")
    ap.add_argument("--run-label", default="model")
    ap.add_argument("--variant", default="remaining_budget")
    ap.add_argument("--budgets", default="4,8,12,18,24,32,40")
    ap.add_argument("--num-examples", type=int, default=128)
    ap.add_argument("--problems-jsonl", default="/root/data/eval.jsonl")
    ap.add_argument("--dataset-seed", type=int, default=777)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.85)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    grid = [float(x) for x in args.budgets.split(",") if x.strip()]
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    enable_lora = args.adapter_path is not None
    print(f"[acc-vs-budget] {args.run_label}: loading vLLM (lora={'on' if enable_lora else 'off'})", flush=True)
    llm = LLM(model=args.base_model, enable_lora=enable_lora,
              max_lora_rank=args.max_lora_rank if enable_lora else 16,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_fraction,
              dtype="bfloat16")
    sampling = SamplingParams(temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_completion_tokens,
                              stop=["</answer>"], include_stop_str_in_output=True)
    lora_request = LoRARequest(args.adapter_name, 1, args.adapter_path) if enable_lora else None

    summary, per_rollout = [], {}
    for T in grid:
        t0 = time.time()
        s, recs = asyncio.run(run_budget(args, llm, lora_request, sampling, T, args.variant))
        s["seconds"] = round(time.time() - t0, 1)
        summary.append(s); per_rollout[str(T)] = recs
        print(f"  T={T:>5.1f}s  acc={s['accuracy']:.3f}  commit={s['commit_rate']:.2f}  "
              f"mean_t={s['mean_elapsed']:.1f}s  ({s['seconds']}s)", flush=True)

    result = {"label": args.run_label, "variant": args.variant,
              "base_model": args.base_model, "adapter": args.adapter_path,
              "num_examples": args.num_examples, "summary": summary}
    if args.output_json:
        p = pathlib.Path(args.output_json); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            json.dump({**result, "per_rollout": per_rollout}, fh)
        print(f"  wrote {p}", flush=True)
    print("RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
