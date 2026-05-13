"""GRPO-lite RL on the multi-turn wallclock-injection rollout.

Design choices, driven by observations from the baseline sweeps:

  - Reward shape is asymmetric across four failure modes. Observed in the
    multi-model baselines:
      * "timeout"        — model loops, no <answer> tag emitted at all
      * "gave up"        — model writes "Not possible" / "None" / LaTeX
                           (parses as None — unparseable arithmetic)
      * "wrong commit"   — model writes a real expression that's wrong
                           (parses as False — math is parseable but doesn't hit target)
      * "correct"        — what we want

    The interesting case is "gave up" — Qwen2.5-3B does this 30% of the time.
    A coarse reward that lumps "gave up" with "wrong commit" gives the model
    no incentive to attempt a real expression; quitting feels equally safe.
    So we penalize them separately:

      R = 1 + α · max(0, (T - t) / T)         if correct (parses, hits target)
        = -β_wrong                             if wrong (parses, misses target)
        = -β_quit                              if gave up (unparseable)
        = -γ                                   if timeout (no <answer> emitted)

      defaults: α=1.0  β_wrong=0.1  β_quit=0.5  γ=1.5

    Ordering γ > β_quit > β_wrong > 0 says: real-but-wrong is the cheapest
    failure (good — it's the closest to success), quitting is moderately
    bad, looping until deadline is worst.

  - Group baseline (GRPO-style variance reduction) over G rollouts sharing
    the same (problem, T): advantage_i = (R_i - mean) / (std + eps).
    Different (problem, T) cells contribute on equal footing.

  - LoRA on q_proj/v_proj — enough capacity to express budget-conditioned
    behaviour without the memory cost of full fine-tuning.

  - Sample T from a wide log-uniform distribution per cell so the model
    sees pressure regimes from "essentially unlimited" to "tight" within
    one training run.

  - Use transformers .generate() for rollout AND for the gradient pass.
    This is slower than a vLLM rollout + transformers train split, but
    sidesteps the off-policy / log-prob-alignment issue and keeps the
    implementation one file.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from interoception.solver import validate_solution
from interoception.tasks import load_problems, CountdownProblem
from interoception.sim_wallclock import WallclockEstimator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    sim_model: str = "Qwen2.5-7B"
    hardware: str = "GH200"
    problems_file: str = "data/train.jsonl"
    eval_problems_file: str = "data/eval.jsonl"

    # rollout
    chunk_tokens: int = 256
    max_turns: int = 16
    temperature: float = 1.0  # higher temp for exploration during RL
    target_s_min: float = 15.0
    target_s_max: float = 120.0

    # reward
    alpha: float = 1.0       # speed bonus weight on correct
    beta_wrong: float = 0.1  # parses but wrong arithmetic
    beta_quit: float = 0.5   # emits but unparseable ("Not possible", LaTeX, etc.)
    gamma: float = 1.5       # timeout — never emitted <answer>

    # GRPO
    group_size: int = 4            # rollouts per (problem, T) cell
    problems_per_step: int = 4     # cells per gradient update
    n_steps: int = 20
    eval_every: int = 5
    eval_n_problems: int = 30

    # optim
    lr: float = 1e-5
    lora_r: int = 8
    lora_alpha: int = 16
    grad_clip: float = 1.0

    seed: int = 0
    out_dir: str = "runs/rl_qwen7b"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RolloutData:
    problem: CountdownProblem
    target_s: float
    token_ids: torch.Tensor       # [seq_len], int64
    assistant_mask: torch.Tensor  # [seq_len], 0/1
    answer: str | None
    correct: bool | None          # via validate_solution
    elapsed_s: float
    timed_out: bool
    answer_emitted: bool
    n_assistant_tokens: int
    transcript: str = ""          # rendered, for logging


# ---------------------------------------------------------------------------
# Prompt + rollout
# ---------------------------------------------------------------------------

def build_system_prompt(target_s: float) -> str:
    return (
        "You are solving a problem under a wallclock time budget.\n"
        f"Your budget is {target_s:.0f} seconds.\n"
        "Between your turns the user will inject messages of the form "
        '"[X seconds elapsed]" telling you how much wallclock time has passed.\n'
        "You should pace yourself: think when there is time, commit to an answer "
        "when time runs short. When you are ready, output your final answer "
        "inside <answer>...</answer> tags. Anything after </answer> is ignored."
    )


def build_problem_prompt(problem: CountdownProblem) -> str:
    return (
        f"Using the numbers {list(problem.numbers)} and the operators +, -, *, / "
        f"(each number used exactly once), find an expression that equals {problem.target}. "
        "Show your reasoning, then put the final expression inside <answer>...</answer>."
    )


@torch.no_grad()
def do_rollout(
    model,
    tokenizer,
    problem: CountdownProblem,
    target_s: float,
    estimator: WallclockEstimator,
    cfg: TrainConfig,
    seed: int,
) -> RolloutData:
    """Multi-turn rollout, returning the full token sequence + assistant mask.

    Tokenizer-agnostic: we let apply_chat_template handle the glue between
    turns (different per model — Qwen uses <|im_end|>, Llama uses
    <|eot_id|>, etc.), and update the token list + mask by diffing the
    re-tokenized prefix against our incrementally tracked one.
    """
    import re
    device = next(model.parameters()).device

    msgs = [
        {"role": "system", "content": build_system_prompt(target_s)},
        {"role": "user", "content": build_problem_prompt(problem)},
    ]
    initial_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    initial_ids_list = tokenizer(initial_text, add_special_tokens=False).input_ids
    token_ids_list: list[int] = list(initial_ids_list)
    mask_list: list[int] = [0] * len(token_ids_list)

    elapsed = 0.0
    answer: str | None = None
    answer_emitted = False
    timed_out = False
    transcript_chunks = [initial_text]

    for turn in range(cfg.max_turns):
        prev_ctx_tokens = len(token_ids_list)

        prompt_tensor = torch.tensor([token_ids_list], device=device)
        attn = torch.ones_like(prompt_tensor)
        out = model.generate(
            prompt_tensor,
            attention_mask=attn,
            max_new_tokens=cfg.chunk_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        new_ids = out.sequences[0, prev_ctx_tokens:].tolist()
        if len(new_ids) == 0:
            break

        token_ids_list.extend(new_ids)
        mask_list.extend([1] * len(new_ids))

        chunk_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        transcript_chunks.append(f"[ASST t{turn}] {chunk_text}\n")

        elapsed += estimator.turn_time_s(
            prev_ctx_tokens=prev_ctx_tokens,
            chunk_tokens=len(new_ids),
            include_prefill=(turn == 0),
        )

        # Did the model emit an answer? Check across all chunks generated so far.
        full_gen_text = tokenizer.decode(token_ids_list[len(initial_ids_list):], skip_special_tokens=True)
        m = re.search(r"<answer>(.*?)</answer>", full_gen_text, re.DOTALL | re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
            answer_emitted = True
            break
        if "<answer>" in full_gen_text and "</answer>" not in full_gen_text:
            tail = full_gen_text.split("<answer>", 1)[1]
            answer = tail.strip()
            answer_emitted = True
            break

        if elapsed >= target_s:
            timed_out = True
            break

        # Append assistant turn (closed) + user inject + opening of next asst turn.
        # Compute the glue by re-tokenizing the updated message list and diffing.
        msgs.append({"role": "assistant", "content": chunk_text})
        msgs.append({"role": "user", "content": f"[{elapsed:.1f}s elapsed]"})
        next_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        next_ids_list = tokenizer(next_text, add_special_tokens=False).input_ids

        # Sanity: the re-tokenized prefix should match our incremental tokens.
        # Find the longest common prefix length.
        common = 0
        for a, b in zip(token_ids_list, next_ids_list):
            if a != b:
                break
            common += 1
        if common < len(token_ids_list):
            # Tokenizer non-determinism on boundaries (rare with BPE). Rather
            # than rebuild and lose training signal, just terminate the rollout
            # here — we'll use what's already in token_ids_list (with its mask
            # intact) and skip the rest of the multi-turn loop.
            transcript_chunks.append(f"[WARN] tokenizer prefix mismatch, ending rollout early\n")
            break
        # Suffix is glue + user content + open-asst; mark as non-assistant.
        suffix = next_ids_list[common:]
        token_ids_list.extend(suffix)
        mask_list.extend([0] * len(suffix))
        transcript_chunks.append(f"[USER]  [{elapsed:.1f}s elapsed]\n")
        # Pop the speculative messages so we can re-append cleanly next iteration
        # (we already accounted for them in tokens). Wait — we WANT them in msgs
        # for the *next* re-tokenization, so leave them in place.

    correct = validate_solution(answer, problem.numbers, problem.target) if answer_emitted else None

    return RolloutData(
        problem=problem,
        target_s=target_s,
        token_ids=torch.tensor(token_ids_list, dtype=torch.long),
        assistant_mask=torch.tensor(mask_list, dtype=torch.long),
        answer=answer,
        correct=correct,
        elapsed_s=elapsed,
        timed_out=timed_out,
        answer_emitted=answer_emitted,
        n_assistant_tokens=int(sum(mask_list)),
        transcript="".join(transcript_chunks),
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(r: RolloutData, cfg: TrainConfig) -> float:
    if not r.answer_emitted:
        return -cfg.gamma
    if r.correct is True:
        speed = max(0.0, (r.target_s - r.elapsed_s) / r.target_s)
        return 1.0 + cfg.alpha * speed
    if r.correct is False:
        # parseable expression with the right multiset but wrong arithmetic —
        # the closest-to-correct failure mode
        return -cfg.beta_wrong
    # correct is None — unparseable. Either gave up ("Not possible"), wrote
    # LaTeX, or got cut off. Worse than a real-but-wrong attempt.
    return -cfg.beta_quit


# ---------------------------------------------------------------------------
# Loss (per-rollout sum log-prob of assistant tokens × advantage)
# ---------------------------------------------------------------------------

def rollout_log_prob_sum(model, rollout: RolloutData) -> torch.Tensor:
    """Forward pass over the rollout's token sequence, return the sum of
    log-probs of assistant-generated tokens. Differentiable w.r.t. model params.
    """
    device = next(model.parameters()).device
    ids = rollout.token_ids.unsqueeze(0).to(device)
    mask = rollout.assistant_mask.to(device)

    out = model(ids, use_cache=False)
    logits = out.logits[0]                                 # [seq, vocab]
    log_probs = F.log_softmax(logits[:-1], dim=-1)         # [seq-1, vocab]
    targets = ids[0, 1:]                                   # [seq-1]
    token_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # mask[t+1] tells us whether token at position t+1 is an assistant token.
    asst_mask_shifted = mask[1:].float()
    return (token_log_probs * asst_mask_shifted).sum()


# ---------------------------------------------------------------------------
# Eval (cheap: same rollout fn, no grad, no training-time augmentations)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, problems: list, estimator, cfg: TrainConfig, target_s_list=(30.0, 60.0)) -> dict:
    model.eval()
    rng = random.Random(cfg.seed)
    sample = rng.sample(problems, min(cfg.eval_n_problems, len(problems)))
    n = len(sample) * len(target_s_list)
    n_correct = n_emit = n_timeout = 0
    rewards = []
    elapsed_total = 0.0
    for prob in sample:
        for T in target_s_list:
            r = do_rollout(model, tokenizer, prob, T, estimator, cfg, seed=rng.randint(0, 1 << 30))
            if r.answer_emitted: n_emit += 1
            if r.timed_out: n_timeout += 1
            if r.correct is True: n_correct += 1
            rewards.append(compute_reward(r, cfg))
            elapsed_total += r.elapsed_s
    model.train()
    return {
        "eval_n": n,
        "correct_pct": 100 * n_correct / n,
        "emit_pct": 100 * n_emit / n,
        "timeout_pct": 100 * n_timeout / n,
        "avg_reward": sum(rewards) / len(rewards),
        "avg_elapsed_s": elapsed_total / n,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    for fld in TrainConfig.__dataclass_fields__.values():
        kw = {"default": fld.default}
        if fld.type == "bool":
            kw["action"] = "store_true"
        elif fld.type == "int":
            kw["type"] = int
        elif fld.type == "float":
            kw["type"] = float
        else:
            kw["type"] = str
        p.add_argument(f"--{fld.name.replace('_', '-')}", **kw)
    ns = p.parse_args()
    return TrainConfig(**{f.name: getattr(ns, f.name) for f in TrainConfig.__dataclass_fields__.values()})


def main():
    cfg = parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.out_dir) / "train_log.jsonl"
    log_f = log_path.open("w")

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"loading {cfg.hf_model} with LoRA r={cfg.lora_r}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )

    estimator = WallclockEstimator(hardware=cfg.hardware, sim_model=cfg.sim_model)
    train_problems = load_problems(cfg.problems_file)
    eval_problems = load_problems(cfg.eval_problems_file)
    print(f"train: {len(train_problems)} problems  eval: {len(eval_problems)} problems")

    # Initial eval
    print("\n=== initial eval ===")
    init_metrics = evaluate(model, tokenizer, eval_problems, estimator, cfg)
    for k, v in init_metrics.items():
        print(f"  {k}: {v}")
    log_f.write(json.dumps({"step": 0, "phase": "init_eval", **init_metrics}) + "\n")
    log_f.flush()

    t_train = time.time()
    for step in range(1, cfg.n_steps + 1):
        # Sample a batch of cells
        cells: list[tuple[CountdownProblem, float]] = []
        for _ in range(cfg.problems_per_step):
            prob = rng.choice(train_problems)
            log_T = rng.uniform(math.log(cfg.target_s_min), math.log(cfg.target_s_max))
            T = math.exp(log_T)
            cells.append((prob, T))

        # Rollouts per cell
        all_rollouts: list[RolloutData] = []
        rewards: list[float] = []
        cell_idx: list[int] = []
        for ci, (prob, T) in enumerate(cells):
            for g in range(cfg.group_size):
                r = do_rollout(model, tokenizer, prob, T, estimator, cfg, seed=rng.randint(0, 1 << 30))
                all_rollouts.append(r)
                rewards.append(compute_reward(r, cfg))
                cell_idx.append(ci)

        # Group-relative advantages
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        cell_idx_t = torch.tensor(cell_idx)
        advantages = torch.zeros_like(rewards_t)
        for ci in range(cfg.problems_per_step):
            mask = cell_idx_t == ci
            grp = rewards_t[mask]
            mean = grp.mean()
            std = grp.std() + 1e-6
            advantages[mask] = (grp - mean) / std

        # Compute loss across all rollouts (sequential — one fwd/bwd per rollout)
        optimizer.zero_grad()
        total_loss = 0.0
        for r, adv in zip(all_rollouts, advantages):
            if r.n_assistant_tokens == 0:
                continue
            lp = rollout_log_prob_sum(model, r)
            # Normalise by #assistant tokens so long rollouts don't dominate
            lp_norm = lp / max(r.n_assistant_tokens, 1)
            loss = -adv.to(lp_norm.device) * lp_norm
            loss = loss / len(all_rollouts)
            loss.backward()
            total_loss += float(loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip,
        )
        optimizer.step()

        # Metrics
        n_correct = sum(1 for r in all_rollouts if r.correct is True)
        n_emit = sum(1 for r in all_rollouts if r.answer_emitted)
        n_timeout = sum(1 for r in all_rollouts if r.timed_out)
        avg_R = sum(rewards) / len(rewards)
        avg_el = sum(r.elapsed_s for r in all_rollouts) / len(all_rollouts)
        avg_tok = sum(r.n_assistant_tokens for r in all_rollouts) / len(all_rollouts)
        elapsed_train = time.time() - t_train
        print(f"step {step:3d}/{cfg.n_steps}  loss={total_loss:.4f}  "
              f"avg_R={avg_R:+.3f}  correct={n_correct}/{len(all_rollouts)}  "
              f"emit={n_emit}/{len(all_rollouts)}  timeout={n_timeout}  "
              f"avg_el={avg_el:.1f}s  avg_tok={avg_tok:.0f}  "
              f"wall={elapsed_train:.0f}s")

        log_f.write(json.dumps({
            "step": step, "phase": "train",
            "loss": total_loss, "avg_reward": avg_R,
            "correct": n_correct, "emit": n_emit, "timeout": n_timeout,
            "total_rollouts": len(all_rollouts),
            "avg_elapsed_s": avg_el, "avg_tokens": avg_tok,
            "wall_s": elapsed_train,
        }) + "\n")
        log_f.flush()

        if step % cfg.eval_every == 0:
            print(f"\n=== eval at step {step} ===")
            ev = evaluate(model, tokenizer, eval_problems, estimator, cfg)
            for k, v in ev.items():
                print(f"  {k}: {v}")
            log_f.write(json.dumps({"step": step, "phase": "eval", **ev}) + "\n")
            log_f.flush()

    # Final eval (always, regardless of eval_every)
    print(f"\n=== final eval ===")
    final_metrics = evaluate(model, tokenizer, eval_problems, estimator, cfg)
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")
    log_f.write(json.dumps({"step": cfg.n_steps, "phase": "final_eval", **final_metrics}) + "\n")
    log_f.flush()

    # Save the LoRA adapter (small — a few MB)
    ckpt = Path(cfg.out_dir) / "final_model"
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f"saved adapter to {ckpt}")

    log_f.close()
    print(f"\nDone. Logs at {log_path}")


if __name__ == "__main__":
    main()
