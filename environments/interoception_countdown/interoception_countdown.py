"""Countdown under a wallclock budget — verifiers MultiTurnEnv.

The model receives a Countdown problem and a budget T. Between its turns, the
environment injects `[X seconds elapsed]` messages. The model wins by emitting
`<answer>EXPR</answer>` where EXPR uses each input number exactly once with
+/-/*/÷ and evaluates to the target.

Two timing sources are supported (configurable):
  - "real" (default): X is the actual per-turn wallclock measured by the
    framework in state["timing"].model.spans. Most honest for eval.
  - "sim":  X is hwprop.simulate_latency(hardware, sim_model, ...). Useful for
    RL because it's deterministic (cleaner GRPO baselines), supports
    counterfactual hardware ("train as if deployed on H100"), and decouples
    perceived-budget from actual training wall time.

Reward decomposes into four buckets, mirroring the asymmetric shape from the
original rl_train.py:
    correct (parses, hits target)    : +1 + alpha * speed_bonus
    wrong   (parses, misses target)  : -beta_wrong   (closest to success)
    quit    (unparseable / "None")   : -beta_quit
    timeout (no <answer> emitted)    : -gamma        (worst)

Ordering gamma > beta_quit > beta_wrong > 0 says: real-but-wrong is the cheapest
failure (it's the closest to success), quitting is moderately bad, looping
until deadline is worst.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset
from pydantic import BaseModel
from typing import Literal

from _solver import validate_solution, is_parseable_arithmetic

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


class InteroceptionConfig(BaseModel):
    # Budget range (sampled log-uniform ONCE per dataset row at load time;
    # see _load_dataset). All GRPO rollouts of one problem share its T — the
    # advantage estimate (r_i - mean_group)/std_group is only meaningful when
    # the group sees the same input distribution. Sampling per-rollout (the
    # prior behavior) gave each rollout in a group a different T and a different
    # system prompt, poisoning the advantage signal.
    target_s_min: float = 15.0
    target_s_max: float = 120.0
    # Budget sampling distribution. "uniform" samples T ~ U(min, max);
    # "log" samples log-uniform. Kanishk's call: log-uniform biases the model
    # too hard toward small budgets, so v2 uses uniform.
    target_s_dist: Literal["uniform", "log"] = "uniform"
    # Seed for per-row target_s assignment. Two phase1 seeds (0, 1) get
    # different T-assignments across the dataset — the primary source of
    # cross-seed variance we want.
    dataset_seed: int = 0
    # Rollout shape.
    max_turns: int = 16
    # Reward shape — see `correctness_with_time` for details.
    #   "hyperbolic"  : c · 1 if t≤T,   c · T/min(t,max_t) if t>T   (smooth gradient density past T)
    #   "exponential" : c · 1 if t≤T,   c · exp(-α·(t-T)/T) if t>T  (steeper penalty for overshoot)
    #   "asymmetric"  : old 4-bucket reward from rl_train.py (kept for fallback / comparison)
    reward_shape: Literal["hyperbolic", "exponential", "asymmetric"] = "hyperbolic"
    # Exponential decay coefficient (only used when reward_shape="exponential").
    reward_alpha: float = 1.0
    # Control flags for the f(t,T) ablations (Kanishk's 2026-05-24 controls thread):
    #   reward_time_term=False -> f(t,T)=1 always, so the reward collapses to pure
    #     correctness c (no timing term). Used by control A (no time reward) and
    #     control B. The f_term diagnostic is still logged so we can SEE whether the
    #     model paces even without being rewarded for it.
    #   inject_elapsed=False -> the env never injects "[X seconds elapsed]" and the
    #     system prompt omits the injection mechanism (still states the budget). Used
    #     by control B together with max_turns=1 (single-turn, no time signal at all).
    reward_time_term: bool = True
    inject_elapsed: bool = True
    # Attempt bonus: reward for emitting *any* parseable arithmetic expression,
    # whether or not it satisfies the multiset/target constraints.
    # LEGACY / OFF BY DEFAULT: this is the v1 confound that reward-hacked Qwen2.5-3B
    # (the policy maximized "emit any arithmetic" and ignored correctness). Kanishk
    # dropped it. Default 0.0 so a config that omits it can't silently reintroduce
    # the confound; set explicitly (>0) only for legacy sweeps.
    attempt_bonus: float = 0.0
    # Max-time multiplier. When enforce_max_time=True the env cuts off the rollout
    # at `multiplier * target_s`. v2 sets enforce_max_time=False: no time-based
    # cutoff at all (only max_turns / seq_len bound the rollout), and the
    # hyperbolic reward becomes pure c·min(1, T/t) with no decay cap.
    max_time_multiplier: float = 5.0
    enforce_max_time: bool = True
    # Asymmetric-shape-only weights (mirror rl_train.py defaults).
    alpha: float = 1.0       # speed bonus on correct
    beta_wrong: float = 0.1  # penalty when parseable but wrong
    beta_quit: float = 0.5   # penalty when unparseable (gave up, LaTeX, etc.)
    gamma: float = 1.5       # penalty when no <answer> emitted at all
    # Dataset — JSONL produced by scripts/build_dataset.py.
    problems_jsonl: str = "data/train.jsonl"
    # Timing source for the [Xs elapsed] signal.
    #   "real" — actual per-turn wallclock from state["timing"].model.spans (default; eval-honest).
    #   "sim"  — hwprop.simulate_latency(...). Deterministic; supports counterfactual hardware.
    #            Requires `pip install hwprop` (optional dep).
    timing_source: Literal["real", "sim"] = "real"
    # Only used when timing_source="sim" — names in the hwprop catalog.
    # The sim assumes prefix caching is enabled (only charges prefill on turn 1).
    hardware: str = "A100_80GB"
    sim_model: str = "Qwen3-4B"


def _build_prompt(nums: list[int], target: int) -> str:
    return (
        f"Using the numbers {nums} and the operators +, -, *, / "
        f"(each number used exactly once), find an expression that equals {target}. "
        "Show your reasoning, then put the final expression inside <answer>...</answer>."
    )


def _build_system_prompt(target_s: float, inject_elapsed: bool = True) -> str:
    base = (
        "You are solving a problem under a wallclock time budget.\n"
        f"Your budget is {target_s:.0f} seconds.\n"
    )
    if inject_elapsed:
        return base + (
            'Between your turns the user will inject messages of the form "[X seconds elapsed]" '
            "telling you how much wallclock time has passed.\n"
            "You should pace yourself: think when there is time, commit to an answer when "
            "time runs short. When you are ready, output your final answer inside "
            "<answer>...</answer> tags. Anything after </answer> is ignored."
        )
    # Control B: no elapsed signal. State the budget, but don't describe an
    # injection mechanism that won't fire (single-turn, no [Xs elapsed]).
    return base + (
        "Work within this budget. When you are ready, output your final answer inside "
        "<answer>...</answer> tags. Anything after </answer> is ignored."
    )


class CountdownTimeBudgetEnv(vf.MultiTurnEnv):
    def __init__(self, cfg: InteroceptionConfig, **kwargs):
        super().__init__(max_turns=cfg.max_turns, **kwargs)
        self.cfg = cfg

    async def setup_state(self, state: vf.State) -> vf.State | None:
        # target_s resolution order:
        #   1. pre-set state["target_s"] (eval pins this for fixed-budget cells)
        #   2. state["info"]["target_s"] (set by _load_dataset, one T per problem)
        # Reading from info is the GRPO-correctness path — all rollouts of one
        # problem share T, so the group advantage estimate is meaningful. If
        # info["target_s"] is missing, fail loud: a per-rollout fallback would
        # silently re-introduce the original group-baseline poisoning bug.
        if "target_s" not in state:
            info = state.get("info") or {}
            t = info.get("target_s")
            if t is None:
                raise RuntimeError(
                    "info['target_s'] missing — every row must be assigned T at "
                    "load time (see _load_dataset) for GRPO group-baseline "
                    "correctness; per-rollout sampling poisons the group advantage."
                )
            state["target_s"] = t
        state["elapsed_s"] = 0.0
        state["answer_emitted"] = False
        state["parsed_answer"] = None
        # Stash a tiny dict of cfg fields the reward functions need to read.
        # Reward fns are pure functions and don't have access to `self`/cfg directly.
        state["_cfg"] = {
            "reward_shape": self.cfg.reward_shape,
            "reward_alpha": self.cfg.reward_alpha,
            "max_time_multiplier": self.cfg.max_time_multiplier,
            "enforce_max_time": self.cfg.enforce_max_time,
            "attempt_bonus": self.cfg.attempt_bonus,
            "reward_time_term": self.cfg.reward_time_term,
            # Needed so the reward-time elapsed finalizer (_finalize_elapsed) can
            # recompute sim latency over the FULL trajectory, incl. the final turn.
            "timing_source": self.cfg.timing_source,
            "hardware": self.cfg.hardware,
            "sim_model": self.cfg.sim_model,
        }

        # Inject the budget-aware system prompt into the prompt list if not already there.
        prompt = state.get("prompt") or []
        if not prompt or prompt[0].get("role") != "system":
            state["prompt"] = [
                {"role": "system", "content": _build_system_prompt(state["target_s"], self.cfg.inject_elapsed)},
                *prompt,
            ]
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        # The model just produced a turn; the latest trajectory step has its tokens.
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return []
        last = trajectory[-1]
        last_completion = last.get("completion") or []
        chunk_text = last_completion[-1].get("content", "") if last_completion else ""

        if self.cfg.timing_source == "real":
            # Use the framework's recorded model-turn wallclock as the elapsed signal.
            # state["timing"].model.spans is populated by the rollout loop with perf_counter
            # timestamps around each get_model_response call. The most recent span is the
            # turn we're reacting to. Most honest for eval — no simulator approximation.
            timing = state.get("timing")
            spans = timing.model.spans if timing is not None else []
            if spans:
                state["elapsed_s"] += spans[-1].end - spans[-1].start
        else:
            # Sim: deterministic per-turn cost via hwprop. Useful for RL where we want
            # low-variance group baselines + counterfactual hardware. The chunk token
            # count comes from response.usage (step["tokens"] is None for the OpenAI
            # chat-completions client). prev_ctx_tokens is summed from past prompt sizes.
            from hwprop.simulator import simulate_latency  # lazy: only import if used
            prev_ctx_tokens, chunk_tokens = _step_usage(last)
            if chunk_tokens > 0:
                r = simulate_latency(
                    self.cfg.hardware, self.cfg.sim_model,
                    prompt_len=max(prev_ctx_tokens, 1), decode_steps=chunk_tokens,
                )
                include_prefill = len(trajectory) == 1
                state["elapsed_s"] += (
                    (r.prefill_time_s if include_prefill else 0.0) + r.total_decode_time_s
                )

        # Did the model commit?
        m = ANSWER_RE.search(chunk_text)
        if m:
            state["answer_emitted"] = True
            state["parsed_answer"] = m.group(1).strip()
            # final_env_response short-circuits via the base-class
            # has_final_env_response @vf.stop; is_completed=True is belt-and-suspenders
            # in case stop-hook ordering changes upstream.
            state["final_env_response"] = [
                {"role": "user", "content": f"[committed at {state['elapsed_s']:.1f}s]"}
            ]
            state["is_completed"] = True
            return []
        # Handle the "model emitted <answer>...</answer>" was cut off mid-tag case:
        if "<answer>" in chunk_text and "</answer>" not in chunk_text:
            tail = chunk_text.split("<answer>", 1)[1].strip()
            state["answer_emitted"] = True
            state["parsed_answer"] = tail
            state["final_env_response"] = [
                {"role": "user", "content": f"[committed at {state['elapsed_s']:.1f}s]"}
            ]
            state["is_completed"] = True
            return []

        # Optional hard cutoff at max_time = multiplier × target_s. v2 disables
        # this (enforce_max_time=False): rollouts end only via <answer> or
        # max_turns/seq_len, never a T-dependent time wall. The T-dependent wall
        # was a confound — at small T it forced eviction before the model could
        # commit, which looked like "pacing" but was just truncation.
        if self.cfg.enforce_max_time:
            max_time = self.cfg.max_time_multiplier * state["target_s"]
            if state["elapsed_s"] >= max_time:
                state["final_env_response"] = [
                    {"role": "user", "content": f"[hard cutoff at {state['elapsed_s']:.1f}s (max_time={max_time:.0f}s)]"}
                ]
                state["is_completed"] = True
                return []

        # Control B: no time signal. elapsed_s is still tracked above for the
        # f_term / elapsed_over_target diagnostics, but the model never sees it.
        # (Paired with max_turns=1, the rollout is single-turn anyway.)
        if not self.cfg.inject_elapsed:
            return []

        return [{"role": "user", "content": f"[{state['elapsed_s']:.1f}s elapsed]"}]

    @vf.stop
    async def budget_exhausted(self, state: vf.State) -> bool:
        # Only fires when enforce_max_time=True. v2 disables the time wall, so
        # this always returns False and rollouts end via <answer> or max_turns.
        if not self.cfg.enforce_max_time:
            return False
        max_time = self.cfg.max_time_multiplier * state.get("target_s", float("inf"))
        return state.get("elapsed_s", 0.0) >= max_time


# --- Reward functions.
# `correctness_with_time` + `parseable_bonus` are the new shapes (hyperbolic/exponential).
# `correct_with_speed_bonus` + 4 bucket penalties are the original asymmetric shape (kept for fallback).
# `is_correct/wrong/quit/timeout/parseable` are 0/1 diagnostic metrics (weight=0 in rubric).

def _last_assistant_text(state: vf.State) -> str:
    """Text of the model's final turn, read straight from the trajectory.

    `env_response` (which captures the committed answer live) fires *between*
    turns, so it never runs after the last/only turn — single-turn rollouts
    (max_turns=1) and multi-turn rollouts that commit on the final allowed turn
    would otherwise be scored as timeouts. Reading the trajectory at reward time
    makes scoring independent of that hook."""
    traj = state.get("trajectory") or []
    if not traj:
        return ""
    completion = traj[-1].get("completion") or []
    if not completion:
        return ""
    return completion[-1].get("content", "") or ""


def _committed_answer(state: vf.State) -> str | None:
    """The answer the model committed, for scoring. Prefers the value captured
    live in env_response (multi-turn mid-rollout commit); falls back to scanning
    the final assistant turn so single-turn / last-turn commits aren't mis-scored
    as timeouts."""
    if state.get("parsed_answer") is not None:
        return state["parsed_answer"]
    text = _last_assistant_text(state)
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    if "<answer>" in text:  # opened but cut off before </answer>
        return text.split("<answer>", 1)[1].strip()
    return None


def _step_usage(step) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) for one trajectory step.

    The OpenAI chat-completions client leaves step['tokens'] as None, so the live
    token counts come from the response usage object; fall back to token_usage /
    tokens dicts for other client shapes."""
    usage = getattr(step.get("response"), "usage", None)
    if usage is not None and getattr(usage, "completion_tokens", None) is not None:
        return int(getattr(usage, "prompt_tokens", 0) or 0), int(usage.completion_tokens)
    tu = step.get("token_usage")
    if isinstance(tu, dict) and tu.get("output_tokens") is not None:
        return int(tu.get("input_tokens") or 0), int(tu["output_tokens"])
    tok = step.get("tokens")
    if tok is not None:
        out = getattr(tok, "output_tokens", None)
        inp = getattr(tok, "input_tokens", None)
        if out is None and isinstance(tok, dict):
            out, inp = tok.get("output_tokens"), tok.get("input_tokens")
        if out is not None:
            return int(inp or 0), int(out)
    return 0, 0


def _finalize_elapsed(state: vf.State) -> float:
    """Total elapsed wallclock across ALL turns, computed at scoring time from the
    trajectory. env_response only advances elapsed_s *between* turns, so it never
    accounts for the final/only turn (single-turn rollouts, or multi-turn commits on
    the last allowed turn) — leaving the reward-time time term and the
    f_term/elapsed_over_target diagnostics stale. Recomputing here closes that gap,
    mirroring env_response's per-turn cost exactly (prefill on turn 0 only)."""
    cfg = state.get("_cfg") or {}
    if cfg.get("timing_source", "real") == "real":
        timing = state.get("timing")
        spans = timing.model.spans if timing is not None else []
        return float(sum(s.end - s.start for s in spans))
    from hwprop.simulator import simulate_latency
    total = 0.0
    for i, step in enumerate(state.get("trajectory") or []):
        pt, ct = _step_usage(step)
        if ct > 0:
            r = simulate_latency(cfg.get("hardware", "A100_80GB"), cfg.get("sim_model", "Qwen3-4B"),
                                 prompt_len=max(pt, 1), decode_steps=ct)
            total += (r.prefill_time_s if i == 0 else 0.0) + r.total_decode_time_s
    return total


def _elapsed(state: vf.State) -> float:
    """Finalized elapsed, computed once per rollout and cached on state."""
    if "_elapsed_final" not in state:
        state["_elapsed_final"] = _finalize_elapsed(state)
    return state["_elapsed_final"]


def _bucket(state: vf.State, answer) -> str:
    """Classify the rollout outcome. answer is the dataset row's answer dict."""
    parsed = _committed_answer(state)
    if parsed is None:
        return "timeout"  # never emitted <answer> anywhere
    ok = validate_solution(parsed, answer["nums"], answer["target"])
    if ok is True:
        return "correct"
    if ok is False:
        return "wrong"
    return "quit"  # parsed present but unparseable


# --- New reward shapes (hyperbolic / exponential) ---

def _time_factor(t: float, T: float, shape: str, alpha: float,
                 max_time_multiplier: float, enforce_max_time: bool = True) -> float:
    """Pure time term f(t,T), independent of correctness: 1.0 in budget, decayed past T.

    With no time-based cutoff (v2/controls), the hyperbolic factor is pure
    min(1, T/t) — no decay cap, t bounded naturally by max_turns/seq_len."""
    if t <= T:
        return 1.0
    capped_t = t if not enforce_max_time else min(t, max_time_multiplier * T)
    if shape == "hyperbolic":
        # T/t at t=T is 1, at t=2T is 0.5, at t=4T is 0.25, ...
        return T / capped_t
    elif shape == "exponential":
        # exp(-α(t-T)/T) at t=T is 1, at t=2T is exp(-α)≈0.37 (α=1).
        return math.exp(-alpha * (capped_t - T) / T)
    else:
        raise ValueError(f"Unknown reward shape: {shape!r}")


def _correctness_term(state: vf.State, answer, shape: str, alpha: float,
                      max_time_multiplier: float, enforce_max_time: bool = True,
                      reward_time_term: bool = True) -> float:
    """Reward for a correct answer, decayed by overshoot beyond target_s.
    Returns 0 if not correct. With reward_time_term=False (controls A/B) the
    time term is dropped entirely: a correct answer is worth 1.0 regardless of t."""
    if _bucket(state, answer) != "correct":
        return 0.0
    if not reward_time_term:
        return 1.0  # f(t,T) ≡ 1: pure correctness reward
    return _time_factor(_elapsed(state), state["target_s"], shape, alpha,
                        max_time_multiplier, enforce_max_time)


@vf.reward
def correctness_with_time(state, answer, **_) -> float:
    """Main reward for hyperbolic/exponential shapes. Rubric weight applies to this.
    For shape='asymmetric', returns 0 — the old `correct_with_speed_bonus` takes over."""
    cfg = state.get("_cfg")  # injected in setup_state below
    if cfg is None or cfg["reward_shape"] == "asymmetric":
        return 0.0
    return _correctness_term(
        state, answer,
        shape=cfg["reward_shape"],
        alpha=cfg["reward_alpha"],
        max_time_multiplier=cfg["max_time_multiplier"],
        enforce_max_time=cfg.get("enforce_max_time", True),
        reward_time_term=cfg.get("reward_time_term", True),
    )


@vf.reward
def parseable_bonus(state, answer, **_) -> float:
    """0.05 if the model emitted any parseable arithmetic expression (regardless
    of multiset/target). Pulls the model from quitting toward attempts.

    Note: we use raw 1.0 here; the 0.05 weight is applied at the Rubric level
    via cfg.attempt_bonus."""
    return 1.0 if is_parseable_arithmetic(_committed_answer(state)) else 0.0


# --- Original asymmetric reward (kept for reward_shape="asymmetric") ---

@vf.reward
def correct_with_speed_bonus(state, answer, **_) -> float:
    """Asymmetric-shape only. Returns 0 unless reward_shape=='asymmetric' AND correct.
    Rubric weight = cfg.alpha."""
    cfg = state.get("_cfg")
    if cfg is None or cfg["reward_shape"] != "asymmetric":
        return 0.0
    if _bucket(state, answer) != "correct":
        return 0.0
    speed = max(0.0, (state["target_s"] - _elapsed(state)) / state["target_s"])
    return 1.0 + speed


@vf.reward
def wrong_penalty(state, answer, **_) -> float:
    cfg = state.get("_cfg")
    if cfg is None or cfg["reward_shape"] != "asymmetric":
        return 0.0
    return -1.0 if _bucket(state, answer) == "wrong" else 0.0


@vf.reward
def quit_penalty(state, answer, **_) -> float:
    cfg = state.get("_cfg")
    if cfg is None or cfg["reward_shape"] != "asymmetric":
        return 0.0
    return -1.0 if _bucket(state, answer) == "quit" else 0.0


@vf.reward
def timeout_penalty(state, answer, **_) -> float:
    cfg = state.get("_cfg")
    if cfg is None or cfg["reward_shape"] != "asymmetric":
        return 0.0
    return -1.0 if _bucket(state, answer) == "timeout" else 0.0


# --- Diagnostic metrics (weight=0 in rubric — show in wandb, don't enter loss) ---

@vf.reward
def is_correct(state, answer, **_) -> float:
    return 1.0 if _bucket(state, answer) == "correct" else 0.0


@vf.reward
def is_wrong(state, answer, **_) -> float:
    return 1.0 if _bucket(state, answer) == "wrong" else 0.0


@vf.reward
def is_quit(state, answer, **_) -> float:
    return 1.0 if _bucket(state, answer) == "quit" else 0.0


@vf.reward
def is_timeout(state, answer, **_) -> float:
    return 1.0 if _bucket(state, answer) == "timeout" else 0.0


@vf.reward
def is_parseable(state, answer, **_) -> float:
    return 1.0 if is_parseable_arithmetic(_committed_answer(state)) else 0.0


@vf.reward
def elapsed_over_target(state, answer, **_) -> float:
    """t/T ratio — the headline pacing metric. <1 = on time, >1 = over budget."""
    T = state.get("target_s")
    if not T:
        return 0.0
    return _elapsed(state) / T


@vf.reward
def f_term(state, answer, **_) -> float:
    """The pure time term f(t,T)=min(1,T/t) (hyperbolic) over ALL rollouts, logged
    natively at weight=0. The c·f reward folds correctness and timing together;
    this exposes the timing half on its own so the split is visible on wandb without
    backing it out by division (Kanishk 2026-05-24). For controls A/B (reward_time_term
    off) this is the key control metric: does the model still land in-budget even
    when timing isn't rewarded?"""
    cfg = state.get("_cfg") or {}
    T = state.get("target_s")
    t = _elapsed(state)
    if not T or t is None:
        return 1.0
    return _time_factor(t, T, cfg.get("reward_shape", "hyperbolic"),
                        cfg.get("reward_alpha", 1.0),
                        cfg.get("max_time_multiplier", 5.0),
                        cfg.get("enforce_max_time", True))


@vf.reward
def mean_n_turns(state, **_) -> float:
    """Number of turns the rollout took. Weight=0 diagnostic — shows policy behavior."""
    return float(len(state.get("trajectory", [])))


@vf.reward
def mean_completion_tokens(state, **_) -> float:
    """Sum of output tokens across all turns. Weight=0 diagnostic for cost / verbosity.
    Reads via _step_usage (response.usage) — step['tokens'] is None for the OpenAI
    chat-completions client, which previously made this read a constant 0."""
    return float(sum(_step_usage(step)[1] for step in state.get("trajectory", [])))


def _load_dataset(jsonl_path: str, cfg: InteroceptionConfig) -> Dataset:
    """Load problems and assign each one a deterministic target_s.

    target_s is sampled log-uniform with a seeded RNG so:
      - all G rollouts of the same problem in one step share T (GRPO group invariant)
      - the assignment is reproducible: same dataset_seed → same T per problem
      - different dataset_seeds produce different per-problem T-assignments
        (the primary source of cross-seed variance in the sweep)
    """
    path = Path(jsonl_path)
    log_lo = math.log(cfg.target_s_min)
    log_hi = math.log(cfg.target_s_max)
    rows = []
    with path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rng = random.Random(cfg.dataset_seed ^ (idx * 2654435761 & 0xFFFFFFFF))
            if cfg.target_s_dist == "log":
                target_s = math.exp(rng.uniform(log_lo, log_hi))
            else:  # uniform (v2 default)
                target_s = rng.uniform(cfg.target_s_min, cfg.target_s_max)
            rows.append({
                "prompt": [{"role": "user", "content": _build_prompt(r["nums"], r["target"])}],
                "answer": {"nums": list(r["nums"]), "target": int(r["target"])},
                "info": {
                    "solution_count": r.get("solution_count", 0),
                    "target_s": target_s,
                },
            })
    return Dataset.from_list(rows)


def load_environment(**kwargs) -> vf.Environment:
    """Entry point discovered by `prime env install` / `prime eval run`.

    All InteroceptionConfig fields can be passed as kwargs (e.g. from a TOML config).

    Rubric composition depends on cfg.reward_shape:
      - "hyperbolic" / "exponential":
            correctness_with_time (w=1) + parseable_bonus (w=cfg.attempt_bonus)
            + 4 bucket-frequency metrics (w=0) + elapsed_over_target metric (w=0)
      - "asymmetric" (old shape, fallback):
            correct_with_speed_bonus (w=cfg.alpha) + wrong/quit/timeout penalties
            + same diagnostic metrics
    """
    cfg_fields = set(InteroceptionConfig.model_fields.keys())
    cfg = InteroceptionConfig(**{k: v for k, v in kwargs.items() if k in cfg_fields})

    # Fail loud if hwprop catalog strings are misspelled (otherwise silent
    # fallback to wrong latency profile, polluting the elapsed signal).
    if cfg.timing_source == "sim":
        from hwprop.simulator import simulate_latency
        simulate_latency(cfg.hardware, cfg.sim_model, prompt_len=1, decode_steps=1)

    dataset = _load_dataset(cfg.problems_jsonl, cfg)

    # Diagnostic metrics added to every shape — weight=0 means they're tracked
    # in wandb (avg per batch) but don't enter the loss.
    diagnostic_metrics = [
        (is_correct, 0.0),
        (is_wrong, 0.0),
        (is_quit, 0.0),
        (is_timeout, 0.0),
        (is_parseable, 0.0),
        (elapsed_over_target, 0.0),
        (f_term, 0.0),
        (mean_n_turns, 0.0),
        (mean_completion_tokens, 0.0),
    ]

    if cfg.reward_shape == "asymmetric":
        reward_funcs = [
            (correct_with_speed_bonus, cfg.alpha),
            (wrong_penalty, cfg.beta_wrong),
            (quit_penalty, cfg.beta_quit),
            (timeout_penalty, cfg.gamma),
        ]
    else:  # hyperbolic or exponential
        reward_funcs = [
            (correctness_with_time, 1.0),
            (parseable_bonus, cfg.attempt_bonus),
        ]

    all_funcs = reward_funcs + diagnostic_metrics
    rubric = vf.Rubric(
        funcs=[f for f, _ in all_funcs],
        weights=[w for _, w in all_funcs],
    )
    return CountdownTimeBudgetEnv(cfg=cfg, dataset=dataset, rubric=rubric)
