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

from _solver import validate_solution

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


class InteroceptionConfig(BaseModel):
    # Budget range (sampled log-uniform per rollout unless target_s is preset in state).
    target_s_min: float = 15.0
    target_s_max: float = 120.0
    # Rollout shape.
    max_turns: int = 16
    # Reward weights — match rl_train.py defaults.
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
    hardware: str = "A100_80GB"
    sim_model: str = "Qwen3-4B"


def _build_prompt(nums: list[int], target: int) -> str:
    return (
        f"Using the numbers {nums} and the operators +, -, *, / "
        f"(each number used exactly once), find an expression that equals {target}. "
        "Show your reasoning, then put the final expression inside <answer>...</answer>."
    )


def _build_system_prompt(target_s: float) -> str:
    return (
        "You are solving a problem under a wallclock time budget.\n"
        f"Your budget is {target_s:.0f} seconds.\n"
        'Between your turns the user will inject messages of the form "[X seconds elapsed]" '
        "telling you how much wallclock time has passed.\n"
        "You should pace yourself: think when there is time, commit to an answer when "
        "time runs short. When you are ready, output your final answer inside "
        "<answer>...</answer> tags. Anything after </answer> is ignored."
    )


class CountdownTimeBudgetEnv(vf.MultiTurnEnv):
    def __init__(self, cfg: InteroceptionConfig, **kwargs):
        super().__init__(max_turns=cfg.max_turns, **kwargs)
        self.cfg = cfg

    async def setup_state(self, state: vf.State) -> vf.State | None:
        # Pre-set target_s in state lets `prime eval run` pin a fixed budget;
        # otherwise sample log-uniform within [target_s_min, target_s_max].
        if "target_s" not in state:
            log_T = random.uniform(
                math.log(self.cfg.target_s_min), math.log(self.cfg.target_s_max)
            )
            state["target_s"] = math.exp(log_T)
        state["elapsed_s"] = 0.0
        state["answer_emitted"] = False
        state["parsed_answer"] = None

        # Inject the budget-aware system prompt into the prompt list if not already there.
        prompt = state.get("prompt") or []
        if not prompt or prompt[0].get("role") != "system":
            state["prompt"] = [
                {"role": "system", "content": _build_system_prompt(state["target_s"])},
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
            spans = state["timing"].model.spans
            if spans:
                state["elapsed_s"] += spans[-1].end - spans[-1].start
        else:
            # Sim: deterministic per-turn cost via hwprop. Useful for RL where we want
            # low-variance group baselines + counterfactual hardware. The chunk token
            # count comes from response.usage (step["tokens"] is None for the OpenAI
            # chat-completions client). prev_ctx_tokens is summed from past prompt sizes.
            from hwprop.simulator import simulate_latency  # lazy: only import if used
            usage = getattr(last.get("response"), "usage", None)
            chunk_tokens = int(getattr(usage, "completion_tokens", 0)) if usage else 0
            prev_ctx_tokens = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
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
            # final_env_response short-circuits the rollout loop (see has_final_env_response stop).
            state["final_env_response"] = [
                {"role": "user", "content": f"[committed at {state['elapsed_s']:.1f}s]"}
            ]
            return []
        # Handle the "model emitted <answer>...</answer>" was cut off mid-tag case:
        if "<answer>" in chunk_text and "</answer>" not in chunk_text:
            tail = chunk_text.split("<answer>", 1)[1].strip()
            state["answer_emitted"] = True
            state["parsed_answer"] = tail
            state["final_env_response"] = [
                {"role": "user", "content": f"[committed at {state['elapsed_s']:.1f}s]"}
            ]
            return []

        # Budget check: short-circuit BEFORE the framework generates another model turn.
        # The framework's stop checks run at the top of each iteration, but env_response
        # runs mid-iteration as part of get_prompt_messages. Without this short-circuit
        # the model would get one extra turn after the budget exhausts.
        if state["elapsed_s"] >= state["target_s"]:
            state["final_env_response"] = [
                {"role": "user", "content": f"[budget exhausted at {state['elapsed_s']:.1f}s]"}
            ]
            return []

        return [{"role": "user", "content": f"[{state['elapsed_s']:.1f}s elapsed]"}]

    @vf.stop
    async def budget_exhausted(self, state: vf.State) -> bool:
        # Safety net — env_response already short-circuits via final_env_response,
        # but this stop catches edge cases (e.g. env_response error path).
        return state.get("elapsed_s", 0.0) >= state.get("target_s", float("inf"))


# --- Reward functions: one per failure-mode bucket.
# Names become metric labels so Rubric reports per-bucket frequencies for free.
# Each returns 0 unless its bucket applies; the Rubric sums weighted contributions.

def _bucket(state: vf.State, answer) -> str:
    """Classify the rollout outcome. answer is the dataset row's answer dict."""
    if not state.get("answer_emitted"):
        return "timeout"
    parsed = state.get("parsed_answer")
    ok = validate_solution(parsed, answer["nums"], answer["target"])
    if ok is True:
        return "correct"
    if ok is False:
        return "wrong"
    return "quit"  # parsed is None / unparseable


@vf.reward
def correct_with_speed_bonus(state, answer, **_) -> float:
    if _bucket(state, answer) != "correct":
        return 0.0
    speed = max(0.0, (state["target_s"] - state["elapsed_s"]) / state["target_s"])
    # alpha is folded in by the Rubric weight below; the base reward is 1 + speed.
    return 1.0 + speed


@vf.reward
def wrong_penalty(state, answer, **_) -> float:
    return -1.0 if _bucket(state, answer) == "wrong" else 0.0


@vf.reward
def quit_penalty(state, answer, **_) -> float:
    return -1.0 if _bucket(state, answer) == "quit" else 0.0


@vf.reward
def timeout_penalty(state, answer, **_) -> float:
    return -1.0 if _bucket(state, answer) == "timeout" else 0.0


def _load_dataset(jsonl_path: str) -> Dataset:
    path = Path(jsonl_path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "prompt": [{"role": "user", "content": _build_prompt(r["nums"], r["target"])}],
                "answer": {"nums": list(r["nums"]), "target": int(r["target"])},
                "info": {"solution_count": r.get("solution_count", 0)},
            })
    return Dataset.from_list(rows)


def load_environment(**kwargs) -> vf.Environment:
    """Entry point discovered by `prime env install` / `prime eval run`.

    All InteroceptionConfig fields can be passed as kwargs (e.g. from a TOML
    config). Reward weights are mapped from config onto the Rubric:
        correct -> weight=1.0      (alpha applies inside the function)
        wrong   -> weight=beta_wrong
        quit    -> weight=beta_quit
        timeout -> weight=gamma
    """
    # Filter kwargs to known config fields so caller-supplied trainer plumbing keys are ignored.
    cfg_fields = set(InteroceptionConfig.model_fields.keys())
    cfg = InteroceptionConfig(**{k: v for k, v in kwargs.items() if k in cfg_fields})

    dataset = _load_dataset(cfg.problems_jsonl)
    rubric = vf.Rubric(
        funcs=[correct_with_speed_bonus, wrong_penalty, quit_penalty, timeout_penalty],
        weights=[cfg.alpha, cfg.beta_wrong, cfg.beta_quit, cfg.gamma],
    )
    return CountdownTimeBudgetEnv(cfg=cfg, dataset=dataset, rubric=rubric)
