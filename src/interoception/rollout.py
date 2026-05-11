"""Multi-turn rollout with simulated wallclock injection.

Each turn:
  1. Sample up to `chunk_tokens` tokens from the model.
  2. Ask the wallclock simulator how long that chunk would have taken on the
     target hardware, accumulate it into `elapsed_s`.
  3. If the response contains the closing `</answer>` tag, stop.
  4. If `elapsed_s >= target_s`, stop with a timeout.
  5. Otherwise append a user message of the form "[{elapsed}s elapsed]"
     and loop.

The model never sees the wallclock as a number it computed itself — it sees
it as text injected by the user between its turns. The point of this script
is to look at what an off-the-shelf model does with that signal, before any
RL training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams

from .sim_wallclock import WallclockEstimator


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class TurnRecord:
    role: str
    content: str
    output_tokens: int = 0
    elapsed_s_at_end: float = 0.0


@dataclass
class RolloutResult:
    target_s: float
    elapsed_s: float
    turns: list[TurnRecord] = field(default_factory=list)
    answer: str | None = None
    timed_out: bool = False
    answer_emitted: bool = False

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns if t.role == "assistant")


def build_system_prompt(target_s: float, style: str = "default") -> str:
    """Build the system prompt for a rollout.

    Three styles, varying in how much guidance the model gets about the
    elapsed-time signal:

    - "default": Mentions the budget and the [Xs elapsed] tag and says
      "pace yourself," but gives no specific decision rules. Closest to a
      naive "we told the model what's happening" baseline.

    - "medium":  Same setup as default, but explicitly frames the elapsed
      messages as a signal to "take seriously and adjust strategy
      accordingly." Tests whether *emphasizing* the signal as important
      is enough to elicit time-aware behavior, without specifying any
      policy.

    - "strong":  Full prescriptive policy — state remaining budget after
      each elapsed message, switch strategies if repeating, commit at 70%,
      commit immediately if correct. Tests the ceiling of what prompting
      alone can install.
    """
    if style == "default":
        return (
            "You are solving a problem under a wallclock time budget.\n"
            f"Your budget is {target_s:.0f} seconds.\n"
            "Between your turns the user will inject messages of the form "
            '"[X seconds elapsed]" telling you how much wallclock time has passed.\n'
            "You should pace yourself: think when there is time, commit to an answer "
            "when time runs short. When you are ready, output your final answer "
            "inside <answer>...</answer> tags. Anything after </answer> is ignored."
        )
    if style == "medium":
        return (
            f"You are solving a problem under a strict wallclock time budget of {target_s:.0f} seconds.\n\n"
            'The user will inject "[X seconds elapsed]" messages between your turns telling you '
            "how much wallclock time has passed. Take these signals seriously and adjust your "
            "strategy accordingly.\n\n"
            "Output your final answer inside <answer>...</answer> tags. Anything after </answer> is ignored."
        )
    if style == "strong":
        return (
            f"You are solving a problem under a strict wallclock time budget of {target_s:.0f} seconds.\n\n"
            'The user will inject "[X seconds elapsed]" messages between your turns telling you '
            "how much wallclock time has passed. You MUST use these signals to pace yourself:\n\n"
            f"1. After each elapsed message, briefly state how much of the {target_s:.0f}s budget is "
            "left and whether your current approach is making progress.\n"
            "2. If your reasoning is repeating itself — trying combinations you've already tried, "
            "or getting the same wrong answer — STOP that line and switch strategies.\n"
            "3. If more than 70% of your budget is gone and you do not have a confident answer, "
            "commit to your best current guess rather than continuing to search. A wrong answer "
            "is better than no answer (a timeout scores zero).\n"
            "4. If you find a correct answer, commit immediately — do not keep exploring.\n\n"
            "Output your final answer inside <answer>...</answer> tags. Anything after </answer> is ignored."
        )
    raise ValueError(f"unknown prompt style: {style!r}")


def run_rollout(
    llm: LLM,
    question: str,
    target_s: float,
    estimator: WallclockEstimator,
    *,
    chunk_tokens: int = 256,
    max_turns: int = 32,
    temperature: float = 0.7,
    seed: int | None = None,
    prompt_style: str = "default",
) -> RolloutResult:
    tokenizer = llm.get_tokenizer()

    system_prompt = build_system_prompt(target_s, style=prompt_style)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    result = RolloutResult(target_s=target_s, elapsed_s=0.0)
    result.turns.append(TurnRecord(role="system", content=system_prompt))
    result.turns.append(TurnRecord(role="user", content=question))

    elapsed_s = 0.0

    for turn_idx in range(max_turns):
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prev_ctx_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        sp = SamplingParams(
            max_tokens=chunk_tokens,
            temperature=temperature,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            seed=seed,
        )
        outs = llm.generate(
            prompts=[prompt_text],
            sampling_params=sp,
            use_tqdm=False,
        )
        out = outs[0].outputs[0]
        chunk_text = out.text
        chunk_n_tokens = len(out.token_ids)

        chunk_time_s = estimator.turn_time_s(
            prev_ctx_tokens=prev_ctx_tokens,
            chunk_tokens=chunk_n_tokens,
            include_prefill=(turn_idx == 0),
        )
        elapsed_s += chunk_time_s

        messages.append({"role": "assistant", "content": chunk_text})
        result.turns.append(
            TurnRecord(
                role="assistant",
                content=chunk_text,
                output_tokens=chunk_n_tokens,
                elapsed_s_at_end=elapsed_s,
            )
        )

        match = ANSWER_TAG_RE.search(chunk_text)
        if match:
            result.answer = match.group(1).strip()
            result.answer_emitted = True
            break
        # Also handle the case where stop=</answer> cut the string just before the closing tag —
        # the model emitted "<answer>...":
        if "<answer>" in chunk_text and "</answer>" not in chunk_text:
            # Stop was hit but the regex above only matches complete tags. Capture what's there.
            after = chunk_text.split("<answer>", 1)[1]
            result.answer = after.strip()
            result.answer_emitted = True
            break

        if elapsed_s >= target_s:
            result.timed_out = True
            break

        injected = f"[{elapsed_s:.1f}s elapsed]"
        messages.append({"role": "user", "content": injected})
        result.turns.append(TurnRecord(role="user", content=injected, elapsed_s_at_end=elapsed_s))

    result.elapsed_s = elapsed_s
    return result
