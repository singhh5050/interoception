"""Wraps hwprop.simulate_latency to give per-turn wallclock estimates.

For an exploratory rollout we only need the decode time of each new chunk
given the accumulated context length. Prefill of newly appended user
messages (the "[Xs elapsed]" tag) is small and assumed cached after the
first turn — we add it on turn 0 only.
"""
from __future__ import annotations

from dataclasses import dataclass

from hwprop.simulator import simulate_latency


@dataclass
class WallclockEstimator:
    hardware: str = "H100_SXM"
    sim_model: str = "Qwen2.5-7B"

    def turn_time_s(self, prev_ctx_tokens: int, chunk_tokens: int, *, include_prefill: bool) -> float:
        """Estimated wallclock seconds for one turn.

        prev_ctx_tokens: tokens already in context when this turn starts decoding
        chunk_tokens:    tokens generated this turn
        include_prefill: True only for the very first turn (initial prompt prefill)
        """
        if chunk_tokens <= 0:
            return 0.0
        r = simulate_latency(
            self.hardware,
            self.sim_model,
            prompt_len=max(prev_ctx_tokens, 1),
            decode_steps=chunk_tokens,
        )
        return (r.prefill_time_s if include_prefill else 0.0) + r.total_decode_time_s
