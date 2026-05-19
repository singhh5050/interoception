"""End-to-end test: run a few rollouts through the new env package using a
local vLLM OpenAI server. Verifies env_response, stop hooks, and rewards work
against a real LLM."""
import asyncio
import json
import sys

from interoception_countdown import load_environment
from openai import AsyncOpenAI
import verifiers as vf


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = "Qwen/Qwen3-4B-Instruct-2507"

    env = load_environment(
        problems_jsonl="data/eval.jsonl",
        target_s_min=30.0,
        target_s_max=30.0,
        hardware="A100_80GB",
        sim_model="Qwen3-4B",
        max_turns=8,
    )
    print(f"env loaded — dataset size {len(env.dataset)}, picking first {n} rows")

    raw = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    client = vf.OpenAIChatCompletionsClient(raw)

    sampling = {"max_completion_tokens": 256, "temperature": 0.7}

    for i in range(n):
        row = env.dataset[i]
        print(f"\n--- rollout {i}: {row['answer']} ---")
        state = await env.rollout(input=row, client=client, model=model, sampling_args=sampling)
        await env.rubric.score_rollout(state)
        print(f"  turns          : {len(state['trajectory'])}")
        print(f"  target_s       : {state.get('target_s'):.1f}")
        print(f"  elapsed_s      : {state.get('elapsed_s'):.1f}")
        print(f"  answer_emitted : {state.get('answer_emitted')}")
        print(f"  parsed_answer  : {state.get('parsed_answer')!r}")
        print(f"  stop_condition : {state.get('stop_condition')}")
        rewards = state.get("rewards") or {}
        metrics = state.get("metrics") or {}
        print(f"  reward (total) : {state.get('reward')}")
        print(f"  rewards (per)  : {rewards}")
        print(f"  metrics        : {metrics}")
        print(f"  model spans n  : {len(state['timing'].model.spans)}")
        if state["timing"].model.spans:
            real_total = sum(s.end - s.start for s in state["timing"].model.spans)
            print(f"  real model time: {real_total:.2f}s (sim-driven elapsed_s was {state['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
