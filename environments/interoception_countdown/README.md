# interoception-countdown

A verifiers `MultiTurnEnv` for testing whether LLMs use an elapsed-time signal injected between their own turns.

The model gets a Countdown problem (4 numbers + a target) and a wallclock budget `T` seconds. Between its turns, the env injects `[X seconds elapsed]` messages. `X` comes from one of two sources depending on `timing_source`:

- `"real"` (default) — actual per-turn wallclock from `state["timing"].model.spans`. Most honest for eval.
- `"sim"` — `hwprop.simulate_latency(...)`. Deterministic; supports counterfactual hardware ("train as if deployed on H100"); decouples model-perceived budget from actual training wall time. Useful for RL.

The rollout ends when the model emits `<answer>EXPR</answer>`, accumulated elapsed time hits `T`, or `max_turns` is reached.

Reward decomposes into four buckets (see the module docstring):

| Bucket | Condition | Weight (config field) |
|---|---|---|
| `correct` | parses, uses each number once, hits target | `alpha` (+ speed bonus inside the function) |
| `wrong` | parses, misses target | `beta_wrong` (penalty) |
| `quit` | emitted `<answer>...</answer>` but unparseable | `beta_quit` (penalty) |
| `timeout` | never emitted `<answer>` | `gamma` (penalty) |

Names become Rubric metric labels, so per-bucket frequencies are reported for free.

## Install (local checkout)

```bash
# Default: real-timing mode, no extra deps needed.
pip install -e ./environments/interoception_countdown

# For sim-timing mode (RL): also install hwprop.
pip install -e ./environments/interoception_countdown[sim]
# (or install hwprop separately from a sibling checkout)
```

## Use from verifiers

```python
import verifiers as vf
env = vf.load_environment("interoception_countdown",
    problems_jsonl="data/eval.jsonl",
    target_s_min=30.0, target_s_max=30.0,  # pin T=30 for an eval reproducing the baseline
)
```

## Eval via `prime eval run`

```bash
prime eval run --env interoception_countdown --model Qwen/Qwen3-4B-Instruct-2507 \
    --env-args '{"problems_jsonl":"data/eval.jsonl","target_s_min":30,"target_s_max":30}'
```

## RL via prime-rl

See `configs/rl/train_qwen3_4b.toml` at the repo root. The trainer is `vf.GRPOTrainer` with `lora_defaults`.

## Config reference

All `InteroceptionConfig` fields (Pydantic) are passable as kwargs to `load_environment`:

| Field | Default | Meaning |
|---|---|---|
| `target_s_min` | `15.0` | Lower bound for log-uniform budget sampling (per rollout). |
| `target_s_max` | `120.0` | Upper bound. Set min==max for a fixed-budget eval. |
| `max_turns` | `16` | Hard cap on model turns per rollout. |
| `alpha` | `1.0` | Speed bonus weight on correct rollouts. |
| `beta_wrong` | `0.1` | Penalty weight on parseable-but-wrong. |
| `beta_quit` | `0.5` | Penalty weight on unparseable / "Not possible" / LaTeX. |
| `gamma` | `1.5` | Penalty weight on never-emitted-`<answer>` (timeout). |
| `problems_jsonl` | `"data/train.jsonl"` | Path to JSONL produced by `scripts/build_dataset.py`. |
| `timing_source` | `"real"` | `"real"` = framework wallclock; `"sim"` = hwprop simulator (requires `[sim]` extra). |
| `hardware` | `"A100_80GB"` | hwprop catalog name. Used only when `timing_source="sim"`. |
| `sim_model` | `"Qwen3-4B"` | hwprop catalog name. Used only when `timing_source="sim"`. |

## Notes

- **Time source.** Default is real per-turn wallclock from `state["timing"].model.spans[-1].end - .start`. Switch to `timing_source="sim"` for RL — gives deterministic per-rollout cost (cleaner GRPO baselines), supports training under a counterfactual hardware profile, and decouples model-perceived budget from actual training wall time.
- **Why not change the task.** Verifiers doesn't ship a Countdown env. Per the project's design call, we inherit `MultiTurnEnv` rather than switching to a task that's already covered (e.g. Wordle).
- **Known parsing limitation.** Inherited from `_solver.validate_solution`: model outputs like `(21+19) + (48/48) = 41` fail `ast.parse` because of the trailing `= 41`, so they're bucketed as "quit" even though the arithmetic is correct. Matches the old `rollout.py` behavior. Fix would be a small `re.sub(r'\s*=\s*\d+\s*$', '', expr)` in `_solver.validate_solution` before parsing — deliberately deferred so this env reproduces the historical baseline behavior exactly.
