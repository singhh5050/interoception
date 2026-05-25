# interoception-countdown

A verifiers `MultiTurnEnv` for testing whether an LLM uses (and is rewarded for) an elapsed-time signal injected between its own turns.

The model gets a Countdown problem (4 numbers + a target) and a wallclock budget `T` seconds. Between its turns the env injects `[X seconds elapsed]` messages (unless `inject_elapsed=false`). `X` comes from one of two sources depending on `timing_source`:

- `"real"` — actual per-turn wallclock from `state["timing"].model.spans`. Most honest for eval.
- `"sim"` — `hwprop.simulate_latency(...)`. Deterministic; supports counterfactual hardware ("train as if deployed on H100"); decouples model-perceived budget from actual training wall time. Used for RL.

The rollout ends when the model emits `<answer>EXPR</answer>`, `max_turns` is reached, or — **only if `enforce_max_time=true`** — accumulated elapsed time hits `max_time_multiplier × T`.

## Reward

The primary reward (`reward_shape="hyperbolic"`, the default) is a **product**:

```
reward = c · f(t, T)      c = 1 if correct else 0      f(t,T) = min(1, T/t)
```

`f` is 1 inside the budget and decays past `T` (hyperbolic `T/t`, or `exp(-α(t−T)/T)` with `reward_shape="exponential"`). Correctness and timeliness are **not separable** — a correct-but-late answer is partially rewarded; a wrong answer scores 0 regardless of timing.

Control knobs for the ablations:

- `reward_time_term=false` → `f ≡ 1`, so reward collapses to pure correctness `c` (controls A and B).
- `inject_elapsed=false` (+ `max_turns=1`) → the model never sees elapsed time; only the budget stated in the system prompt (control B).

Diagnostics logged at **weight 0** (tracked in wandb, not in the loss): `is_correct`, `is_wrong`, `is_quit`, `is_timeout`, `is_parseable`, `elapsed_over_target`, `f_term` (the pure `min(1,T/t)` time term, logged natively so the c·f split is visible), `mean_n_turns`, `mean_completion_tokens`.

A legacy 4-bucket asymmetric reward (`correct`/`wrong`/`quit`/`timeout` with `alpha`/`beta_wrong`/`beta_quit`/`gamma`) is available behind `reward_shape="asymmetric"`.

## Scoring robustness

- **Answer is extracted at scoring time from the final assistant turn**, not only from the `env_response` hook (which doesn't fire after the last/only turn). So single-turn rollouts (`max_turns=1`) and multi-turn rollouts that commit on the final allowed turn are scored correctly.
- **Elapsed time is finalized at scoring time over the whole trajectory** (`_finalize_elapsed`), so the time term and `f_term`/`elapsed_over_target` include the final turn.
- **Operator/whitespace normalization**: `_solver` maps Unicode math operators (`× ÷ − ∗ ⋅ ∕`) and Unicode spaces to ASCII before `ast.parse`, and strips a trailing `= 79`. Models routinely emit `<answer>29 × (10 − 8) + 21</answer>`; without normalization a correct answer would be bucketed as `quit`.

## Install (local checkout)

```bash
pip install -e ./environments/interoception_countdown          # real-timing mode
pip install -e ./environments/interoception_countdown[sim]     # + hwprop for sim-timing (RL)
```

## Use from verifiers

```python
import verifiers as vf
env = vf.load_environment("interoception_countdown",
    problems_jsonl="data/eval.jsonl",
    target_s_min=30.0, target_s_max=30.0,  # pin T=30 for a fixed-budget eval cell
)
```

## RL via prime-rl

See `configs/rl/v2_qwen3_4b.toml` (treatment) and `configs/rl/ctrl{A,B}_qwen3_4b.toml` (controls) at the repo root.

## Config reference

All `InteroceptionConfig` fields (Pydantic) are passable as kwargs to `load_environment`:

| Field | Default | Meaning |
|---|---|---|
| `target_s_min` / `target_s_max` | `15.0` / `120.0` | Budget range. Set min==max for a fixed-budget eval cell. |
| `target_s_dist` | `"uniform"` | `"uniform"` (T~U) or `"log"` (log-uniform). Sampled once per dataset row (GRPO group invariant). |
| `dataset_seed` | `0` | Seeds per-row `target_s` assignment; different seeds → different per-problem T. |
| `max_turns` | `16` | Hard cap on model turns. Set `1` for single-turn (control B). |
| `reward_shape` | `"hyperbolic"` | `"hyperbolic"` (c·min(1,T/t)), `"exponential"`, or legacy `"asymmetric"`. |
| `reward_alpha` | `1.0` | Decay coefficient (only for `"exponential"`). |
| `reward_time_term` | `true` | `false` → `f≡1`, reward = pure correctness `c` (controls A/B). |
| `inject_elapsed` | `true` | `false` → no `[Xs elapsed]` message; system prompt still states the budget (control B). |
| `enforce_max_time` | `true` | `false` → no time cutoff; rollout ends only on `<answer>` / `max_turns` (v2/controls). |
| `max_time_multiplier` | `5.0` | Cutoff at `multiplier × T` (only when `enforce_max_time=true`). |
| `attempt_bonus` | `0.0` | **Legacy/off.** Bonus for any parseable arithmetic — the v1 confound that reward-hacked Qwen2.5-3B. Set >0 only for legacy sweeps. |
| `alpha` / `beta_wrong` / `beta_quit` / `gamma` | `1.0` / `0.1` / `0.5` / `1.5` | Legacy asymmetric-shape weights. |
| `problems_jsonl` | `"data/train.jsonl"` | JSONL from `scripts/build_dataset.py`. |
| `timing_source` | `"real"` | `"real"` = framework wallclock; `"sim"` = hwprop (requires `[sim]` extra). |
| `hardware` / `sim_model` | `"A100_80GB"` / `"Qwen3-4B"` | hwprop catalog names (only when `timing_source="sim"`). |

## Notes

- **Why not change the task.** Verifiers doesn't ship a Countdown env; per the project's design call we inherit `MultiTurnEnv` rather than switching to an already-covered task.
- **Sim prefill.** The sim charges prefill only on turn 1, assuming a deployment with prefix caching enabled.
