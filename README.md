# interoception

Experiment: can an LLM be made wallclock-aware? Given a budget `T` and `[X seconds elapsed]` messages injected between its own turns, does the model pace its reasoning and commit before the deadline?

Task: 4-number Countdown problems (use +/-/×/÷ with each number exactly once to hit a target). Dataset built from [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).

## Layout

```
environments/interoception_countdown/   # verifiers MultiTurnEnv package
configs/rl/phase1_{hyp,exp}_s{0,1}.toml # Phase 1 sweep: 2 reward shapes × 2 seeds
configs/rl/smoke.toml                   # 5-step pipeline check (Modal)
scripts/dev/render_phase1_tomls.py      # template generator for the 4 phase1 TOMLs
scripts/dev/bundle_phase1_audit.py      # regenerate phase1_audit.txt audit bundle
scripts/dev/patch_prime_rl_pyproject.py # drops missing workspace members at image build
scripts/build_dataset.py                # rebuild data/{train,eval}.jsonl
data/{train,eval}.jsonl                 # stratified Countdown problems
modal_app.py                            # Modal app: smoke (1 run) + phase1 (4-run fanout)
runs/                                   # historical sweep + reproduction outputs
```

The env subclasses `vf.MultiTurnEnv`. It injects `[X seconds elapsed]` between model turns where `X` comes from either real per-turn wallclock (default, eval) or `hwprop.simulate_latency` (opt-in, RL). Reward is `c · f(t, T) + 0.05 · is_parseable_answer`, where `f` is one of `hyperbolic` (`T/min(t, 5T)` past T) or `exponential` (`exp(-(t-T)/T)` past T). Each problem is assigned a single `target_s` at dataset load time (deterministic in `dataset_seed`), so all GRPO group rollouts of one problem share T — required for valid advantage estimation.

## Install

```bash
pip install -e environments/interoception_countdown[sim]
# sim extra pulls hwprop. For real-timing-only eval, drop the [sim] extra.
```

## Run

```bash
# Rebuild dataset (one-time)
python scripts/build_dataset.py --out-dir data

# RL training via Modal (prime-rl pinned at SHA b22e768)
modal run modal_app.py::smoke               # 5-step plumbing check, ~7 min
modal run --detach modal_app.py::phase1     # 4-run sweep, ~80-100 min each, parallel

# Local eval
prime eval run --env interoception-countdown --model Qwen/Qwen2.5-3B-Instruct \
  --env-args '{"problems_jsonl":"data/eval.jsonl","target_s_min":30,"target_s_max":30}' \
  --num-examples 100
```

Modal env vars expected: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (in `.env`), plus a `wandb` Modal secret. The image clones prime-rl + hwprop + installs the env package; resolved configs land at `/cache/runs/<run>/configs/` on the `interoception-cache` volume.

When iterating on Phase 1 hyperparameters, edit `scripts/dev/render_phase1_tomls.py` and re-run it — the four TOMLs are guaranteed to differ only in `(shape, seed)`.

## Findings so far

Track A trajectory sweeps (n=20 per cell, 3 models × {sim, real} timing):

| Model | correct% | notes |
|---|---:|---|
| gemma-4-E4B-it | 25–35% | strongest by far; only fails by timing out |
| Qwen3-4B-Instruct-2507 | 20% | format adherence is the main blocker |
| Qwen2.5-3B-Instruct | 0% | confidently wrong + quitting |

Sim vs real timing made no meaningful difference within any model — the model is not sensitive enough to timing-source detail for it to swing the experimental result.
