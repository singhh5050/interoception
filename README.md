# interoception

Experiment: can an LLM be made wallclock-aware? Given a budget `T` and `[X seconds elapsed]` messages injected between its own turns, does the model pace its reasoning and commit before the deadline?

Task: 4-number Countdown problems (use +/-/×/÷ with each number exactly once to hit a target). Dataset built from [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).

## Layout

```
environments/interoception_countdown/   # verifiers MultiTurnEnv package
configs/rl/train_qwen3_4b.toml          # prime-rl training config
configs/eval/README.md                  # prime eval run howto
scripts/build_dataset.py                # rebuild data/{train,eval}.jsonl
data/{train,eval}.jsonl                 # stratified Countdown problems
runs/                                   # historical sweep + reproduction outputs
```

The env subclasses `vf.MultiTurnEnv`. It injects `[X seconds elapsed]` between model turns where `X` comes from either real per-turn wallclock (default, eval) or `hwprop.simulate_latency` (opt-in, RL). Reward decomposes into four buckets — correct (+ speed bonus), wrong, quit, timeout — with the asymmetric weighting from the original experiment.

## Install

```bash
# Env package only (real-timing mode):
pip install -e environments/interoception_countdown

# Add simulator (for RL training with deterministic per-rollout cost):
pip install -e environments/interoception_countdown[sim]
# hwprop is currently a sibling checkout; see https://github.com/singhh5050/hardware-proprioception
```

## Run

```bash
# Rebuild the dataset (one-time)
python scripts/build_dataset.py --out-dir data

# Eval a model on the env
prime eval run --env interoception-countdown --model Qwen/Qwen3-4B-Instruct-2507 \
  --env-args '{"problems_jsonl":"data/eval.jsonl","target_s_min":30,"target_s_max":30}' \
  --num-examples 50

# RL training (prime-rl)
uv run trainer @ configs/rl/train_qwen3_4b.toml
```

## Findings so far

Track A trajectory sweeps (n=20 per cell, 3 models × {sim, real} timing):

| Model | correct% | notes |
|---|---:|---|
| gemma-4-E4B-it | 25–35% | strongest by far; only fails by timing out |
| Qwen3-4B-Instruct-2507 | 20% | format adherence is the main blocker |
| Qwen2.5-3B-Instruct | 0% | confidently wrong + quitting |

Sim vs real timing made no meaningful difference within any model — the model is not sensitive enough to timing-source detail for it to swing the experimental result.
