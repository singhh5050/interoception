# Standalone eval

Standalone evals (without launching a full RL run) use the `prime` CLI's `prime eval run`. No separate TOML is required — env args are passed inline.

## Reproduce a baseline cell

The original baselines used T=30s and T=60s on 50 problems from `data/eval.jsonl`. After installing the env (`prime env install ./environments/interoception_countdown`), this reproduces the Qwen3-4B T=60 cell:

```bash
prime eval run \
  --env interoception-countdown \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --env-args '{
    "problems_jsonl":"data/eval.jsonl",
    "target_s_min":60.0,
    "target_s_max":60.0,
    "hardware":"A100_80GB",
    "sim_model":"Qwen3-4B"
  }' \
  --num-examples 50
```

For RL-time periodic eval, see `configs/rl/train_qwen3_4b.toml` (`[orchestrator.eval]` section).
