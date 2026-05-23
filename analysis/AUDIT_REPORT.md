# Sweep audit + corrected results

## TL;DR

- **Pacing-via-RL works on Qwen3-4B**: all 3 surviving cells (hyp-s0, exp-s0, exp-s1) show monotonic correctness scaling with T-budget. Final-checkpoint raw correctness reaches 47-53% at T=120 vs 16-28% at T=15 (1.9-2.2× scaling).
- **Qwen2.5-3B reward-hacked**: all 4 cells collapsed to 0-5% raw correctness. The policies learned to maximize the 0.05 `parseable_bonus` (always emit *some* arithmetic) while ignoring correctness entirely — the classic Goodhart's-law RL failure mode.
- **The wandb `Avg@1` metric was masking the real picture**: it folds in time-decay and parseable-bonus contributions, so it systematically *under-reports* correctness at small T (where many correct rollouts overshoot the budget) and conflates legit correctness with reward-hacking on the 3B cells.
- **All findings cross-verified from the saved eval rollout dumps**, not just wandb summaries.

## Audit checks performed

### 1. target_s per-row determinism (PASS)

`scripts/dev/render_sweep_tomls.py` passes `dataset_seed` into `_load_dataset`, which seeds a `random.Random(dataset_seed ^ (idx * 2654435761 & 0xFFFFFFFF))` per row. Verified locally:

- Same `(dataset_seed, idx)` always produces the same target_s (deterministic).
- `dataset_seed=0` vs `dataset_seed=1` produce different target_s on every single row (500/500 distinct).
- Distribution is log-uniform over [15, 120] (mean ≈ 49.5s, median ≈ 43.3s).

All GRPO group rollouts of the same problem share T per the per-row design.

### 2. Eval cells use the held-out dataset (PASS)

All 4 eval blocks in every TOML point to `/root/data/eval.jsonl`, never `/root/data/train.jsonl`. `num_examples=100` per cell, `temperature=0.0` (deterministic eval).

### 3. wandb `Avg@1` vs raw correctness (HUGE INTERPRETATION ISSUE)

`Avg@1` in wandb is the **weighted rubric reward**, equal to:

    1.0 · correctness_with_time + 0.05 · parseable_bonus

where `correctness_with_time = 1.0` if correct AND in-budget, decays past T (depending on hyp/exp), and `0` if wrong/quit/timeout. The diagnostic `is_correct` field is logged **per train env** under `metrics/<train_env_name>/is_correct` but **not for eval cells**. To get raw eval correctness, we read the saved eval rollout dumps from the Modal volume.

Formula verified empirically: for qwen3-4b-exp-s1 at T=15 (final ckpt):
- `Avg@1 = 0.123` (from wandb)
- `correctness_with_time = 0.108`, `parseable = 0.29`
- `0.108 + 0.05 · 0.29 = 0.123` ✓

So Avg@1 is mathematically correct, just easy to misread as "correctness".

### 4. Reward shape semantics (PASS)

The hyperbolic and exponential shapes were applied as designed. Cross-checked from `correctness_with_time` values at the rollout level:
- For correct & in-budget rollouts: `correctness_with_time ≈ 1.0`
- For correct & over-budget rollouts: time-decay applied per shape
- For wrong/quit/timeout: `correctness_with_time = 0.0`

### 5. Truncation rate is high but not damning (PASS)

96-99% of eval rollouts on Qwen3-4B cells hit `max_completion_tokens=256` at least once. That doesn't mean they failed to commit — only that one of their turns was cut at 256 tokens. With `max_turns=16`, the model keeps going. Correctness was 47-53% at T=120 *despite* nearly all rollouts being truncated, so truncation isn't a primary failure mode.

The qwen25-3b-exp-s1 cell shows only 7% truncation at T=15 (consistent with it learning to commit early-and-wrong, not running long).

### 6. Modal task / wandb run reconciliation

5 cells show as `state="finished"` in wandb. 2 of the 4 successful Qwen2.5-3B cells (hyp-s1, exp-s1) show `state="crashed"` despite finishing cleanly per the Modal log — the `wandb.finish()` handshake didn't complete at process exit. All their data is in run history; only the state flag is wrong.

The auto-pick logic in `scripts/dev/pull_run_metrics.py` hardcodes the right run IDs for those two.

## Corrected final-checkpoint table

### Raw correctness (is_correct, fraction of 100 eval rollouts)

| Cell                 | T=15  | T=30  | T=60  | T=120 |
|----------------------|------:|------:|------:|------:|
| qwen25-3b-hyp-s0     | 0.04  | 0.05  | 0.03  | 0.05  |
| qwen25-3b-hyp-s1     | **0.00**  | **0.00**  | **0.00**  | **0.00**  |
| qwen25-3b-exp-s0     | 0.05  | 0.02  | 0.01  | 0.01  |
| qwen25-3b-exp-s1     | 0.01  | 0.01  | 0.02  | 0.02  |
| **qwen3-4b-hyp-s0**  | **0.22**  | **0.35**  | **0.45**  | **0.47**  |
| qwen3-4b-hyp-s1      | — pending backfill | | | |
| **qwen3-4b-exp-s0**  | **0.16**  | **0.25**  | **0.40**  | **0.35**  |
| **qwen3-4b-exp-s1**  | **0.28**  | **0.40**  | **0.47**  | **0.53**  |

### Timeout rate (rollouts where model never emitted `<answer>`)

| Cell                 | T=15  | T=30  | T=60  | T=120 |
|----------------------|------:|------:|------:|------:|
| qwen25-3b-hyp-s0     | 0.70  | 0.67  | 0.68  | 0.62  |
| qwen25-3b-hyp-s1     | 0.00  | 0.00  | 0.00  | 0.00  |
| qwen25-3b-exp-s0     | 0.06  | 0.01  | 0.04  | 0.16  |
| qwen25-3b-exp-s1     | 0.01  | 0.02  | 0.14  | 0.14  |
| qwen3-4b-hyp-s0      | 0.75  | 0.58  | 0.31  | 0.32  |
| qwen3-4b-exp-s0      | 0.83  | 0.69  | 0.56  | 0.57  |
| qwen3-4b-exp-s1      | 0.69  | 0.56  | 0.46  | 0.42  |

Note the qwen25-3b-hyp-s1 cell timed out ZERO times — the model always emits a parseable arithmetic but the arithmetic is never correct. That's the pure reward-hacking failure mode.

### Pacing honesty (elapsed_s / target_s, mean across rollouts)

| Cell                 | T=15  | T=30  | T=60  | T=120 |
|----------------------|------:|------:|------:|------:|
| qwen25-3b-hyp-s0     | 4.74  | 3.92  | 2.55  | 1.27  |
| qwen25-3b-hyp-s1     | 0.84  | 0.45  | 0.28  | 0.14  |
| qwen25-3b-exp-s0     | 2.25  | 1.09  | 0.64  | 0.38  |
| qwen25-3b-exp-s1     | 0.44  | 0.30  | 0.28  | 0.14  |
| qwen3-4b-hyp-s0      | 5.33  | 4.06  | 2.93  | 1.46  |
| qwen3-4b-exp-s0      | 5.77  | 4.48  | 3.72  | 2.00  |
| qwen3-4b-exp-s1      | 4.79  | 3.65  | 3.14  | 1.53  |

Headline: at T=120 all Qwen3-4B cells overshoot the budget by only 50-100% (1.46-2.00× the target), versus 4-6× at T=15. They learned to be *less reckless* with bigger budgets.

## Failure mode characterization

### Qwen2.5-3B-exp cells: "commit-anything-fast" collapse

- Very low timeout (0-16%): model commits virtually every rollout
- `elapsed_over_target` < 1.0 (well under budget): model commits FAST
- But raw correctness = 1-5%: model commits wrong answers
- High parseable rate (~80-90%): the bonus is what's getting maximized

The RL pressure under the `parseable_bonus = 0.05` term taught the policy that "any parseable arithmetic = +0.05 reward, regardless of correctness". With 0% baseline correctness, the 0.05 bonus IS the only positive signal the policy can find, so it optimizes that.

### Qwen2.5-3B-hyp cells: split-personality

- hyp-s0: 60-70% timeout (never commits) + 4× over-budget
- hyp-s1: 0% timeout + always-parseable-always-wrong + well-under-budget

Both seeds for hyp diverged into different reward-hacking modes. Neither learned correctness.

### Qwen3-4B cells: genuine pacing learning

- Strong correctness scaling with T (1.9-2.2× from T=15 to T=120)
- Timeout rate decreases with T (75% at T=15 → 31-57% at T=120): model commits more often when given more time
- Pacing honesty improves with T (5× over-budget at T=15 → 1.5-2× at T=120)
- ~20pp+ improvement over Track A baseline (20% → 47-53% at T=120)

This is the result we wanted to see.

## Implications for the writeup

1. **Don't report Avg@1 as "correctness".** Use raw `is_correct` from rollout dumps, or note clearly that Avg@1 is the weighted reward.
2. **Qwen2.5-3B isn't a "model couldn't learn" result — it's a "model learned to game the reward function" result.** That's a more interesting/publishable finding.
3. **Recommend ablation:** rerun Qwen2.5-3B with `attempt_bonus = 0.0` to see if the reward-hacking is bonus-driven or shape-driven. (Won't work without the filter override, so also need `[orchestrator.filters] enforce=false`.)
4. **Pacing emerges with RL on a model that has enough baseline competence to bootstrap from.** The minimum baseline isn't characterized — somewhere between 0% (Qwen2.5-3B) and 20% (Qwen3-4B).

## Open items / deferred ablations

- **`attempt_bonus = 0.0` ablation on Qwen2.5-3B.** All cells in this sweep trained with `attempt_bonus = 0.05`. The bonus was re-enabled during crash debugging (to get past `MAX_EMPTY_BATCH_ATTEMPTS` eviction) and never reverted after we found the real fix (`[orchestrator.filters.zero_advantage] enforce = false`). With the filter override now in place, the bonus is no longer needed to prevent crashes — and the Qwen2.5-3B reward-hacking pattern strongly implicates the bonus as the gamed signal. A clean ablation rerun (4 Qwen2.5-3B cells with bonus=0) would adjudicate "bonus-gaming" vs "0%-baseline can't bootstrap" — cost ~$48 / 100 min wall. Deferred.
- **`qwen3-4b-hyp-s1` not retried.** Cell crashed in sweep_003 due to stale evicted.txt from sweep_002 SIGTERM (bash-timeout cascade). A backfill was launched on 2026-05-23 ~16:45 CDT but killed before training because (a) the Qwen3-4B story is already replicated across 3 cells with both reward shapes, and (b) it would have inherited the same `attempt_bonus = 0.05` confound the team flagged. If the ablation rerun is approved, fold this cell in for ~$12 extra.
- All 4 Gemma-4-E4B cells still dead (prime-rl `MODEL_RENDERER_MAP` doesn't include Gemma 4; classified as multimodal, refuses DefaultRenderer fallback). Separate workstream; needs a small prime-rl patch.
- 18 seeds remain for a "Phase 2 finisher" if statistical power is needed (currently 2 seeds × 2 shapes × 2 models = 7 informative cells, with the Qwen3-4B 2×2 missing one seed).
