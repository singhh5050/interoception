# Handoff: launching the next interoception experiment

Guide for an AI agent picking up this project. Read this top to bottom before doing
anything. It tells you what the project is, the exact next experiment, how to launch and
monitor it on Modal, how to analyze results, and the operating rules you must follow.

---

## 1. What this project is

**Goal:** teach an instruct LLM to be *wallclock-aware* on Countdown puzzles — to balance
getting the answer right against a stated time budget. There is no real clock: elapsed
time is **simulated** by `hwprop` (an analytical roofline GPU simulator) from the tokens
the model generates.

**Stack (do not change):** Prime Intellect's `prime-rl` (GRPO trainer + vLLM + orchestrator)
running the `verifiers` env in `environments/interoception_countdown/`, all executed on
**Modal**. We do *not* migrate frameworks.

**Reward:** `reward = c · f(t, T)` where `c ∈ {0,1}` is correctness and
`f = min(1, T/t)` is timeliness — `t` = simulated elapsed seconds, `T` = the per-problem
time budget. So a correct answer scores its timeliness `f`; a wrong answer scores 0.
GRPO requires all `G` rollouts of one problem share the same `T` (it's sampled per
problem, not per rollout).

---

## 2. Current state (what just ran)

`ctrl0-qwen3-4b-u1-40` (config `configs/rl/ctrl0_u1_40_qwen3_4b.toml`) — the treatment
(c·f, multi-turn, `[Xs elapsed]` injected) at **T~U(1,40)**, 128 tok/chunk, 16 turns,
A100, 200 steps. Finished cleanly. Smoothed (10-step) eval/training trends:

| curve | start → end |
|---|---|
| correctness `c` | ~0.21 → ~0.42 |
| timeliness `f(t,T)` | ~0.35 → ~0.46 |
| reward `c·f` | ~0.13 → ~0.24 |

**Diagnosis:** a correct Countdown solve costs the model ~1200–1700 tokens ≈ **42–60 s** of
sim time (at ~0.035 s/token on A100), while T~U(1,40) budgets sit mostly *below* that — so
the model is usually over budget and `f` stays low. The model *does* learn to solve
(correctness climbs) and both `c` and `f` trend up. **Kanishk's call: this is fine — the
goal is to learn the time/correctness *balance*, not to always land under budget.** Do not
"fix" this by loosening the budget. See `analysis/figures/24_reward_vs_elapsed.png` (the
reward geometry) and `25_u1_40_smoothed_curves.png`.

---

## 3. The next experiment (approved by Kanishk, 2026-05-26 thread)

Two runs. **Everything stays identical to `ctrl0_u1_40` except the noted fields.** Do NOT
change the budget (T~U(1,40)), chunk size (128 tok), or turns (16) — Kanishk was explicit.

1. **Train for longer** — `configs/rl/ctrl0_u1_40_long_qwen3_4b.toml`
   - `max_steps` 200 → **500**; `[ckpt] interval` 50 → 100 (fewer weight saves); eval
     interval stays 50 (10 eval points).
   - Launch: `next_long_qwen3_4b`.

2. **YOLO run with Qwen2.5-3B-Instruct** — `configs/rl/ctrl0_u1_40_qwen25_3b.toml`
   - `model.name` → `Qwen/Qwen2.5-3B-Instruct`; `sim_model` → `Qwen2.5-3B` (so hwprop
     charges 3B decode time, ~3.4 s/128-tok chunk vs 4.5 s for the 4B).
   - Launch: `next_yolo_qwen25_3b`.

`max_steps = 500` is the main knob; bump it if Kanishk wants even longer. If you go *much*
longer, consider lengthening `[trainer.scheduler] decay_steps` (currently 10) for a gentler
LR tail.

---

## 4. How to launch (Modal)

Secrets live in `.env` at repo root (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
`WANDB_API_KEY`) — **gitignored, never commit them**. Load them, then launch detached:

```bash
set -a && source .env && set +a
modal run --detach modal_app.py::next_long_qwen3_4b      # the longer Qwen3-4B run
# or
modal run --detach modal_app.py::next_yolo_qwen25_3b     # the Qwen2.5-3B YOLO run
```

- `--detach` keeps the run alive on Modal independent of your terminal. The local process
  is just a log stream and will drop after ~12 min (Modal's stream cap) — **that is not the
  run ending.** The run survives server-side.
- `train_run` uses 2×A100-80GB, 6 h timeout, mounts the `interoception-cache` volume at
  `/cache`, writes weights to `output_dir`. A 500-step run takes a few hours.
- Expect ~10–15 min of image/dependency setup before step 0.

**hwprop is only ever run on Modal** (see `chunk_latency_calc` in `modal_app.py` for the
pattern: `modal run modal_app.py::chunk_latency`). Do not `pip install` or clone it
locally — that path is intentionally not used here.

---

## 5. How to monitor

The reliable completion signal is the **final checkpoint + its `STABLE` marker** on the
volume, not the log stream exiting.

```bash
set -a && source .env && set +a
modal app list --json | python3 -c "import sys,json;[print(a['Name'],a['State']) for a in json.load(sys.stdin)]"
modal volume ls interoception-cache runs/ctrl0_u1_40_long_qwen3_4b/weights          # step_100..500
modal volume ls interoception-cache runs/ctrl0_u1_40_long_qwen3_4b/weights/step_500 # expect a STABLE file
```

Live metrics: wandb project `interoception`, entity
`singhh5050-stanford-university/interoception`, run names `ctrl0-qwen3-4b-u1-40-long` /
`ctrl0-qwen25-3b-u1-40-yolo`.

> **wandb gotcha that will waste your time:** metric keys are namespaced by the **env
> `name`**, NOT the run's display name. For the long run that's
> `ctrl0-u1-40-long-qwen3-4b-train`, so reward is
> `reward/ctrl0-u1-40-long-qwen3-4b-train/mean` and diagnostics are
> `metrics/ctrl0-u1-40-long-qwen3-4b-train/{is_correct,f_term,elapsed_over_target,mean_completion_tokens,...}`.
> The custom `step` field is only co-logged with `reward/*`; for `metrics/*` use `_step`
> and treat sorted order as the training step.

---

## 6. How to analyze

- **Smoothed training curves** (correctness, f, reward): adapt
  `scripts/dev/plot_u1_40_curves.py` — change `RUN` (display name) and `ENV` (the
  namespace prefix above), rerun. Produces a 3-panel figure into `analysis/figures/`.
- **Eval acc-vs-T + pacing** (base vs final, binned + logistic slope):
  `scripts/dev/analyze_controls.py` is the template. It pulls the eval sample tables and
  uses a key shortcut enabled by `attempt_bonus = 0`:
  - `reward > 0  ⟺  is_correct` (correctness comes straight from reward sign), and
  - for the treatment, `reward == f` among correct rollouts (pacing comes from reward value).
  Per-rollout `T` is recovered from `dataset_seed=777` via the env's sampling formula
  (`random.Random(777 ^ (example_id*2654435761 & 0xFFFFFFFF)).uniform(T_lo, T_hi)`) — note
  `T_lo, T_hi = 1, 40` here, not the old 15/130.
- The eval ran on a **held-out** set (`data/eval.jsonl`, 498 problems), uniform T, temp 1.0,
  `dataset_seed=777` so (problem, T) pairs are fixed and paired across runs/steps.

---

## 7. Key technical facts (so you don't re-derive them)

- **Time is tokens in disguise:** `elapsed_s ≈ prefill (turn 1 only, prefix-caching
  assumed) + total_decode_tokens × per_token_time`. Per-token decode ≈ 0.035 s on A100 for
  the 4B (128-tok chunk ≈ 4.5 s; ~3.4 s for the 3B). **Chunk size does NOT change total
  time** for a given amount of generation — only feedback granularity. **Bigger seq_len /
  more turns only raise the token ceiling, not the model's speed.**
- **seq_len 2048** minus the ~300-tok system+problem prompt ≈ 1750 generation tokens ≈
  ~13–14 chunks ≈ **~60 s elapsed ceiling** for the 4B. Budgets above that are physically
  unreachable (this is why the old T~U(15,130) had no time pressure).
- **Decoding stops on max tokens, never on max time** (`enforce_max_time = false`). Keep it
  that way — Kanishk asked specifically.
- Env config flags worth knowing: `reward_time_term` (true = use f; false = reward is just
  c), `inject_elapsed` (whether `[Xs elapsed]` messages are added), `timing_source="sim"`,
  `hardware="A100_80GB"`, `sim_model`. The solver normalizes Unicode operators (× ÷ −) and
  strips trailing `= N` before parsing, and matches TinyZero's 1e-5 target tolerance.

---

## 8. Operating rules (non-negotiable)

1. **Never launch a training run or push to git without the human's explicit OK.** Before
   launching, post the config to Kanishk in the thread ("send configs before launching").
2. **Push to BOTH remotes** when authorized: `origin` (github.com/singhh5050/interoception)
   AND `upstream` (github.com/nicolesplaining/interoception). Commit directly to `main`.
3. **Never commit `.env`** or paste Modal tokens. (The wandb key being visible is tolerated
   by the human, but don't go out of your way.)
4. **Stay on prime-rl + verifiers.** No framework pivots.
5. **Run hwprop via Modal only**, never locally.

---

## 9. Repo map

```
configs/rl/                         prime-rl TOML configs (one per run)
  ctrl0_u1_40_qwen3_4b.toml         the run that just finished (treatment, T~U(1,40))
  ctrl0_u1_40_long_qwen3_4b.toml    NEXT: train longer (500 steps)
  ctrl0_u1_40_qwen25_3b.toml        NEXT: Qwen2.5-3B YOLO
  ctrl0/A/B_qwen3_4b.toml           the earlier 3-way f(t,T) ablation
environments/interoception_countdown/
  interoception_countdown.py        the verifiers env (reward, multi-turn, sim timing)
  _solver.py                        Countdown scorer (Unicode-robust, TinyZero-compatible)
  README.md                         env-level docs
modal_app.py                        Modal entrypoints (train_run + per-experiment launchers)
scripts/dev/                        analysis + plotting (analyze_controls.py, plot_*.py)
analysis/figures/                   committed result figures
data/{train,eval}.jsonl             Countdown problems (9999 train / 498 held-out eval)
```
