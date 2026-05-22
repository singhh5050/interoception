"""Bundle every file relevant to a sweep launch into a single audit txt.

Designed to be handed to a second agent for review before burning ~5 hours of
parallel GPU on 12 Modal runs. The bundle is regenerated whenever any of the
constituent files change; do not edit sweep_audit.txt by hand.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sweep_audit.txt"

# Order matters: highest-stakes files first so an auditor opening the bundle
# sees the showstopper-class code (env + TOMLs) before the surrounding infra.
FILES = [
    "environments/interoception_countdown/interoception_countdown.py",
    "environments/interoception_countdown/_solver.py",
    "environments/interoception_countdown/__init__.py",
    "environments/interoception_countdown/pyproject.toml",
    "environments/interoception_countdown/README.md",
    # Phase 1: Qwen2.5-3B
    "configs/rl/phase1_qwen25_3b_hyp_s0.toml",
    "configs/rl/phase1_qwen25_3b_hyp_s1.toml",
    "configs/rl/phase1_qwen25_3b_exp_s0.toml",
    "configs/rl/phase1_qwen25_3b_exp_s1.toml",
    # Phase 2a: Qwen3-4B
    "configs/rl/phase2_qwen3_4b_hyp_s0.toml",
    "configs/rl/phase2_qwen3_4b_hyp_s1.toml",
    "configs/rl/phase2_qwen3_4b_exp_s0.toml",
    "configs/rl/phase2_qwen3_4b_exp_s1.toml",
    # Phase 2b: gemma-4-E4B
    "configs/rl/phase2_gemma4_e4b_hyp_s0.toml",
    "configs/rl/phase2_gemma4_e4b_hyp_s1.toml",
    "configs/rl/phase2_gemma4_e4b_exp_s0.toml",
    "configs/rl/phase2_gemma4_e4b_exp_s1.toml",
    "configs/rl/smoke.toml",
    "scripts/dev/render_sweep_tomls.py",
    "scripts/dev/patch_prime_rl_pyproject.py",
    "modal_app.py",
]

PREAMBLE = """\
================================================================================
SWEEP AUDIT BUNDLE
================================================================================

Generated: {ts}

You are reviewing the code and configs for a 12-run RL sweep. The goal of this
audit is to catch bugs / silent failures BEFORE we launch and burn ~5 hours of
parallel A100-80GB time on broken runs.

--------------------------------------------------------------------------------
EXPERIMENTAL DESIGN
--------------------------------------------------------------------------------

Train 3 instruct models with LoRA (r=32 alpha=32, all attn+MLP targets) on
Countdown puzzles under simulated wallclock budgets. Each problem comes with a
target time T; between turns the env injects "[X seconds elapsed]" using
hwprop-simulated latency. Model must learn to pace itself and emit
<answer>EXPR</answer> in time.

Reward shape (the per-rollout score that GRPO normalizes):
    R(t, T) = c(answer) * f(t, T) + 0.05 * is_parseable_arithmetic(answer)

where c(answer) = 1 if correct else 0, and f is one of:
    hyperbolic:   f = 1 if t <= T else T/min(t, 5T)
    exponential:  f = 1 if t <= T else exp(-(t-T)/T), clamped at t=5T

Hard cutoff: rollout stops at t = 5T regardless. attempt_bonus=0.05 pulls the
model from quit -> attempt (committing any parseable arithmetic, even wrong).

--------------------------------------------------------------------------------
SWEEP MATRIX (12 runs = 3 models x 2 shapes x 2 seeds)
--------------------------------------------------------------------------------

| model                          | baseline | sim_model    | role                  |
| ------------------------------ | -------- | ------------ | --------------------- |
| Qwen/Qwen2.5-3B-Instruct       |    ~0%   | Qwen2.5-3B   | weakest base (Phase 1)|
| Qwen/Qwen3-4B-Instruct-2507    |   ~20%   | Qwen3-4B     | mid base    (Phase 2) |
| google/gemma-4-E4B-it          | 25-35%   | Gemma-3-4B   | strongest   (Phase 2) |

Each model x {{hyperbolic, exponential}} x {{seed 0, seed 1}} = 4 cells per model = 12 total.

The original design (in conversation history pre-compaction) was 3 models x 2 shapes
x 5 seeds = 30 runs, phased as Phase 1 (4-run validation) -> Phase 2 (26 more). This
sweep launches Phase 1 + the cross-model cells together at 2 seeds per (model, shape)
to hedge the Qwen2.5-3B-null-result risk (0% baseline -> sparse GRPO advantage).
The remaining 18 seeds are a "Phase 2 finisher" added if/when results are clean.

Phase / model naming in the configs:
  phase1_qwen25_3b_*.toml    -> 4 runs (Phase 1, original validation cohort)
  phase2_qwen3_4b_*.toml     -> 4 runs (Phase 2 - mid baseline)
  phase2_gemma4_e4b_*.toml   -> 4 runs (Phase 2 - strongest baseline)

Launch: modal run --detach modal_app.py::sweep

--------------------------------------------------------------------------------
INFRASTRUCTURE
--------------------------------------------------------------------------------

- Modal serverless: 2x A100-80GB per run, 4 runs via train_run.starmap parallelism.
- prime-rl pinned at SHA b22e768fc419a1e8664729fd3fdfde98d1c13766 (production RL framework).
- verifiers env package installed via `uv pip install -e /root/env_pkg` in the Modal image.
- hwprop cloned + installed for sim latency (timing_source="sim" in training).
- prime-rl writes resolved configs to /cache/runs/<run>/configs/{{trainer,orchestrator,inference}}.toml.

--------------------------------------------------------------------------------
PINNED DECISIONS
--------------------------------------------------------------------------------

- Base model: Qwen/Qwen2.5-3B-Instruct (HF weight name; hwprop catalog key is "Qwen2.5-3B").
- LoRA: r=32, alpha=32, dropout=0, all attn+MLP targets.
- Optimizer: AdamW, lr=5e-5 (matches the prime-rl LoRA CI test; default 1e-6 is too low).
- Scheduler: linear with warmup_steps=10.
- KL coef: kl_tau=1e-3 (the default, pinned explicitly).
- Batch: orchestrator.batch_size=64, GRPO group rollouts_per_example=4 -> 16 problems/step.
- Steps: 200. Eval interval: every 20 steps.
- Eval cells: T=15, T=30, T=60, T=120, num_examples=100 each, temperature=0.0 (deterministic).
- Train target_s: log-uniform in [15, 120].

--------------------------------------------------------------------------------
SMOKE RUN VALIDATION (rev 3 — passed)
--------------------------------------------------------------------------------

Modal smoke (5 training steps + 2 evals) completed end-to-end. Resolved configs
on volume at /cache/runs/smoke-001/configs/ show LoRA applied correctly:

- trainer.model.lora: rank=32, alpha=32.0, dropout=0.0, all 7 target_modules.
- trainer.optim: lr=5e-05, max_norm=1.0, weight_decay=0.01, betas (0.9, 0.999).
- trainer.loss: type="default", kl_tau=1e-3, dppo_mask defaults.
- inference.enable_lora=true and max_lora_rank=32 — both AUTO-SET by
  RLConfig.auto_setup_lora (validates our minimal LoRA-only TOML works).
- Per-step trainer metrics: Loss ≈ -0.008, Entropy ≈ 1.24, Mismatch KL ≈ 0.0005,
  Grad. Norm ≈ 0.007, MFU 9-14%, Peak Mem 27-48 GiB on 2×A100-80GB.
- Eval Avg@1 went 0.040 (step 0) -> 0.000 (final). Noise — 5 steps is way too
  few to learn; smoke validates plumbing, not learning.

Two bugs caught + fixed during smoke iteration (both in the TOMLs, not the env code):

1. **Missing top-level [ckpt] block.** `[trainer.ckpt.weights] save_adapter_separately`
   creates trainer.ckpt but not orchestrator.ckpt, and validate_shared_ckpt_config
   rejects asymmetric configs. Fix: add empty `[ckpt]` at top level, which triggers
   auto_setup_ckpt to create both. Matches the prime-rl LoRA CI test pattern.

2. **LinearLR scheduler ZeroDivisionError on small decay_steps.** prime-rl's
   setup_linear_scheduler at trainer/scheduler.py:50 builds `LinearLR(...,
   total_iters=decay_steps - 1)`. If decay_steps=1 -> total_iters=0 -> div-by-zero
   at the first scheduler.step() that hits the decay phase. Fix: smoke uses
   constant scheduler; Phase 1 uses linear with warmup=10, decay=10 (total_iters=9,
   safe). Phase 1 TOMLs are unaffected.

--------------------------------------------------------------------------------
WHAT WAS FIXED IN THIS REVISION (from prior audit)
--------------------------------------------------------------------------------

Two showstoppers found and fixed:

1. **target_s per-rollout sampling broke GRPO group invariant.** Previously
   `setup_state` resampled target_s for every rollout, so the G=4 rollouts of
   one problem each saw different T's AND different system prompts. Fixed: T
   is now assigned once per dataset row at load time (deterministic from
   dataset_seed), and `setup_state` reads `info["target_s"]`. All G rollouts of
   one problem now share T. Across seeds, the T-assignment differs (the main
   variance source).

2. **LoRA TOML section was at the wrong path.** Previously `[model.lora]`,
   which doesn't exist on prime-rl's `SharedModelConfig`. With BaseConfig's
   `extra="forbid"`, Phase 1 would have hard-crashed at TOML validation. Fixed:
   `[trainer.model.lora]` + `[orchestrator.model.lora]` + `[inference] enable_lora`
   + `[trainer.ckpt.weights] save_adapter_separately`, matching the prime-rl
   LoRA CI test config.

Lesser audit items also addressed:

3. EXPR=result regex strip in solver — `(3+5)*4 = 32` no longer mis-buckets as quit.
4. Eval: temperature=0, num_examples=100, added T=15 and T=120 cells.
5. Trainer hyperparameters pinned explicitly (lr, max_norm, kl_tau, warmup_steps).
6. state["timing"] defensive guard.
7. final_env_response paths also set state["is_completed"]=True (belt-and-suspenders).
8. hwprop catalog assertion at env init (fail loud on misspelled hardware/model strings).
9. Two extra weight=0 diagnostic metrics: mean_n_turns and mean_completion_tokens.
10. Phase 1 TOMLs are now rendered from a single template (scripts/dev/render_phase1_tomls.py)
    to prevent copy-paste drift between the four configs.

--------------------------------------------------------------------------------
RESPONSE TO PRIOR AUDITOR (rev 2 — addressed)
--------------------------------------------------------------------------------

The previous auditor flagged five potential schema mismatches. Verified all
five against b22e768 source directly:

1. `[[orchestrator.train.env]]` vs `[[orchestrator.env]]` — CONFIRMED CORRECT at b22e768.
   Verified via examples/alphabet_sort/rl.toml and examples/reverse_text/rl.toml
   at the pinned SHA (not main; PR #1392 reshapes this on main but not at b22e768).

2. `[orchestrator.model.lora] name = ...` — Verified VALID at b22e768 (orchestrator.py:35
   defines LoRAConfig with `name`, `rank`, `alpha`). Worth noting: this block is now
   REMOVED from our TOMLs entirely because RLConfig.auto_setup_lora (rl.py:718-765)
   auto-creates it with `name = f"r{{rank}}-a{{alpha}}"` when [trainer.model.lora] is set.

3. `[trainer.model.lora]` vs `[trainer.model.experimental.lora]` — Verified CORRECT
   at b22e768 (ModelConfig.lora is a direct field, not under .experimental).
   The .experimental path is a post-b22e768 change.

4. `[trainer.ckpt.weights] save_adapter_separately = true` — Verified VALID at
   trainer.py:589 in WeightCheckpointConfig. There's even a model validator at
   rl.py:950 that REQUIRES LoRA to be enabled when this is True (consistency check).

5. `[trainer.loss] type = "default"` and `[trainer.scheduler] type = "linear"` —
   Both discriminator values verified at trainer.py:687 (DefaultLossConfig) and
   trainer.py:493 (LinearSchedulerConfig).

Additional change driven by the audit: LinearSchedulerConfig has a `decay_steps`
field that defaults to 10, meaning lr decays only over the final 10 of 200 steps.
Pinned `decay_steps = 10` explicitly so future-readers don't have to dig.

Simplified the TOML — removed the redundant `[orchestrator.model.lora]` block and
the `enable_lora = true` on `[inference]` since auto_setup_lora handles both.
This now matches the production alphabet_sort example pattern exactly.

VLLM_ALLOW_RUNTIME_LORA_UPDATING env var: not needed at our level — the inference
server sets it itself (`src/prime_rl/inference/server.py`). CI tests set it
manually only because they spawn inference differently.

--------------------------------------------------------------------------------
WHAT WAS VERIFIED PRE-AUDIT
--------------------------------------------------------------------------------

Source-verified against prime-rl@b22e768:
- TOML key paths for: seq_len (top), seed (orchestrator), batch_size, lr (trainer.optim),
  kl_tau (trainer.loss), warmup_steps (trainer.scheduler), lora (trainer.model.lora).
- BaseConfig has extra="forbid": unknown keys hard-crash on launch.
- Loss config is a discriminated union; we set `type = "default"` explicitly.

Source-verified against hwprop:
- "A100_80GB" and "Qwen2.5-3B" exist in the catalog.
- simulate_latency(hardware, model, prompt_len=, decode_steps=) signature is exact.
- r.prefill_time_s and r.total_decode_time_s are real attributes.

Source-verified against verifiers framework:
- Dataset row `info` field threads to `state["info"]` automatically.
- Reward funcs receive `state`, `answer`, `info` via kwargs (signature introspection).
- `@vf.reward` + Rubric(funcs=, weights=) — weights argument takes precedence.
- `state["timing"].model.spans` is the right path; populated on every model turn.

Local sanity:
- data/train.jsonl (9999 rows) and data/eval.jsonl (498 rows) have the expected shape.
- per-row target_s sampler is deterministic in (dataset_seed, idx) and produces
  log-uniform values across [15, 120] (verified numerically).
- _solver regex strip correctly handles `(3+5)*4 = 32` -> validates True.

--------------------------------------------------------------------------------
WHAT'S NOT IN THIS BUNDLE
--------------------------------------------------------------------------------

- prime-rl source (cloned in Modal image; pinned SHA above).
- verifiers source (cloned + installed via prime-rl's deps/verifiers submodule).
- hwprop source (cloned in Modal image; install line in modal_app.py).
- HF model weights (downloaded by trainer on first run).
- Dataset jsonls (in repo at data/train.jsonl and data/eval.jsonl).

--------------------------------------------------------------------------------
WHAT WOULD BE GOOD FOR YOU (AUDITOR) TO CHECK
--------------------------------------------------------------------------------

1. The per-row target_s assignment in interoception_countdown.py:_load_dataset —
   is this actually shared across GRPO group rollouts? Confirm by reading
   setup_state and the verifiers env contract.

2. The four TOML files — any remaining schema mismatches against prime-rl's
   Pydantic config classes that would cause `extra_forbidden` at launch?

3. The reward shape implementation — any edge cases in correctness_with_time
   or parseable_bonus that would silently misweight a rollout?

4. The final_env_response + is_completed early-termination — any way this
   double-fires or fails to terminate?

5. Anything else: the goal is to find things that don't show up in a 5-step
   smoke but would silently break 200-step Phase 1.

================================================================================
FILES
================================================================================

"""


def main() -> None:
    chunks = [PREAMBLE.format(ts=datetime.now().isoformat(timespec="seconds"))]
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            chunks.append(f"\n[MISSING] {rel}\n")
            continue
        chunks.append(f"\n================== FILE: {rel} ==================\n")
        chunks.append(path.read_text())
    OUT.write_text("".join(chunks))
    n_lines = OUT.read_text().count("\n")
    n_bytes = OUT.stat().st_size
    print(f"wrote {OUT} ({n_lines} lines, {n_bytes} bytes)")


if __name__ == "__main__":
    main()
