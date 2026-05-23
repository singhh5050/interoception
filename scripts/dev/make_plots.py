"""Generate standard RL training visualizations from the sweep data.

Reads analysis/data/<cell>.json (one per cell, full metric history) and writes
PNG plots into analysis/figures/.

Plots produced:
  01_headline_pacing.png         — final eval correctness vs T-budget (THE figure)
  02_eval_curves_per_T.png       — eval Avg@1 over training steps, one panel per T
  03_train_reward_curves.png     — train reward over training steps, all cells
  04_train_correctness_curves.png — is_correct (per-cell train) over time
  05_diagnostic_breakdown.png    — is_correct / is_quit / is_timeout / is_parseable per cell
  06_pacing_at_final_step.png    — pacing slope per cell (correct% vs log(T))
  07_loss_entropy_gradnorm.png   — training dynamics health
  08_turns_and_completion.png    — mean_n_turns + completion_len behavior

Convention: red = qwen2.5-3b cells, blue = qwen3-4b cells; solid = hyp, dashed = exp.
"""
from __future__ import annotations
import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

DATA = pathlib.Path("analysis/data")
OUT = pathlib.Path("analysis/figures")
OUT.mkdir(parents=True, exist_ok=True)

CELLS = [
    "qwen25-3b-hyp-s0", "qwen25-3b-hyp-s1", "qwen25-3b-exp-s0", "qwen25-3b-exp-s1",
    "qwen3-4b-hyp-s0",  "qwen3-4b-hyp-s1",  "qwen3-4b-exp-s0",  "qwen3-4b-exp-s1",
]
TBUDGETS = [15, 30, 60, 120]


def load_cell(name: str) -> dict | None:
    p = DATA / f"{name}.json"
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    # Filter cells with effectively no training data (e.g. backfill still spinning up)
    if d.get("n_steps", 0) < 50:
        return None
    return d


def style_for(name: str) -> dict:
    """Color by model, linestyle by reward shape, alpha by seed."""
    if "qwen25-3b" in name:
        color = "#c0392b" if "hyp" in name else "#e74c3c"  # red family
    else:  # qwen3-4b
        color = "#2c3e50" if "hyp" in name else "#3498db"  # blue family
    ls = "-" if "hyp" in name else "--"
    alpha = 1.0 if "s0" in name else 0.65
    return dict(color=color, linestyle=ls, alpha=alpha, linewidth=1.7,
                label=name)


def last_value(metric_list: list[tuple[int, float]]) -> float | None:
    if not metric_list:
        return None
    return metric_list[-1][1]


def history(d: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for one metric."""
    rows = d["metrics"].get(key, [])
    if not rows:
        return np.array([]), np.array([])
    rows = sorted(rows, key=lambda x: x[0])
    s = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    return s, v


def get_train_metric(d: dict, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """The train metrics are namespaced by env name. Find the right key."""
    name = d["name"]
    # Try a few naming patterns
    for tpl in [
        f"metrics/{name}-train/{suffix}",
        f"metrics/{name}/{suffix}",
    ]:
        if tpl in d["metrics"]:
            return history(d, tpl)
    # Fallback: search any metrics/*/<suffix>
    for k in d["metrics"]:
        if k.endswith(f"/{suffix}") and k.startswith("metrics/"):
            return history(d, k)
    return np.array([]), np.array([])


# Load everything once
loaded = {n: load_cell(n) for n in CELLS}
ACTIVE = [n for n in CELLS if loaded[n] is not None]
print(f"Loaded {len(ACTIVE)} cells with substantial data: {ACTIVE}")


# ============================================================================
# 01: Headline figure — final eval correctness (Avg@1) vs T-budget, per cell
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
xs = np.arange(len(TBUDGETS))
width = 0.10
n = len(ACTIVE)
for i, name in enumerate(ACTIVE):
    d = loaded[name]
    vals = []
    for T in TBUDGETS:
        _, v = history(d, f"eval/eval-t{T}/avg@1")
        vals.append(v[-1] if len(v) else np.nan)
    offset = (i - n / 2 + 0.5) * width
    bars = ax.bar(xs + offset, vals, width, **{k: v for k, v in style_for(name).items()
                                                 if k not in ("linestyle", "linewidth", "alpha")},
                  edgecolor="black", linewidth=0.4)
    # Custom alpha (bar doesn't pick up from style_for dict cleanly)
    for b in bars:
        b.set_alpha(style_for(name)["alpha"])
ax.set_xticks(xs)
ax.set_xticklabels([f"T={T}s" for T in TBUDGETS])
ax.set_ylabel("Eval Avg@1 at final checkpoint (rubric reward)")
ax.set_title("Pacing as a function of T-budget (final checkpoint)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
fig.tight_layout()
fig.savefig(OUT / "01_headline_pacing.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '01_headline_pacing.png'}")


# ============================================================================
# 02: Eval Avg@1 over training steps — one panel per T-budget
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
for ax, T in zip(axes.flat, TBUDGETS):
    for name in ACTIVE:
        d = loaded[name]
        s, v = history(d, f"eval/eval-t{T}/avg@1")
        if len(s):
            ax.plot(s, v, **style_for(name))
    ax.set_title(f"Eval Avg@1 at T={T}s over training")
    ax.set_xlabel("wandb _step")
    ax.set_ylabel("Avg@1 (rubric reward)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.6)
axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "02_eval_curves_per_T.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '02_eval_curves_per_T.png'}")


# ============================================================================
# 03: Training reward over training steps (all cells overlaid)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
for name in ACTIVE:
    d = loaded[name]
    s, v = history(d, "reward/all/mean")
    if len(s):
        ax.plot(s, v, **style_for(name))
ax.set_xlabel("wandb _step")
ax.set_ylabel("Train reward (per-batch mean)")
ax.set_title("Training reward over RL steps")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "03_train_reward_curves.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '03_train_reward_curves.png'}")


# ============================================================================
# 04: True train correctness (is_correct) over time — raw signal, no time decay
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
for name in ACTIVE:
    d = loaded[name]
    s, v = get_train_metric(d, "is_correct")
    if len(s):
        ax.plot(s, v, **style_for(name))
ax.set_xlabel("wandb _step")
ax.set_ylabel("is_correct (fraction of train rollouts)")
ax.set_title("Training-rollout correctness over RL steps (raw signal)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "04_train_correctness_curves.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '04_train_correctness_curves.png'}")


# ============================================================================
# 05: Bucket diagnostics per cell (correct / wrong / quit / timeout / parseable)
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
bucket_colors = {
    "is_correct":   "#27ae60",
    "is_wrong":     "#f39c12",
    "is_quit":      "#7f8c8d",
    "is_timeout":   "#c0392b",
    "is_parseable": "#2980b9",
}
for ax, name in zip(axes.flat, ACTIVE):
    d = loaded[name]
    for bucket, color in bucket_colors.items():
        s, v = get_train_metric(d, bucket)
        if len(s):
            ax.plot(s, v, color=color, label=bucket, linewidth=1.4)
    ax.set_title(name, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("wandb _step")
axes[0, 0].legend(loc="upper left", fontsize=7)
fig.suptitle("Per-cell training-rollout outcome breakdown", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "05_diagnostic_breakdown.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '05_diagnostic_breakdown.png'}")


# ============================================================================
# 06: Pacing slope — final eval correctness vs log2(T), one line per cell
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for name in ACTIVE:
    d = loaded[name]
    finals = []
    for T in TBUDGETS:
        _, v = history(d, f"eval/eval-t{T}/avg@1")
        finals.append(v[-1] if len(v) else np.nan)
    if any(not math.isnan(x) for x in finals):
        ax.plot(TBUDGETS, finals, marker="o", markersize=6, **style_for(name))
ax.set_xscale("log", base=2)
ax.set_xticks(TBUDGETS)
ax.set_xticklabels([str(T) for T in TBUDGETS])
ax.set_xlabel("Target time budget T (seconds, log scale)")
ax.set_ylabel("Eval Avg@1 at final checkpoint")
ax.set_title("Pacing curve: does eval correctness scale with T?")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0, 0.6)
fig.tight_layout()
fig.savefig(OUT / "06_pacing_at_final_step.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '06_pacing_at_final_step.png'}")


# ============================================================================
# 07: Loss / entropy / grad-norm (training dynamics health)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
keys_n_label = [
    ("loss/mean", "Loss (mean)"),
    ("entropy/mean", "Entropy (mean)"),
    ("optim/grad_norm", "Gradient norm"),
]
for ax, (key, label) in zip(axes, keys_n_label):
    for name in ACTIVE:
        d = loaded[name]
        s, v = history(d, key)
        if len(s):
            ax.plot(s, v, **style_for(name))
    ax.set_xlabel("wandb _step")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.set_title(label)
axes[0].legend(loc="upper right", fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "07_loss_entropy_gradnorm.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '07_loss_entropy_gradnorm.png'}")


# ============================================================================
# 08: Behavior: turns and completion length
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for name in ACTIVE:
    d = loaded[name]
    s, v = get_train_metric(d, "mean_n_turns")
    if len(s):
        ax.plot(s, v, **style_for(name))
ax.set_xlabel("wandb _step")
ax.set_ylabel("mean turns per train rollout")
ax.set_title("How many turns does the model take?")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
for name in ACTIVE:
    d = loaded[name]
    s, v = get_train_metric(d, "elapsed_over_target")
    if len(s):
        ax.plot(s, v, **style_for(name))
ax.set_xlabel("wandb _step")
ax.set_ylabel("elapsed_s / target_s (train rollouts)")
ax.set_title("Are rollouts in budget? (1.0 = at deadline)")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.6, label="budget")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "08_turns_and_completion.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '08_turns_and_completion.png'}")


# ============================================================================
# 09: Final-checkpoint truncation rate (data integrity sanity)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(len(TBUDGETS))
n = len(ACTIVE)
for i, name in enumerate(ACTIVE):
    d = loaded[name]
    vals = []
    for T in TBUDGETS:
        _, v = history(d, f"eval/eval-t{T}/is_truncated/mean")
        vals.append(v[-1] if len(v) else np.nan)
    offset = (i - n / 2 + 0.5) * 0.10
    bars = ax.bar(xs + offset, vals, 0.10, color=style_for(name)["color"],
                  alpha=style_for(name)["alpha"], edgecolor="black", linewidth=0.4,
                  label=name)
ax.set_xticks(xs)
ax.set_xticklabels([f"T={T}s" for T in TBUDGETS])
ax.set_ylabel("is_truncated/mean (eval rollouts)")
ax.set_title("Eval truncation rate — fraction of rollouts hitting max_completion_tokens")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(OUT / "09_eval_truncation.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '09_eval_truncation.png'}")

print("\nDone. Figures written to analysis/figures/")
