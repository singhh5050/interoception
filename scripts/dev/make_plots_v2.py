"""V2 plots — use the rigorously-computed eval rollout stats instead of wandb Avg@1.

After auditing the saved eval_rollouts.jsonl files, we found Avg@1 systematically
under-reports correctness because it folds in time-decay and parseable bonus.
This script uses analysis/data/final_eval_stats.json (computed by audit_eval_rollouts.py)
which has per-cell, per-T raw stats.
"""
from __future__ import annotations
import json
import math
import pathlib
import matplotlib.pyplot as plt
import numpy as np

DATA = pathlib.Path("analysis/data/final_eval_stats.json")
OUT = pathlib.Path("analysis/figures")
OUT.mkdir(parents=True, exist_ok=True)

with DATA.open() as f:
    stats = json.load(f)

TBUDGETS = [15, 30, 60, 120]
CELL_ORDER = [
    "qwen25-3b-hyp-s0", "qwen25-3b-hyp-s1", "qwen25-3b-exp-s0", "qwen25-3b-exp-s1",
    "qwen3-4b-hyp-s0",                       "qwen3-4b-exp-s0",  "qwen3-4b-exp-s1",
]
CELL_ORDER = [c for c in CELL_ORDER if c in stats]


def style_for(name: str) -> dict:
    if "qwen25-3b" in name:
        color = "#c0392b" if "hyp" in name else "#e74c3c"
    else:
        color = "#2c3e50" if "hyp" in name else "#3498db"
    ls = "-" if "hyp" in name else "--"
    alpha = 1.0 if "s0" in name else 0.65
    return dict(color=color, linestyle=ls, alpha=alpha, linewidth=1.7, label=name)


# Helper: get a metric value safely
def gv(cell, T, metric):
    d = stats[cell].get(f"T={T}")
    return None if d is None else d.get(metric)


# ============================================================================
# 10: HEADLINE V2 — raw is_correct vs T, per cell. Bar chart.
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
xs = np.arange(len(TBUDGETS))
n = len(CELL_ORDER)
width = 0.85 / n
for i, name in enumerate(CELL_ORDER):
    vals = [gv(name, T, "is_correct") for T in TBUDGETS]
    s = style_for(name)
    offset = (i - n/2 + 0.5) * width
    bars = ax.bar(xs + offset, vals, width, color=s["color"], alpha=s["alpha"],
                  edgecolor="black", linewidth=0.4, label=name)
ax.set_xticks(xs)
ax.set_xticklabels([f"T={T}s" for T in TBUDGETS])
ax.set_ylabel("Raw correctness (is_correct, fraction of eval rollouts)")
ax.set_title("Pacing-via-RL: true eval correctness vs T-budget (final checkpoint)\n"
             "Computed directly from saved eval rollouts (not wandb Avg@1)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 0.6)
# Annotate the bars
for i, name in enumerate(CELL_ORDER):
    for j, T in enumerate(TBUDGETS):
        v = gv(name, T, "is_correct")
        if v is None: continue
        x = xs[j] + (i - n/2 + 0.5) * width
        ax.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=6, color="black", rotation=90)
fig.tight_layout()
fig.savefig(OUT / "10_headline_is_correct.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '10_headline_is_correct.png'}")


# ============================================================================
# 11: REWARD DECOMPOSITION — stacked: correctness_with_time + parseable bonus
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, model_prefix, label in [(axes[0], "qwen25-3b", "Qwen2.5-3B (4 cells)"),
                                  (axes[1], "qwen3-4b", "Qwen3-4B (3 cells)")]:
    cells = [c for c in CELL_ORDER if c.startswith(model_prefix)]
    n = len(cells)
    width = 0.85 / n
    for i, name in enumerate(cells):
        is_corr = [gv(name, T, "is_correct") or 0 for T in TBUDGETS]
        cwt = [gv(name, T, "correctness_with_time") or 0 for T in TBUDGETS]
        late = [c - w for c, w in zip(is_corr, cwt)]  # correct-but-late: contributes to is_correct, partially to cwt
        # 0.05 * parseable contribution
        parse_contrib = [(gv(name, T, "parseable") or 0) * 0.05 for T in TBUDGETS]
        s = style_for(name)
        offset = (i - n/2 + 0.5) * width
        # Stack: correct-and-in-budget (= cwt for correct) + correct-but-late (decay residual) + parseable-bonus
        ax.bar(np.arange(4) + offset, cwt, width, color=s["color"], alpha=s["alpha"],
               edgecolor="black", linewidth=0.4, label=name)
        # Show wandb Avg@1 as a small marker on top
        avg1 = [gv(name, T, "wandb_avg_at_1") for T in TBUDGETS]
        ax.scatter(np.arange(4) + offset, avg1, marker="_", color="black", s=60, zorder=10)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"T={T}s" for T in TBUDGETS])
    ax.set_title(f"{label}\n(bars = correctness_with_time, marks = wandb Avg@1)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("Reward / correctness")
fig.tight_layout()
fig.savefig(OUT / "11_reward_decomposition.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '11_reward_decomposition.png'}")


# ============================================================================
# 12: PACING SLOPE V2 with raw correctness
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for name in CELL_ORDER:
    vals = [gv(name, T, "is_correct") for T in TBUDGETS]
    ax.plot(TBUDGETS, vals, marker="o", markersize=7, **style_for(name))
ax.set_xscale("log", base=2)
ax.set_xticks(TBUDGETS)
ax.set_xticklabels([str(T) for T in TBUDGETS])
ax.set_xlabel("Target budget T (seconds, log scale)")
ax.set_ylabel("Raw correctness (is_correct, fraction)")
ax.set_title("Pacing curve: raw correctness vs T")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0, 0.6)
fig.tight_layout()
fig.savefig(OUT / "12_pacing_slope_raw.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '12_pacing_slope_raw.png'}")


# ============================================================================
# 13: TIMEOUT RATE vs T — model failing to commit at all
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for name in CELL_ORDER:
    vals = [gv(name, T, "is_timeout") for T in TBUDGETS]
    ax.plot(TBUDGETS, vals, marker="o", markersize=7, **style_for(name))
ax.set_xscale("log", base=2)
ax.set_xticks(TBUDGETS)
ax.set_xticklabels([str(T) for T in TBUDGETS])
ax.set_xlabel("Target budget T (seconds)")
ax.set_ylabel("Timeout rate (no <answer> emitted)")
ax.set_title("Timeout rate vs T — does the model commit more with more time?")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "13_timeout_rate_vs_T.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '13_timeout_rate_vs_T.png'}")


# ============================================================================
# 14: ELAPSED/TARGET ratio — pacing honesty (does the model finish in budget?)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for name in CELL_ORDER:
    vals = [gv(name, T, "elapsed_over_target") for T in TBUDGETS]
    ax.plot(TBUDGETS, vals, marker="o", markersize=7, **style_for(name))
ax.axhline(1.0, color="black", linestyle=":", alpha=0.6, label="budget")
ax.set_xscale("log", base=2)
ax.set_xticks(TBUDGETS)
ax.set_xticklabels([str(T) for T in TBUDGETS])
ax.set_xlabel("Target budget T (seconds)")
ax.set_ylabel("elapsed_s / target_s (mean across rollouts)")
ax.set_title("Pacing honesty: does the model finish within T? (1.0 = at deadline)")
ax.legend(loc="upper right", fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0, 6)
fig.tight_layout()
fig.savefig(OUT / "14_elapsed_over_target.png", dpi=150)
plt.close(fig)
print(f"  wrote {OUT / '14_elapsed_over_target.png'}")

print("\nDone.")
