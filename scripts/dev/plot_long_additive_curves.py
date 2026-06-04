"""Smoothed training curves for long-additive (Qwen3-4B + remaining_budget prompt,
additive reward c + 0.5·f, 100 steps).

Mirror of plot_long_strict_curves.py — single wandb session, no stitching needed.
The wandb run state shows "crashed" because of the wandb teardown
ConnectionResetError, but the 100-step training trajectory is complete.

Note: the reward shape is ADDITIVE here (c + 0.5·f), not multiplicative.
The "reward" curve is labeled accordingly.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "singhh5050-stanford-university/interoception"
RUN = "ctrl0-qwen3-4b-u1-40-long-additive"
ENV = "ctrl0-u1-40-long-additive-qwen3-4b-train"
P = f"metrics/{ENV}/"
SERIES = [
    ("correctness c",       P + "is_correct",       "#1f77b4"),
    ("f(t, T)",             P + "f_term",            "#2ca02c"),
    ("reward  c + 0.5·f",   f"reward/{ENV}/mean",    "#d62728"),
]
WIN = 10

api = wandb.Api()
runs_chrono = sorted(api.runs(ENTITY, filters={"display_name": RUN}),
                     key=lambda r: r.created_at)
run = runs_chrono[-1]
print(f"using wandb run {run.id} (state={run.state})")


def pull(key):
    pairs = []
    for row in run.scan_history(keys=["_step", key], page_size=10000):
        if row.get("_step") is not None and row.get(key) is not None:
            pairs.append((int(row["_step"]), float(row[key])))
    pairs.sort()
    ys = [v for _, v in pairs]
    return np.arange(len(ys)), np.array(ys)


def runavg(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, (label, key, color) in zip(axes, SERIES):
    x, y = pull(key)
    ax.plot(x, y, color=color, alpha=0.22, lw=1.0)
    if len(y) >= WIN:
        xs = x[WIN - 1:]
        ax.plot(xs, runavg(y, WIN), color=color, lw=2.6, label=f"{WIN}-step running avg")
    ax.set_ylabel(label, fontsize=10.5)
    ax.set_ylim(0, max(0.6, float(np.nanmax(y)) * 1.1) if len(y) else 0.6)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    final_val = runavg(y, WIN)[-1] if len(y) >= WIN else (y[-1] if len(y) else 0)
    ax.text(0.99, 0.06, f"final≈{final_val:.2f}", transform=ax.transAxes,
            ha="right", fontsize=9, color=color, fontweight="bold")

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen3-4B — T~U(1,40), 100 steps with additive reward (c + 0.5·f) + remaining_budget prompt",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/41_long_additive_smoothed_curves.png"
fig.savefig(out, dpi=140)
print("wrote", out)
