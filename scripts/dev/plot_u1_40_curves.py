"""Smoothed training curves for ctrl0_u1_40: correctness, f(t,T), reward.
10-step running average (Kanishk's request, 2026-05-26 thread)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "singhh5050-stanford-university/interoception"
RUN = "ctrl0-qwen3-4b-u1-40"          # run display name (for api filter)
ENV = "ctrl0-u1-40-qwen3-4b-train"    # env name (the metric namespace prefix)
P = f"metrics/{ENV}/"
SERIES = [
    ("correctness c", P + "is_correct",       "#1f77b4"),
    ("f(t, T)",       P + "f_term",            "#2ca02c"),
    ("reward c·f",    f"reward/{ENV}/mean",    "#d62728"),
]
WIN = 10

api = wandb.Api()
run = sorted(api.runs(ENTITY, filters={"display_name": RUN}),
             key=lambda r: r.created_at, reverse=True)[0]

def pull(key):
    # _step is always present; metrics/ namespace isn't co-logged with the custom
    # `step` field. Each metric logs once per training step, so sorted order == step.
    d = {}
    for r in run.scan_history(keys=["_step", key]):
        if r.get("_step") is not None and r.get(key) is not None:
            d[int(r["_step"])] = float(r[key])
    ys = [d[s] for s in sorted(d)]
    return np.arange(len(ys)), np.array(ys)

def runavg(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, (label, key, color) in zip(axes, SERIES):
    x, y = pull(key)
    ax.plot(x, y, color=color, alpha=0.22, lw=1.0)                       # raw
    xs = x[WIN - 1:]
    ax.plot(xs, runavg(y, WIN), color=color, lw=2.6, label=f"{WIN}-step running avg")  # smoothed
    ax.set_ylabel(label, fontsize=10.5)
    ax.set_ylim(0, max(0.6, float(np.nanmax(y)) * 1.1))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.99, 0.06, f"final≈{runavg(y, WIN)[-1]:.2f}", transform=ax.transAxes,
            ha="right", fontsize=9, color=color, fontweight="bold")

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("T ~ U(1, 40)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/25_u1_40_smoothed_curves.png"
fig.savefig(out, dpi=140)
print("wrote", out)
