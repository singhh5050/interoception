"""Smoothed training curves for the windowed-reward λ sweep (3 cells:
λ ∈ {0.15, 0.30, 0.50}, all fresh from Qwen3-4B base, reward_shape =
windowed_additive with asymmetric Gaussian f peaked at t=T,
σ_under=0.25, σ_over=0.10).

Tests Kanishk's hypothesis: under the windowed f-shape, the bimodal λ regime
seen with flat-1 additive should disappear, producing a smooth (c, r) tradeoff
curve across λ.

Wandb runs land under entity nicolema-stanford-university (new account for
these experiments).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "nicolema-stanford-university"
CELLS = [
    ("λ=0.15", "interoception-windowed-l15", "ctrl0-qwen3-4b-u1-40-windowed-l15",
     "ctrl0-u1-40-windowed-l15-qwen3-4b-train", "#1f77b4"),
    ("λ=0.30", "interoception-windowed-l30", "ctrl0-qwen3-4b-u1-40-windowed-l30",
     "ctrl0-u1-40-windowed-l30-qwen3-4b-train", "#2ca02c"),
    ("λ=0.50", "interoception-windowed-l50", "ctrl0-qwen3-4b-u1-40-windowed-l50",
     "ctrl0-u1-40-windowed-l50-qwen3-4b-train", "#d62728"),
]
WIN = 10
api = wandb.Api()


def fetch_cell(project, run_name, env_name):
    runs = sorted(api.runs(f"{ENTITY}/{project}", filters={"display_name": run_name}),
                  key=lambda r: r.created_at)
    if not runs:
        print(f"  no run found for {project}/{run_name}")
        return None
    run = runs[-1]
    print(f"  {project}: run {run.id} (state={run.state})")
    keys = [
        f"metrics/{env_name}/is_correct",
        f"metrics/{env_name}/f_term",
        f"reward/{env_name}/mean",
    ]
    out = {k: [] for k in keys}
    for row in run.scan_history(keys=["_step"] + keys, page_size=10000):
        if row.get("_step") is None:
            continue
        for k in keys:
            v = row.get(k)
            if v is not None:
                out[k].append((int(row["_step"]), float(v)))
    return {k: sorted(v) for k, v in out.items()}


def runavg(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


print("Fetching wandb metrics for 3 cells...")
data = {}
for label, project, run_name, env_name, color in CELLS:
    print(f"--- {label} ---")
    data[label] = (color, env_name, fetch_cell(project, run_name, env_name))

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
panels = [
    ("correctness c",      "is_correct", 0),
    ("f(t, T)  [windowed]","f_term",     1),
    ("reward c + λ·f",     "mean",       2),
]
for panel_label, metric_kind, ax_idx in panels:
    ax = axes[ax_idx]
    for label, (color, env_name, series) in data.items():
        if series is None:
            continue
        key = next((k for k in series if k.endswith(f"/{metric_kind}")
                    or (metric_kind == "mean" and k.endswith("/mean"))), None)
        if key is None or not series[key]:
            continue
        pairs = series[key]
        ys = np.array([p[1] for p in pairs])
        x = np.arange(len(ys))
        ax.plot(x, ys, color=color, alpha=0.18, lw=1.0)
        if len(ys) >= WIN:
            sm = runavg(ys, WIN)
            sm_x = x[WIN - 1:]
            ax.plot(sm_x, sm, color=color, lw=2.6,
                    label=f"{label} (final≈{sm[-1]:.2f}, n={len(ys)})")
        else:
            ax.plot(x, ys, color=color, lw=2.6, label=f"{label} (n={len(ys)})")
    ax.set_ylabel(panel_label, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

# Reference lines from probe data on base + remaining_budget:
# c ≈ 0.18 estimated from earlier probes; f = 0.843 computed from
# base_remaining_budget.jsonl (mean of min(1, T/t) over all 498 rollouts).
# Note: f reference uses the OLD flat-1 shape — under windowed_additive
# the base model's f would compute to a smaller value (since it commits
# under T and gets a Gaussian-shape penalty). But for cross-experiment
# comparison the flat-1 reference is the meaningful baseline.
BASE_REFS = {
    0: ("base + loud prompt c ≈ 0.18", 0.18),
}
for ax_idx, (label, val) in BASE_REFS.items():
    axes[ax_idx].axhline(val, color="#666", lw=1.5, ls="--", label=label)
    axes[ax_idx].legend(loc="upper left", fontsize=9, framealpha=0.9)

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen3-4B — windowed-reward sweep (100 steps, G=16/batch=128, "
             "σ_under=0.25, σ_over=0.10)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/46_windowed_sweep_curves.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
