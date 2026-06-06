"""Smoothed training curves for the long-additive-v2 λ sweep (3 cells:
λ ∈ {0.10, 0.15, 0.30}, otherwise identical: 200 steps, G=16/batch=128,
additive reward c + λ·f, prompt_variant=remaining_budget).

Each cell's wandb run lives in its own project (interoception-l10/l15/l30)
because prime-rl's run-id derivation collided otherwise. We fetch each, smooth,
and overlay on a 3-panel chart: correctness c, f(t,T), and reward.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "singhh5050-stanford-university"
CELLS = [
    ("λ=0.10", "interoception-l10", "ctrl0-qwen3-4b-u1-40-long-additive-v2-l10",
     "ctrl0-u1-40-long-additive-v2-l10-qwen3-4b-train", "#1f77b4"),
    ("λ=0.15", "interoception-l15", "ctrl0-qwen3-4b-u1-40-long-additive-v2-l15",
     "ctrl0-u1-40-long-additive-v2-l15-qwen3-4b-train", "#2ca02c"),
    ("λ=0.30", "interoception-l30", "ctrl0-qwen3-4b-u1-40-long-additive-v2-l30",
     "ctrl0-u1-40-long-additive-v2-l30-qwen3-4b-train", "#d62728"),
]
WIN = 10  # smoothing window

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


# Fetch all 3 cells
print("Fetching wandb metrics for 3 cells...")
data = {}
for label, project, run_name, env_name, color in CELLS:
    print(f"--- {label} ---")
    data[label] = (color, env_name, fetch_cell(project, run_name, env_name))

# 3-panel chart: c, f, reward
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
panels = [
    ("correctness c",        "is_correct", 0),
    ("f(t, T)",              "f_term",     1),
    ("reward  c + λ·f",      "mean",       2),
]
for panel_label, metric_kind, ax_idx in panels:
    ax = axes[ax_idx]
    for label, (color, env_name, series) in data.items():
        if series is None:
            continue
        # find matching key
        key = next((k for k in series if k.endswith(f"/{metric_kind}")
                    or (metric_kind == "mean" and k.endswith("/mean"))), None)
        if key is None or not series[key]:
            continue
        pairs = series[key]
        ys = np.array([p[1] for p in pairs])
        # x = training step (index in sorted-by-_step list), since _step is wandb's
        # internal global log counter (incremented per log call, not per training step).
        x = np.arange(len(ys))
        # raw faint line
        ax.plot(x, ys, color=color, alpha=0.18, lw=1.0)
        # smoothed
        if len(ys) >= WIN:
            sm = runavg(ys, WIN)
            sm_x = x[WIN - 1:]
            ax.plot(sm_x, sm, color=color, lw=2.6, label=f"{label} (final≈{sm[-1]:.2f}, n={len(ys)})")
        else:
            ax.plot(x, ys, color=color, lw=2.6, label=f"{label} (n={len(ys)})")
    ax.set_ylabel(panel_label, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

axes[-1].set_xlabel("training step", fontsize=11)

# --- Reference lines for the base model + loud prompt (no RL, no trajectory) ---
# c ≈ 0.18 is approximate (from earlier probes). f = 0.843 is computed precisely
# from base_remaining_budget.jsonl (mean of min(1, T/t) across all 498 rollouts).
BASE_LOUD_REFS = {
    0: ("base + loud prompt c ≈ 0.18", 0.18),
    1: ("base + loud prompt f = 0.84", 0.843),
}
for ax_idx, (label, val) in BASE_LOUD_REFS.items():
    axes[ax_idx].axhline(val, color="#666", lw=1.5, ls="--", label=label)
    axes[ax_idx].legend(loc="upper left", fontsize=9, framealpha=0.9)

fig.suptitle("Qwen3-4B — additive reward sweep (200 steps, G=16/batch=128, +remaining_budget prompt)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/43_long_additive_v2_sweep_curves.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
