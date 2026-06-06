"""Smoothed training curves for the stage-2 KL-anchored β sweep (5 cells:
kl_tau ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}, all from the v2-l30-merged base,
c-only reward, 200 steps, G=16/batch=128).

Reward = c since lambda_f = 0 in env args. We still plot the reward panel
for parity with figure 43 (so it's easy to compare side-by-side), even though
it's redundant with the c panel here.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "singhh5050-stanford-university"
CELLS = [
    ("β=0  (no anchor)", "interoception-stage2-b0", "stage2-kl-b0",
     "stage2-kl-b0-train", "#1f77b4"),
    ("β=1e-4",           "interoception-stage2-b4", "stage2-kl-b4",
     "stage2-kl-b4-train", "#2ca02c"),
    ("β=1e-3",           "interoception-stage2-b3", "stage2-kl-b3",
     "stage2-kl-b3-train", "#d62728"),
    ("β=1e-2",           "interoception-stage2-b2", "stage2-kl-b2",
     "stage2-kl-b2-train", "#9467bd"),
    ("β=1e-1 (heavy)",   "interoception-stage2-b1", "stage2-kl-b1",
     "stage2-kl-b1-train", "#8c564b"),
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


print("Fetching wandb metrics for 5 cells...")
data = {}
for label, project, run_name, env_name, color in CELLS:
    print(f"--- {label} ---")
    data[label] = (color, env_name, fetch_cell(project, run_name, env_name))

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
panels = [
    ("correctness c",     "is_correct", 0),
    ("f(t, T)",           "f_term",     1),
    ("reward (= c here)", "mean",       2),
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
        ax.plot(x, ys, color=color, alpha=0.15, lw=1.0)
        if len(ys) >= WIN:
            sm = runavg(ys, WIN)
            sm_x = x[WIN - 1:]
            ax.plot(sm_x, sm, color=color, lw=2.4,
                    label=f"{label} (final≈{sm[-1]:.2f}, n={len(ys)})")
        else:
            ax.plot(x, ys, color=color, lw=2.4, label=f"{label} (n={len(ys)})")
    ax.set_ylabel(panel_label, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

# Reference lines for the stage-1 anchor policy (v2-l30 step_200):
# c ≈ 0.17 matched, f ≈ 0.86. These tell us "where we started" — distance
# from these lines shows how far each β allowed the policy to drift.
ANCHOR_REFS = {
    0: ("stage-1 anchor (v2-l30) c ≈ 0.17", 0.17),
    1: ("stage-1 anchor (v2-l30) f ≈ 0.86", 0.86),
}
for ax_idx, (label, val) in ANCHOR_REFS.items():
    axes[ax_idx].axhline(val, color="#666", lw=1.5, ls="--", label=label)
    axes[ax_idx].legend(loc="upper left", fontsize=9, framealpha=0.9)

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen3-4B — stage-2 KL-anchored β sweep (200 steps, c-only, anchor=v2-l30)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/45_stage2_kl_sweep_curves.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
