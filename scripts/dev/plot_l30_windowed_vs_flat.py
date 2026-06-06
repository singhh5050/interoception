"""Direct comparison: λ=0.30 cell under flat-1 additive (v2-l30) vs windowed
additive (windowed-l30). Same λ, same protocol otherwise. Only the f-shape
differs.

Data lives under two different wandb entities (different accounts), so we use
two wandb.Api instances each authenticated with the right key.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# v2-l30 — flat-1 additive (singhh5050 entity)
KEY_SINGH = "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"
# windowed-l30 — windowed_additive (nicolema entity)
KEY_NICO = "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"

CELLS = [
    # (label, api_key, entity/project, run_name, env_name, color)
    ("flat-1 (v2-l30, 200 steps)",
     KEY_SINGH,
     "singhh5050-stanford-university/interoception-l30",
     "ctrl0-qwen3-4b-u1-40-long-additive-v2-l30",
     "ctrl0-u1-40-long-additive-v2-l30-qwen3-4b-train",
     "#1f77b4"),
    ("windowed (windowed-l30, 100 steps)",
     KEY_NICO,
     "nicolema-stanford-university/interoception-windowed-l30",
     "ctrl0-qwen3-4b-u1-40-windowed-l30",
     "ctrl0-u1-40-windowed-l30-qwen3-4b-train",
     "#d62728"),
]
WIN = 10


def fetch_cell(api_key, project_path, run_name, env_name):
    api = wandb.Api(api_key=api_key)
    runs = sorted(api.runs(project_path, filters={"display_name": run_name}),
                  key=lambda r: r.created_at)
    if not runs:
        print(f"  no run found for {project_path}/{run_name}")
        return None
    run = runs[-1]
    print(f"  {project_path}: run {run.id} (state={run.state})")
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


print("Fetching wandb metrics for 2 cells...")
data = {}
for label, api_key, project_path, run_name, env_name, color in CELLS:
    print(f"--- {label} ---")
    data[label] = (color, env_name, fetch_cell(api_key, project_path, run_name, env_name))

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
panels = [
    ("correctness c",                "is_correct", 0),
    ("f(t, T)  [different shapes!]", "f_term",     1),
    ("reward  c + 0.30·f",           "mean",       2),
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
            ax.plot(sm_x, sm, color=color, lw=2.8,
                    label=f"{label}\n  final≈{sm[-1]:.2f}, n={len(ys)}")
        else:
            ax.plot(x, ys, color=color, lw=2.8, label=f"{label} (n={len(ys)})")
    ax.set_ylabel(panel_label, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen3-4B, λ=0.30  —  flat-1 vs windowed f-shape "
             "(otherwise identical: G=16/batch=128, remaining_budget prompt)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/47_l30_windowed_vs_flat.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
