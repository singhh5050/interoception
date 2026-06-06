"""Overlay comparison: all 3 v2 cells (flat-1 additive, λ ∈ {0.10, 0.15, 0.30},
200 steps) and all 3 windowed cells (windowed_additive, λ ∈ {0.15, 0.30, 0.50},
100 steps) on the same axes.

Color = lambda value. Solid = flat-1 (v2), dotted = windowed.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

KEY_SINGH = "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"
KEY_NICO = "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"

# Color by lambda. Shared lambdas (0.15, 0.30) use matching colors between
# the v2 and windowed conditions, with the windowed variant dotted.
LAMBDA_COLORS = {
    0.10: "#1f77b4",  # blue
    0.15: "#2ca02c",  # green
    0.30: "#d62728",  # red
    0.50: "#9467bd",  # purple
}

CELLS = [
    # (label, api_key, project_path, run_name, env_name, lambda, linestyle)
    ("v2 (flat-1) λ=0.10",
     KEY_SINGH, "singhh5050-stanford-university/interoception-l10",
     "ctrl0-qwen3-4b-u1-40-long-additive-v2-l10",
     "ctrl0-u1-40-long-additive-v2-l10-qwen3-4b-train", 0.10, "-"),
    ("v2 (flat-1) λ=0.15",
     KEY_SINGH, "singhh5050-stanford-university/interoception-l15",
     "ctrl0-qwen3-4b-u1-40-long-additive-v2-l15",
     "ctrl0-u1-40-long-additive-v2-l15-qwen3-4b-train", 0.15, "-"),
    ("v2 (flat-1) λ=0.30",
     KEY_SINGH, "singhh5050-stanford-university/interoception-l30",
     "ctrl0-qwen3-4b-u1-40-long-additive-v2-l30",
     "ctrl0-u1-40-long-additive-v2-l30-qwen3-4b-train", 0.30, "-"),
    ("windowed λ=0.15",
     KEY_NICO, "nicolema-stanford-university/interoception-windowed-l15",
     "ctrl0-qwen3-4b-u1-40-windowed-l15",
     "ctrl0-u1-40-windowed-l15-qwen3-4b-train", 0.15, ":"),
    ("windowed λ=0.30",
     KEY_NICO, "nicolema-stanford-university/interoception-windowed-l30",
     "ctrl0-qwen3-4b-u1-40-windowed-l30",
     "ctrl0-u1-40-windowed-l30-qwen3-4b-train", 0.30, ":"),
    ("windowed λ=0.50",
     KEY_NICO, "nicolema-stanford-university/interoception-windowed-l50",
     "ctrl0-qwen3-4b-u1-40-windowed-l50",
     "ctrl0-u1-40-windowed-l50-qwen3-4b-train", 0.50, ":"),
]
WIN = 10


def fetch_cell(api_key, project_path, run_name, env_name):
    api = wandb.Api(api_key=api_key)
    runs = sorted(api.runs(project_path, filters={"display_name": run_name}),
                  key=lambda r: r.created_at)
    if not runs:
        return None
    run = runs[-1]
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
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


print("Fetching wandb metrics for 6 cells...")
data = []  # list of (label, env_name, lambda_val, linestyle, color, series)
for label, api_key, project_path, run_name, env_name, lam, ls in CELLS:
    print(f"--- {label} ---")
    s = fetch_cell(api_key, project_path, run_name, env_name)
    color = LAMBDA_COLORS[lam]
    data.append((label, env_name, lam, ls, color, s))

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
panels = [
    ("correctness c",        "is_correct", 0),
    ("f(t, T)",              "f_term",     1),
    ("reward c + λ·f",       "mean",       2),
]
for panel_label, metric_kind, ax_idx in panels:
    ax = axes[ax_idx]
    for label, env_name, lam, ls, color, series in data:
        if series is None:
            continue
        key = next((k for k in series if k.endswith(f"/{metric_kind}")
                    or (metric_kind == "mean" and k.endswith("/mean"))), None)
        if key is None or not series[key]:
            continue
        pairs = series[key]
        ys = np.array([p[1] for p in pairs])
        x = np.arange(len(ys))
        # Skip the raw faint line — 6 cells gets too cluttered.
        if len(ys) >= WIN:
            sm = runavg(ys, WIN)
            sm_x = x[WIN - 1:]
            ax.plot(sm_x, sm, color=color, lw=2.4, ls=ls,
                    label=f"{label} (final≈{sm[-1]:.2f}, n={len(ys)})")
        else:
            ax.plot(x, ys, color=color, lw=2.4, ls=ls, label=f"{label} (n={len(ys)})")
    ax.set_ylabel(panel_label, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncols=2)

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen3-4B  —  flat-1 (solid) vs windowed (dotted) reward, color = λ",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/48_windowed_vs_flat_overlay.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
