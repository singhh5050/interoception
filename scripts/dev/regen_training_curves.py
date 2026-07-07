"""Regenerate the three training-curve figures from wandb (clean styling):
  - conly composite (train reward + MA + eval acc)   -> report/figures/gen_conly_curve.png
  - windowed sigma_under sweep (c / f / reward)        -> report/figures/gen_sigma_curves.png
  - stage-2 KL beta sweep (c / f / reward)             -> report/figures/gen_kl_curves.png

Fixes over the originals: shared legend BELOW each figure (never over the data),
lighter raw-noise alpha, heavier moving average, larger fonts.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT = "singhh5050-stanford-university"
WIN = 15
api = wandb.Api(timeout=120)


def fetch(project, run_name, keys):
    runs = sorted(api.runs(f"{ENT}/{project}", filters={"display_name": run_name}),
                  key=lambda r: r.created_at)
    if not runs:
        print(f"  NO RUN {project}/{run_name}"); return None
    run = runs[-1]
    out = {k: [] for k in keys}
    for row in run.scan_history(keys=["_step"] + keys, page_size=10000):
        s = row.get("_step")
        if s is None:
            continue
        for k in keys:
            if row.get(k) is not None:
                out[k].append((int(s), float(row[k])))
    return {k: sorted(v) for k, v in out.items()}


def ma(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


# ---------- 1. windowed sigma_under sweep (3-panel) ----------
def panel_sweep(cells, env_of, title, out, anchors=None):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    panels = [("correctness  $c$", "is_correct"),
              ("time-fit  $f(t,T)$", "f_term"),
              ("reward", "mean")]
    handles = None
    for label, project, run_name, color in cells:
        env = env_of(run_name, project)
        keys = [f"metrics/{env}/is_correct", f"metrics/{env}/f_term", f"reward/{env}/mean"]
        s = fetch(project, run_name, keys)
        if s is None:
            continue
        for ax, (plabel, kind) in zip(axes, panels):
            key = next(k for k in keys if k.endswith("/" + kind))
            ys = np.array([p[1] for p in s[key]]); x = np.arange(len(ys))
            ax.plot(x, ys, color=color, alpha=0.08, lw=0.9)
            sm = ma(ys, WIN); sx = x[WIN - 1:] if len(ys) >= WIN else x
            ax.plot(sx, sm, color=color, lw=2.6, label=label)
    for ax, (plabel, _) in zip(axes, panels):
        ax.set_ylabel(plabel, fontsize=12); ax.grid(alpha=0.25); ax.tick_params(labelsize=10)
    if anchors:
        for idx, (lab, val) in anchors.items():
            axes[idx].axhline(val, color="#555", lw=1.4, ls="--")
            axes[idx].text(0.99, val, " " + lab, color="#555", fontsize=9,
                           va="bottom", ha="right", transform=axes[idx].get_yaxis_transform())
    axes[-1].set_xlabel("training step", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out, dpi=160); print("wrote", out)


SIGMA = [
    ("$\\sigma_{under}$=0.25 (gentle)",    "interoception-windowed-su25-200", "ctrl0-qwen3-4b-u1-40-windowed-su25-200", "#1f77b4"),
    ("$\\sigma_{under}$=0.17 (mid)",       "interoception-windowed-su17-200", "ctrl0-qwen3-4b-u1-40-windowed-su17-200", "#E8730C"),
    ("$\\sigma_{under}$=0.10 (symmetric)", "interoception-windowed-su10-200", "ctrl0-qwen3-4b-u1-40-windowed-su10-200", "#9467bd"),
]
panel_sweep(SIGMA,
            env_of=lambda rn, pr: rn.replace("ctrl0-qwen3-4b-", "ctrl0-") + "-train",
            title="Windowed $\\sigma_{under}$ sweep — training curves (200 steps, $\\lambda$=0.30)",
            out="report/figures/gen_sigma_curves.png")

KL = [
    ("$\\beta$=0 (no anchor)", "interoception-stage2-b0", "stage2-kl-b0", "#1f77b4"),
    ("$\\beta$=1e-4",          "interoception-stage2-b4", "stage2-kl-b4", "#2ca02c"),
    ("$\\beta$=1e-3",          "interoception-stage2-b3", "stage2-kl-b3", "#d62728"),
    ("$\\beta$=1e-2",          "interoception-stage2-b2", "stage2-kl-b2", "#9467bd"),
    ("$\\beta$=1e-1 (heavy)",  "interoception-stage2-b1", "stage2-kl-b1", "#8c564b"),
]
panel_sweep(KL,
            env_of=lambda rn, pr: rn + "-train",
            title="Two-stage KL curriculum — Stage-2 training curves (c-only, anchor = additive $\\lambda$=0.30)",
            out="report/figures/gen_kl_curves.png",
            anchors={0: ("Stage-1 anchor c=0.17", 0.17), 1: ("Stage-1 anchor f=0.86", 0.86)})

# ---------- conly composite ----------
env = "ctrl0-u1-40-strict-conly-qwen3-4b-train"
ev = "eval/ctrl0-u1-40-strict-conly-eval-uniform/avg@1"
# fetch separately: per-step reward and sparse eval are logged on different steps,
# so a single scan_history(keys=[...]) would return only their (empty) intersection.
rw = fetch("interoception", "ctrl0-qwen3-4b-u1-40-strict-conly", [f"reward/{env}/mean"])[f"reward/{env}/mean"]
acc = fetch("interoception", "ctrl0-qwen3-4b-u1-40-strict-conly", [ev])[ev]
# wandb _step is a fine logging counter; rescale to optimizer steps. This is the
# resumed leg (global steps 100->500), so map raw _step linearly onto [100, 500].
maxstep = max(p[0] for p in rw)
to_opt = lambda st: 100 + (st / maxstep) * 400
rsteps = np.array([to_opt(p[0]) for p in rw]); ry = np.array([p[1] for p in rw])
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(rsteps, ry, color="#bbbbbb", lw=0.8, alpha=0.7, label="train reward (per step)")
sm = ma(ry, WIN); ax.plot(rsteps[WIN - 1:], sm, color="#1f77b4", lw=2.4, label="train reward (15-step MA)")
if acc:
    ax.plot([to_opt(p[0]) for p in acc], [p[1] for p in acc], "o-", color="#2E8B2E", lw=2.2, ms=6,
            label="in-training eval accuracy (avg@1)")
ax.set_xlim(100, 500)
ax.set_xlabel("training step", fontsize=12); ax.set_ylabel("reward / accuracy", fontsize=12)
ax.set_title("Correctness-only RL ($R=c$): reward & eval accuracy over training", fontsize=12)
ax.grid(alpha=0.25); ax.tick_params(labelsize=10)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, fontsize=10, framealpha=0.95)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("report/figures/gen_conly_curve.png", dpi=160); print("wrote report/figures/gen_conly_curve.png")
