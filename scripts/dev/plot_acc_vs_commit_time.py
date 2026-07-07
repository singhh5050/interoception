"""Clarifying plot: accuracy vs. ACTUAL commit time (not the stated budget).

Answers the advisors' question — "accuracy-only gets ~0.6 but budget-aware gets
0.1-0.3, isn't that a problem?" The catch: accuracy-only ignores the budget and
commits ~47s on every problem, so its 0.54 is plotted (misleadingly) as a flat line
across the budget axis. Put accuracy on the real TIME axis and every model lands on
one rising accuracy-vs-time frontier; accuracy-only is just the far-right (47s) end.
The "gap" is a time-cost gap, not a capability gap.
"""
from __future__ import annotations
import json, glob, re, os, pathlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = "analysis/eval_rollouts/prompt_salience_budget"
OUT = pathlib.Path("analysis/figures/61_acc_vs_commit_time.png")
PAT = re.compile(r"^(.+)_T(\d+)_remaining_budget\.jsonl$")

PACED = ["v2-flat-l30", "windowed-l15", "windowed-l30", "su10", "su25", "su17"]
BLUE, RED, GRAY = "#5b9bd5", "#c0392b", "#9aa0a6"


def load():
    cells = {}
    for f in sorted(glob.glob(f"{D}/*.jsonl")):
        m = PAT.match(os.path.basename(f))
        cell, T = m.group(1), int(m.group(2))
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if len(rows) < 50:
            continue
        c = cells.setdefault(cell, [])
        acc = sum(1 for r in rows if r.get("is_correct")) / len(rows)
        ct = sum(r["elapsed_s"] for r in rows) / len(rows)
        c.append((ct, acc))
    for c in cells.values():
        c.sort()
    return cells


def main():
    cells = load()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 160})
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # paced-RL frontier (point cloud — every (commit_time, acc) across all variants/budgets)
    for cell in PACED:
        if cell in cells:
            xs, ys = zip(*cells[cell])
            ax.scatter(xs, ys, color=BLUE, s=42, alpha=0.7, zorder=2,
                       edgecolor="white", linewidth=0.5)
    # base
    if "base" in cells:
        xs, ys = zip(*cells["base"])
        ax.plot(xs, ys, color=GRAY, lw=2.2, ls="--", marker="o", ms=4, zorder=3)
    # accuracy-only: cluster at ~47s
    if "correctness-only" in cells:
        xs, ys = zip(*cells["correctness-only"])
        ax.scatter(xs, ys, color=RED, marker="s", s=80, zorder=4,
                   edgecolor="white", linewidth=0.6)

    ax.axvspan(44, 50, color=RED, alpha=0.05, zorder=0)
    ax.annotate("Accuracy-only ignores the budget\n— always commits ~47s, acc ~0.54",
                xy=(47, 0.54), xytext=(30, 0.62), fontsize=9.2, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(19, 0.035, "Budget-aware models commit on time (5–35s)\n"
            "and trace the same accuracy-vs-time frontier",
            fontsize=9.2, color="#2e6da4", ha="center")

    ax.set_xlabel("Actual commit time (s)")
    ax.set_ylabel("Accuracy (test set)")
    ax.set_xlim(0, 52); ax.set_ylim(0, 0.7)
    handles = [
        Line2D([0], [0], color=GRAY, lw=2.2, ls="--", marker="o", label="Base (no RL)"),
        Line2D([0], [0], color=RED, lw=0, marker="s", ms=8, label="Accuracy-only (R=c)"),
        Line2D([0], [0], color=BLUE, lw=1.6, marker="o", label="Budget-aware RL (6 variants)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
