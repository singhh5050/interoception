"""Comprehensive multi-pane summary: pacing vs. accuracy across the reward landscape.

Single consistent source: the budget-controlled sweep on the held-out TEST set
(analysis/eval_rollouts/prompt_salience_budget/<cell>_T<budget>_remaining_budget.jsonl,
same problems probed at each fixed budget, n=100/budget). Merges Harsh's cells
(controls + σ-sweep + specialists) with Nicole's (additive + windowed-λ) — all on the
same dataset/pipeline, so every pane is mutually consistent.

  Pane A  Pacing      — mean commit time vs. budget (diagonal = perfect pacing)
  Pane B  Capability  — accuracy vs. budget
  Pane C  Trade-off   — accuracy vs. calibration r, the two basins across all reward shapes
"""
from __future__ import annotations
import json, re, pathlib, statistics as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience_budget")
OUT = pathlib.Path("analysis/figures/60_summary_multipane.png")
PAT = re.compile(r"^(?P<cell>.+?)_T(?P<T>\d+)_remaining_budget\.jsonl$")

PACED_BLUE = "#5b9bd5"          # paced-RL family (additive + windowed variants)
PACED_DOT = "#2e74b5"
GRAY = "#9aa0a6"
REDC = "#c0392b"
G10, G30 = "#27a39a", "#176f63"

# range-trained paced-RL variants (different reward shapes, all collapse to pacing)
PACED = ["v2-flat-l30", "windowed-l15", "windowed-l30", "su10", "su25"]


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx * vy else 0.0


def load():
    cells = {}
    for f in sorted(DATA.glob("*.jsonl")):
        m = PAT.match(f.name)
        if not m:
            continue
        cell, T = m["cell"], int(m["T"])
        rows = [json.loads(l) for l in f.open() if l.strip()]
        if not rows:
            continue
        c = cells.setdefault(cell, {"T": [], "acc": [], "commit": [],
                                    "all_T": [], "all_el": []})
        n = len(rows)
        c["T"].append(T)
        c["acc"].append(sum(1 for r in rows if r.get("is_correct")) / n)
        c["commit"].append(sum(r.get("elapsed_s", 0.0) for r in rows) / n)
        for r in rows:
            c["all_T"].append(float(r.get("target_s", T)))
            c["all_el"].append(float(r.get("elapsed_s", 0.0)))
    for c in cells.values():
        order = sorted(range(len(c["T"])), key=lambda i: c["T"][i])
        for k in ("T", "acc", "commit"):
            c[k] = [c[k][i] for i in order]
        c["overall_acc"] = st.mean(c["acc"])
        c["r"] = pearson(c["all_T"], c["all_el"])
    return cells


def main():
    cells = load()
    # only fully-swept cells (6 budgets) qualify for the figure
    full = {k: v for k, v in cells.items() if len(v["T"]) >= 6}
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 160,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 5.0))

    def curves(ax, key):
        # paced-RL band (thin, one color)
        for cell in PACED:
            if cell in full:
                ax.plot(full[cell]["T"], full[cell][key], color=PACED_BLUE,
                        lw=1.4, alpha=0.6, zorder=1)
        # bold anchors
        if "base" in full:
            ax.plot(full["base"]["T"], full["base"][key], color=GRAY, lw=2.4,
                    ls="--", zorder=3)
        if "correctness-only" in full:
            ax.plot(full["correctness-only"]["T"], full["correctness-only"][key],
                    color=REDC, lw=2.6, marker="s", ms=5, zorder=4)

    # ---- Pane A: pacing ----
    axA.plot([0, 42], [0, 42], ls=":", lw=1.1, color="#555", alpha=0.7, zorder=0)
    axA.text(40.5, 38.5, "perfect\npacing", fontsize=8.5, color="#555",
             ha="right", va="top", style="italic")
    curves(axA, "commit")
    axA.set_xlabel("Budget T (s)"); axA.set_ylabel("Mean commit time (s)")
    axA.set_title("A  Pacing"); axA.set_xlim(0, 42); axA.set_ylim(0, 55)

    # ---- Pane B: capability ----
    curves(axB, "acc")
    axB.set_xlabel("Budget T (s)"); axB.set_ylabel("Accuracy")
    axB.set_title("B  Capability"); axB.set_xlim(0, 42); axB.set_ylim(0, 0.7)

    # ---- shared legend below ----
    handles = [
        Line2D([0], [0], color=GRAY, lw=2.4, ls="--", label="Base (no RL)"),
        Line2D([0], [0], color=REDC, lw=2.6, marker="s", ms=6, label="Accuracy-only (R=c)"),
        Line2D([0], [0], color=PACED_BLUE, lw=2.0, marker="o", ms=6,
               label="Paced RL — additive & windowed variants (5)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04), columnspacing=2.0,
               handletextpad=0.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}\n")
    print(f"{'cell':22s} {'overall_acc':>11s} {'r':>7s}")
    for cell in sorted(full):
        print(f"{cell:22s} {full[cell]['overall_acc']:>11.3f} {full[cell]['r']:>7.2f}")


if __name__ == "__main__":
    main()
