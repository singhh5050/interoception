"""Pacing + Capability panes with a full per-variant legend (slide version).

Same data source and pane layout as plot_summary_multipane.py, but every
paced-RL cell gets its own color and a formula-first legend name, so the
figure is self-contained on a slide.

Data: analysis/eval_rollouts/prompt_salience_budget/<cell>_T<budget>_remaining_budget.jsonl
(held-out TEST set, same problems probed at each fixed budget, n=100/budget).
"""
from __future__ import annotations
import json, re, pathlib, statistics as st
import matplotlib.pyplot as plt

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience_budget")
OUT = pathlib.Path("analysis/figures/62_summary_legend.png")
PAT = re.compile(r"^(?P<cell>.+?)_T(?P<T>\d+)_remaining_budget\.jsonl$")

# cell -> (legend label, color, linestyle, marker)
STYLES = {
    "base": ("Base model (no RL)", "#9aa0a6", "--", None),
    "correctness-only": ("Accuracy-only · R = c", "#c0392b", "-", "s"),
    "v2-flat-l30": ("Flat-top additive · R = c + 0.30·min(1, T/t)", "#27a39a", "-", "o"),
    "windowed-l15": ("Gaussian window · λ=0.15, σᵤ=0.25", "#9ecae1", "-", "o"),
    "windowed-l30": ("Gaussian window · λ=0.30, σᵤ=0.25 (100 steps)", "#5b9bd5", "-", "o"),
    "su25": ("Gaussian window · λ=0.30, σᵤ=0.25 (200 steps)", "#2e74b5", "-", "o"),
    "su10": ("Gaussian window · λ=0.30, σᵤ=0.10 (200 steps)", "#6a51a3", "-", "o"),
}
ORDER = ["base", "correctness-only", "v2-flat-l30",
         "windowed-l15", "windowed-l30", "su25", "su10"]


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
        if not m or m["cell"] not in STYLES:
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
    return {k: v for k, v in cells.items() if len(v["T"]) >= 6}


def main():
    cells = load()
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 160,
    })
    # 12.8 x 5.25 in ~ the 2.45:1 body area of a 16:9 slide under its title;
    # panes on the left ~3/4, legend in the right-hand column
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 5.25))

    def curves(ax, key):
        for cell in ORDER:
            if cell not in cells:
                continue
            label, color, ls, marker = STYLES[cell]
            bold = cell in ("base", "correctness-only")
            ax.plot(cells[cell]["T"], cells[cell][key], color=color,
                    ls=ls, marker=marker, ms=4.5 if marker else 0,
                    lw=2.4 if bold else 1.8, label=label,
                    zorder=4 if bold else 2)

    axA.plot([0, 42], [0, 42], ls=":", lw=1.1, color="#555", alpha=0.7, zorder=0)
    axA.text(40.5, 38.5, "perfect\npacing", fontsize=8.5, color="#555",
             ha="right", va="top", style="italic")
    curves(axA, "commit")
    axA.set_xlabel("Budget T (s)"); axA.set_ylabel("Mean commit time (s)")
    axA.set_title("A  Pacing"); axA.set_xlim(0, 42); axA.set_ylim(0, 55)

    curves(axB, "acc")
    axB.set_xlabel("Budget T (s)"); axB.set_ylabel("Accuracy")
    axB.set_title("B  Capability"); axB.set_xlim(0, 42); axB.set_ylim(0, 0.7)

    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncol=1, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.745, 0.5),
               handletextpad=0.6, labelspacing=0.9)

    fig.tight_layout(rect=[0, 0, 0.75, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}\n")
    print(f"{'cell':22s} {'overall_acc':>11s} {'r':>7s}")
    for cell in ORDER:
        if cell in cells:
            print(f"{cell:22s} {cells[cell]['overall_acc']:>11.3f} {cells[cell]['r']:>7.2f}")


if __name__ == "__main__":
    main()
