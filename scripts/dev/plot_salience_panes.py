"""Lottery-exam (prompt_salience) version of the pacing/capability panes.

Same pane layout as the budget-controlled figure (62_summary_legend.png), but
built from the uniform-T eval, which covers 16 RL cells instead of 7. T~U(1,40)
is binned into 6 ranges (~83 rollouts/cell/bin), so curves are noisier than the
controlled sweep but the coverage is much wider.

  Pane A  Pacing      — mean commit time vs. binned budget (diagonal = perfect)
  Pane B  Capability  — accuracy vs. binned budget
  Pane C  Michael's correction — accuracy vs. ACTUAL mean time used (one point
          per model). Pane B's x-axis is misleading for non-pacing models (they
          spend the same real time no matter the stated budget); on the real
          time axis every model lands on one rising cost-vs-accuracy frontier.

Data: analysis/eval_rollouts/prompt_salience/prompt_salience/<cell>_remaining_budget.jsonl
(held-out set, T~U(1,40) seed-matched across cells, temp 1.0, n=498/cell).
long-500 was logged without is_correct → re-scored with the env's validate_solution.
"""
from __future__ import annotations
import json, re, pathlib, statistics as st, sys
import matplotlib.pyplot as plt

sys.path.insert(0, "environments/interoception_countdown")
from _solver import validate_solution  # noqa: E402

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")
OUT = pathlib.Path("analysis/figures/63_salience_pacing_capability.png")
BIN_EDGES = [1, 7, 14, 20, 27, 34, 40]

NUMS_RE = re.compile(r"Using the numbers \[([\d,\s]+)\].*?equals (\d+)", re.S)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

# (label, file stem, color) — same names/colors as the Pareto scatter (fig 51)
CELLS = [
    ("Base model (no RL)",                 "base",                 "#9aa0a6"),
    ("Accuracy-only R=c",                  "strict-conly",         "#c0392b"),
    ("Multiplicative c·f (quiet prompt)",  "long-500",             "#e67e22"),
    ("Multiplicative c·f (strict prompt)", "long-strict",          "#b9770e"),
    ("Flat-top additive λ=0.10",           "long-additive-v2-l10", "#a1d99b"),
    ("Flat-top additive λ=0.15",           "long-additive-v2-l15", "#74c476"),
    ("Flat-top additive λ=0.30",           "long-additive-v2-l30", "#31a354"),
    ("Flat-top additive λ=0.50",           "long-additive",        "#006d2c"),
    ("Gaussian window λ=0.15",             "windowed-l15",         "#9ecae1"),
    ("Gaussian window λ=0.30",             "windowed-l30",         "#5b9bd5"),
    ("Gaussian window λ=0.50",             "windowed-l50",         "#2e74b5"),
    ("KL curriculum β=0",                  "stage2-kl-b0",         "#dadaeb"),
    ("KL curriculum β=1e-4",               "stage2-kl-b4",         "#bcbddc"),
    ("KL curriculum β=1e-3",               "stage2-kl-b3",         "#9e9ac8"),
    ("KL curriculum β=1e-2",               "stage2-kl-b2",         "#756bb1"),
    ("KL curriculum β=1e-1",               "stage2-kl-b1",         "#54278f"),
]
ANCHORS = {"Base model (no RL)", "Accuracy-only R=c"}


def rescore(text):
    m = NUMS_RE.search(text)
    answers = ANS_RE.findall(text)
    if not m or not answers:
        return 0
    nums = [int(x) for x in m.group(1).split(",")]
    target = int(m.group(2))
    return 1 if validate_solution(answers[-1].strip(), nums, target) is True else 0


def load(stem):
    f = DATA / f"{stem}_remaining_budget.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(l) for l in f.open() if l.strip()]
    rows = [(r["target_s"], r["elapsed_s"],
             r["is_correct"] if "is_correct" in r else rescore(r["completion"]))
            for r in recs]
    binned = {"T": [], "acc": [], "commit": []}
    for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:]):
        sub = [r for r in rows if lo <= r[0] < hi or (hi == BIN_EDGES[-1] and r[0] == hi)]
        if not sub:
            continue
        binned["T"].append(st.mean(r[0] for r in sub))
        binned["commit"].append(st.mean(r[1] for r in sub))
        binned["acc"].append(st.mean(r[2] for r in sub))
    binned["overall_acc"] = st.mean(r[2] for r in rows)
    binned["mean_commit"] = st.mean(r[1] for r in rows)
    return binned


def main():
    cells = {label: (load(stem), color) for label, stem, color in CELLS}
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 160,
    })
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.6))

    def curves(ax, key):
        for label, (c, color) in cells.items():
            if c is None:
                continue
            bold = label in ANCHORS
            ax.plot(c["T"], c[key], color=color, marker="o", ms=3.5,
                    lw=2.4 if bold else 1.6, ls="--" if label == "Base model (no RL)" else "-",
                    label=label, zorder=4 if bold else 2,
                    alpha=1.0 if bold else 0.85)

    axA.plot([0, 42], [0, 42], ls=":", lw=1.1, color="#555", alpha=0.7, zorder=0)
    axA.text(40.5, 38.5, "perfect\npacing", fontsize=8.5, color="#555",
             ha="right", va="top", style="italic")
    curves(axA, "commit")
    axA.set_xlabel("Budget T (s, binned)"); axA.set_ylabel("Mean commit time (s)")
    axA.set_title("A  Pacing"); axA.set_xlim(0, 42); axA.set_ylim(0, 60)

    curves(axB, "acc")
    axB.set_xlabel("Budget T (s, binned)"); axB.set_ylabel("Accuracy")
    axB.set_title("B  Capability vs stated budget"); axB.set_xlim(0, 42); axB.set_ylim(0, 0.7)

    for label, (c, color) in cells.items():
        if c is None:
            continue
        bold = label in ANCHORS
        axC.scatter(c["mean_commit"], c["overall_acc"], color=color, s=90 if bold else 55,
                    edgecolors="black" if bold else "none", linewidths=1.2, zorder=3)
    axC.set_xlabel("Mean ACTUAL time used (s)"); axC.set_ylabel("Overall accuracy")
    axC.set_title("C  Capability vs actual time used"); axC.set_ylim(0, 0.7)

    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.12),
               columnspacing=1.2, handletextpad=0.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}\n")
    print(f"{'cell':38s} {'overall_acc':>11s} {'mean_commit':>12s}")
    for label, (c, _) in cells.items():
        if c:
            print(f"{label:38s} {c['overall_acc']:>11.3f} {c['mean_commit']:>12.1f}")


if __name__ == "__main__":
    main()
