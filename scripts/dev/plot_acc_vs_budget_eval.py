"""Pareto plot: accuracy vs budget T on the eval set, matched prompt.

x = T budget (s), y = mean accuracy per T-bin.
One line per condition: base (no RL), v2-l30 flat-1, 3 windowed cells.
Solid = base/flat-1, dotted = windowed (mirrors figure 49 convention).
"""
import json, pathlib, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe_prompt_salience import BINS

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")

CELLS = [
    # (label, fname, color, linestyle)
    # NOTE: base model probes re-running with fixed scoring; will add line when ready.
    ("v2 (flat-1) λ=0.30",   "long-additive-v2-l30_remaining_budget.jsonl",  "#d62728", "-"),
    ("v2 (flat-1) λ=0.15",   "long-additive-v2-l15_remaining_budget.jsonl",  "#2ca02c", "-"),
    ("windowed λ=0.15",      "windowed-l15_remaining_budget.jsonl",          "#1976D2", ":"),
    ("windowed λ=0.30",      "windowed-l30_remaining_budget.jsonl",          "#2E7D32", ":"),
    ("windowed λ=0.50",      "windowed-l50_remaining_budget.jsonl",          "#8E24AA", ":"),
]


def load_acc_by_bin(fname):
    recs = [json.loads(l) for l in (DATA / fname).open()]
    # Use is_correct if present (newer probes), else reward > 0 (legacy proxy).
    def correct(r):
        if "is_correct" in r:
            return r["is_correct"]
        return 1 if r.get("reward", 0) > 0 else 0
    bin_data = []
    for lo, hi in BINS:
        b = [r for r in recs if lo <= r["target_s"] < hi]
        if not b: continue
        T_mid = st.mean(r["target_s"] for r in b)
        acc = st.mean(correct(r) for r in b)
        bin_data.append((T_mid, acc, len(b)))
    overall = st.mean(correct(r) for r in recs)
    return bin_data, overall, len(recs)


fig, ax = plt.subplots(figsize=(10, 6.5))
for label, fname, color, ls in CELLS:
    if not (DATA / fname).exists():
        print(f"  skipping (no file): {fname}")
        continue
    bin_data, overall_acc, n_total = load_acc_by_bin(fname)
    if not bin_data: continue
    xs = [t for t, _, _ in bin_data]
    ys = [a for _, a, _ in bin_data]
    ax.plot(xs, ys, marker="o", ms=10, color=color, lw=2.6, ls=ls,
            label=f"{label}  (overall acc={overall_acc:.3f}, n={n_total})")

ax.set_xlabel("Budget T (s)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xlim(0, 40); ax.set_ylim(0, 0.65)
ax.set_title("Accuracy vs budget T  —  eval set, matched prompt",
             fontsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(alpha=0.25)

out = pathlib.Path("analysis/figures/50_acc_vs_budget_eval.png")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}")
