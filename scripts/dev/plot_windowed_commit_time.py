"""Commit-time-vs-T graph for all matched-prompt cells: base model + 3 v2
flat-1 cells + 3 windowed cells. Color = λ value (gray = base). Solid = flat-1
(v2), dotted = windowed.

This is the "calibration" picture: does the model commit at T (on the t=T
diagonal), or just scale with T but undershoot?
"""
import json, pathlib, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe_prompt_salience import parse_completion, BINS

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")

# Base + v2-l30 (flat-1) as references; all 3 windowed matched cells.
# Solid = base/flat-1, dotted = windowed.
CELLS = [
    # (label, fname, color, linestyle)
    ("base model (no RL)",   "base_remaining_budget.jsonl",                  "#27ae60", "-"),
    ("v2 (flat-1) λ=0.30",   "long-additive-v2-l30_remaining_budget.jsonl",  "#d62728", "-"),
    ("windowed λ=0.15",      "windowed-l15_remaining_budget.jsonl",          "#1976D2", ":"),
    ("windowed λ=0.30",      "windowed-l30_remaining_budget.jsonl",          "#2E7D32", ":"),
    ("windowed λ=0.50",      "windowed-l50_remaining_budget.jsonl",          "#8E24AA", ":"),
]


def load(fname):
    recs = []
    for line in (DATA / fname).open():
        r = json.loads(line)
        r.update(parse_completion(r["completion"]))
        recs.append(r)
    return recs


fig, ax = plt.subplots(figsize=(10.5, 7.5))
for label, fname, color, ls in CELLS:
    recs = load(fname)
    committers = [r for r in recs if r["has_answer"] and r["elapsed_at_commit"] is not None]
    bx, by = [], []
    for lo, hi in BINS:
        b = [r for r in committers if lo <= r["target_s"] < hi]
        if not b: continue
        bx.append(st.mean(r["target_s"] for r in b))
        by.append(st.mean(r["elapsed_at_commit"] for r in b))
    ax.plot(bx, by, marker="o", ms=10, color=color, lw=2.6, ls=ls,
            label=f"{label}  (n={len(committers)})")

ax.plot([0, 40], [0, 40], color="#888", lw=1.5, ls=":", label="t = T (commit exactly on budget)")
ax.set_xlabel("Budget T (s)", fontsize=12)
ax.set_ylabel("Elapsed at commit (s)", fontsize=12)
ax.set_xlim(0, 40); ax.set_ylim(0, 40)
ax.set_title("Commit time vs budget T (matched prompt) — flat-1 (solid) vs windowed (dotted)",
             fontsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(alpha=0.25)

out = pathlib.Path("analysis/figures/49_windowed_commit_time_vs_T.png")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}")
