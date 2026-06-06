"""Focused commit-time-vs-T figure for the v2 sweep — 4 lines:
base+remaining_budget (reference) plus the 3 sweep matched-prompt cells
(λ ∈ {0.10, 0.15, 0.30}). Mirror of plot_additive_vs_base_focused.py but
showing the full λ sweep instead of just one cell."""
import json, pathlib, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe_prompt_salience import parse_completion, BINS

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")
CELLS = [
    ("base + remaining_budget",          "base_remaining_budget.jsonl",                  "#27ae60"),
    ("v2-λ=0.10 + remaining_budget",     "long-additive-v2-l10_remaining_budget.jsonl",  "#1976D2"),
    ("v2-λ=0.15 + remaining_budget",     "long-additive-v2-l15_remaining_budget.jsonl",  "#2E7D32"),
    ("v2-λ=0.30 + remaining_budget",     "long-additive-v2-l30_remaining_budget.jsonl",  "#BF360C"),
]


def load(fname):
    recs = []
    for line in (DATA / fname).open():
        r = json.loads(line)
        r.update(parse_completion(r["completion"]))
        recs.append(r)
    return recs


fig, ax = plt.subplots(figsize=(9.5, 7.0))
for label, fname, color in CELLS:
    recs = load(fname)
    committers = [r for r in recs if r["has_answer"] and r["elapsed_at_commit"] is not None]
    bx, by = [], []
    for lo, hi in BINS:
        b = [r for r in committers if lo <= r["target_s"] < hi]
        if not b: continue
        bx.append(st.mean(r["target_s"] for r in b))
        by.append(st.mean(r["elapsed_at_commit"] for r in b))
    ax.plot(bx, by, marker="o", ms=12, color=color, lw=2.8,
            label=f"{label}  (n_commit={len(committers)})")

ax.plot([0, 40], [0, 40], color="#888", lw=1.5, ls=":", label="t = T (commit exactly on budget)")
ax.set_xlabel("Budget T (s)", fontsize=12)
ax.set_ylabel("Elapsed at commit (s)", fontsize=12)
ax.set_xlim(0, 40); ax.set_ylim(0, 45)
ax.set_title("Commit time vs budget T — v2 λ sweep (matched prompt) vs base", fontsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(alpha=0.25)

out = pathlib.Path("analysis/figures/44_v2_sweep_commit_time_vs_T.png")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}")
