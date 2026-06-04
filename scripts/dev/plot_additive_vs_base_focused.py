"""Focused 2-line version of figure 38: only the two remaining_budget cells —
base model and long-additive RL'd model. Shows that additive-reward RL
sharpens the pacing curve relative to the base model (slope closer to ideal t=T)."""
import json, pathlib, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe_prompt_salience import parse_completion, BINS

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")
CELLS = [
    ("base / remaining_budget",         "base_remaining_budget.jsonl",         "#27ae60"),
    ("long-additive / remaining_budget","long-additive_remaining_budget.jsonl","#E63946"),
]


def load(fname):
    recs = []
    for line in (DATA / fname).open():
        r = json.loads(line)
        r.update(parse_completion(r["completion"]))
        recs.append(r)
    return recs


fig, ax = plt.subplots(figsize=(8.5, 6.0))
for label, fname, color in CELLS:
    recs = load(fname)
    committers = [r for r in recs if r["has_answer"] and r["elapsed_at_commit"] is not None]
    bx, by = [], []
    for lo, hi in BINS:
        b = [r for r in committers if lo <= r["target_s"] < hi]
        if not b: continue
        bx.append(st.mean(r["target_s"] for r in b))
        by.append(st.mean(r["elapsed_at_commit"] for r in b))
    ax.plot(bx, by, marker="o", ms=12, color=color, lw=3.0,
            label=f"{label}  (n_commit={len(committers)})")

ax.plot([0, 40], [0, 40], color="#888", lw=1.5, ls=":", label="t = T (commit exactly on budget)")
ax.set_xlabel("Budget T (s)", fontsize=12)
ax.set_ylabel("Elapsed at commit (s)", fontsize=12)
ax.set_xlim(0, 40); ax.set_ylim(0, 40)
ax.set_title("Commit time vs budget T — base vs long-additive", fontsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10.5)
ax.grid(alpha=0.25)
ax.set_aspect("equal")

out = pathlib.Path("analysis/figures/42_additive_vs_base_commit_time.png")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
