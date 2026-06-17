"""Plot acc vs budget T from the budget-controlled sweep on test.jsonl.

Each (cell, T) cell has its own probe JSONL where every problem ran at that
fixed budget T. We group by budget and compute mean accuracy across the same
~100 problems at each T level.

Filename pattern: <cell>_T<XX>_<variant>.jsonl
"""
import json, pathlib, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience_budget")
PATTERN = re.compile(r"^(?P<cell>.+?)_T(?P<T>\d+)_(?P<variant>.+)\.jsonl$")

# Color scheme matching figure 51 conventions
CELL_STYLES = {
    "base":         ("base (no RL)",                 "#27ae60", "-"),
    "v1":           ("v1 λ=0.5",                     "#bcbd22", "-"),
    "v2-flat-l10":  ("v2-flat λ=0.10",               "#1f77b4", "-"),
    "v2-flat-l15":  ("v2-flat λ=0.15",               "#2ca02c", "-"),
    "v2-flat-l30":  ("v2-flat λ=0.30",               "#d62728", "-"),
    "windowed-l15": ("windowed λ=0.15",              "#1976D2", ":"),
    "windowed-l30": ("windowed λ=0.30",              "#2E7D32", ":"),
    "windowed-l50": ("windowed λ=0.50",              "#8E24AA", ":"),
}


def correct(r):
    if "is_correct" in r:
        return r["is_correct"]
    return 1 if r.get("reward", 0) > 0 else 0


# Index JSONLs by (cell, T)
files_by_cell = {}
for f in sorted(DATA.glob("*.jsonl")):
    m = PATTERN.match(f.name)
    if not m: continue
    cell = m.group("cell")
    T = int(m.group("T"))
    files_by_cell.setdefault(cell, {})[T] = f

# Compute acc per (cell, T)
print(f"  {'cell':<18} | " + " | ".join(f"T={T:>2}" for T in [2,5,10,20,30,40]))
print("  " + "-" * 70)
results = {}
for cell, by_T in sorted(files_by_cell.items()):
    cell_results = {}
    row = []
    for T in [2, 5, 10, 20, 30, 40]:
        if T not in by_T:
            row.append(" -- ")
            continue
        recs = [json.loads(l) for l in by_T[T].open()]
        acc = sum(correct(r) for r in recs) / len(recs)
        cell_results[T] = (acc, len(recs))
        row.append(f"{acc:.2f}")
    results[cell] = cell_results
    print(f"  {cell:<18} | " + " | ".join(f"{c:>4}" for c in row))

# Plot
fig, ax = plt.subplots(figsize=(10, 6.5))
for cell, by_T in results.items():
    if cell not in CELL_STYLES:
        print(f"  skipping unknown cell {cell}")
        continue
    label, color, ls = CELL_STYLES[cell]
    Ts = sorted(by_T)
    accs = [by_T[T][0] for T in Ts]
    ns = [by_T[T][1] for T in Ts]
    n_total = sum(ns)
    ax.plot(Ts, accs, marker="o", ms=10, color=color, lw=2.6, ls=ls,
            label=f"{label}  (n={ns[0] if ns else 0}/budget, {len(Ts)} budgets)")

ax.set_xlabel("Budget T (s)  — fixed per-rollout", fontsize=12)
ax.set_ylabel("Accuracy on test set", fontsize=12)
ax.set_xlim(0, 42); ax.set_ylim(0, 0.6)
ax.set_title("Accuracy vs budget T — controlled (same problems at each T)\n"
             "test set, matched (remaining_budget) prompt",
             fontsize=11)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(alpha=0.25)

out = pathlib.Path("analysis/figures/52_acc_vs_budget_controlled.png")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {out}")
