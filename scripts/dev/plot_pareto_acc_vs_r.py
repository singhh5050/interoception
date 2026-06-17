"""True Pareto plot: accuracy vs Pearson r(commit_time, T) on the eval set.
Each point = one (condition, prompt) cell. Pareto frontier highlighted.

Two competing objectives:
  x = r(commit, T) — pacing quality / T-tracking
  y = accuracy

A model is Pareto-dominated if any other cell has BOTH higher acc AND higher r.
The frontier is the set of non-dominated cells.

Matched-prompt only for clarity (the primary metric across our experiments).
"""
import json, math, re, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")
ELAPSED_RE = re.compile(r"\[([\d.]+)s elapsed")


def parse(text):
    body_start = text.find("<|im_start|>assistant")
    body = text[body_start:] if body_start != -1 else text
    matches = list(ELAPSED_RE.finditer(body))
    ans_pos = body.find("<answer>")
    has_ans = ans_pos != -1
    eac = None
    if has_ans:
        prior = [float(m.group(1)) for m in matches if m.start() < ans_pos]
        eac = prior[-1] if prior else 0.0
    return has_ans, eac


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3: return float("nan")
    xs, ys = zip(*pairs); n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    return sxy / (sxx*syy)**0.5 if sxx*syy > 0 else float("nan")


# (label, fname, color)
CELLS = [
    ("base (no RL)",        "base_remaining_budget.jsonl",                       "#27ae60"),
    ("v1 λ=0.5",            "long-additive_remaining_budget.jsonl",              "#bcbd22"),
    ("v2-flat λ=0.10",      "long-additive-v2-l10_remaining_budget.jsonl",       "#1f77b4"),
    ("v2-flat λ=0.15",      "long-additive-v2-l15_remaining_budget.jsonl",       "#2ca02c"),
    ("v2-flat λ=0.30",      "long-additive-v2-l30_remaining_budget.jsonl",       "#d62728"),
    ("stage2 β=0",          "stage2-kl-b0_remaining_budget.jsonl",               "#ff9896"),
    ("stage2 β=1e-4",       "stage2-kl-b4_remaining_budget.jsonl",               "#c5b0d5"),
    ("stage2 β=1e-3",       "stage2-kl-b3_remaining_budget.jsonl",               "#c49c94"),
    ("stage2 β=1e-2",       "stage2-kl-b2_remaining_budget.jsonl",               "#f7b6d2"),
    ("stage2 β=1e-1",       "stage2-kl-b1_remaining_budget.jsonl",               "#dbdb8d"),
    ("windowed λ=0.15",     "windowed-l15_remaining_budget.jsonl",               "#17becf"),
    ("windowed λ=0.30",     "windowed-l30_remaining_budget.jsonl",               "#1976D2"),
    ("windowed λ=0.50",     "windowed-l50_remaining_budget.jsonl",               "#8E24AA"),
]


def compute(fname):
    if not (DATA / fname).exists():
        return None
    recs = [json.loads(l) for l in (DATA / fname).open()]
    def correct(r):
        if "is_correct" in r:
            return r["is_correct"]
        return 1 if r.get("reward", 0) > 0 else 0
    acc = sum(correct(r) for r in recs) / len(recs)
    committers = []
    for r in recs:
        has, eac = parse(r["completion"])
        if has and eac is not None:
            committers.append((r["target_s"], eac))
    if len(committers) < 3:
        return acc, float("nan"), len(committers), len(recs)
    r_val = pearson([T for T, _ in committers], [e for _, e in committers])
    return acc, r_val, len(committers), len(recs)


# Compute (acc, r) for each cell
points = []
print(f"  {'cell':<28}  {'acc':>6}  {'r':>7}  {'n_commit':>9}")
print("  " + "-" * 60)
for label, fname, color in CELLS:
    result = compute(fname)
    if result is None:
        print(f"  {label:<28}  (missing file)")
        continue
    acc, r_val, n_c, n_total = result
    note = ""
    points.append((label + note, color, acc, r_val))
    print(f"  {label:<28}  {acc:>6.3f}  {r_val:>+7.3f}  {n_c:>9}")


# Compute Pareto frontier (cells where no other cell has BOTH higher acc AND higher r)
def is_pareto(i, pts):
    a_i, r_i = pts[i][2], pts[i][3]
    for j, (_, _, a_j, r_j) in enumerate(pts):
        if j == i: continue
        if a_j >= a_i and r_j >= r_i and (a_j > a_i or r_j > r_i):
            return False
    return True

pareto_idx = [i for i in range(len(points)) if is_pareto(i, points)]
print()
print(f"Pareto frontier ({len(pareto_idx)} cells):")
for i in pareto_idx:
    print(f"  {points[i][0]:<32}  acc={points[i][2]:.3f}  r={points[i][3]:+.3f}")


# Plot
fig, ax = plt.subplots(figsize=(11, 7))
for i, (label, color, acc, r_val) in enumerate(points):
    on_frontier = i in pareto_idx
    ms = 16 if on_frontier else 11
    edge = "black" if on_frontier else "none"
    lw = 1.8 if on_frontier else 0
    ax.scatter(r_val, acc, c=color, s=ms ** 1.8, edgecolors=edge, linewidths=lw,
               zorder=5 if on_frontier else 3, alpha=0.95)
    # Label with slight offset
    ax.annotate(label, (r_val, acc), xytext=(7, 4), textcoords="offset points",
                fontsize=9, color="#222",
                fontweight="bold" if on_frontier else "normal")

# Pareto frontier line (connect frontier points sorted by r)
front_pts = sorted([points[i] for i in pareto_idx], key=lambda p: p[3])
front_x = [p[3] for p in front_pts]
front_y = [p[2] for p in front_pts]
ax.plot(front_x, front_y, color="#888", lw=1.5, ls="--", alpha=0.6, zorder=1,
        label="Pareto frontier")

ax.axhline(0, color="#ccc", lw=0.6)
ax.axvline(0, color="#ccc", lw=0.6)
ax.set_xlabel("Pearson r(elapsed_at_commit, T)  ←  pacing quality / T-tracking  →",
              fontsize=11)
ax.set_ylabel("Accuracy  ↑", fontsize=11)
ax.set_xlim(-0.2, 1.0); ax.set_ylim(0, 0.6)
ax.set_title("Pareto: accuracy vs T-tracking (eval set, matched prompt)\n"
             "Frontier highlighted with bold outline; up-and-to-the-right = better on both",
             fontsize=11)
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.grid(alpha=0.25)

out = pathlib.Path("analysis/figures/51_pareto_acc_vs_r.png")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {out}")
