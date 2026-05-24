"""Analyze the v2 minimal run (Qwen3-4B, c*min(1,T/t), no bonus, no 5T cutoff, uniform T).

Reads per-checkpoint eval rollouts from analysis/eval_rollouts/v2/step*_eval.jsonl and:
  - prints the per-budget behavior table (accuracy, commit rate, elapsed, in-budget, turns)
  - runs the paired analysis (same problems at every budget) that disentangles the
    turns "selection effect" from genuine budget-conditioned correctness
  - runs a logistic regression of is_correct ~ T (significance of the pacing slope)
  - renders the two table PNGs into analysis/figures/

Key findings (final checkpoint, step 200, single seed, temp-0 eval — caveats apply):
  - accuracy rises with T (0.37 -> 0.48); logistic reg slope p ~= 0.04
  - this is NOT time-pacing: mean elapsed is ~flat; the budget shifts the COMMIT
    threshold (commit less often but more accurately at high T)
  - the "more turns at small T" is mostly a selection effect — paired on the same
    problems committed at both budgets, turns are ~equal (7.5 vs 7.0)
  - paired correctness: 18 problems flip wrong->correct at T=130 vs 7 the other way
    (McNemar p ~= 0.046)

Usage:
    python scripts/dev/analyze_v2.py            # final checkpoint tables + figures
    python scripts/dev/analyze_v2.py --trajectory   # also print per-checkpoint is_correct-vs-T
"""
from __future__ import annotations
import json
import math
import pathlib
import statistics as st
import sys

ROLL = pathlib.Path("analysis/eval_rollouts/v2")
FIG = pathlib.Path("analysis/figures")
T_CELLS = [15, 45, 75, 105, 130]


def load(step: int) -> list[dict]:
    return [json.loads(l) for l in (ROLL / f"step{step}_eval.jsonl").open()]


def by_budget(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(round(r["info"]["target_s"]), []).append(r)
    return out


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def elapsed(r):
    return r["elapsed_over_target"] * r["info"]["target_s"]


def per_budget_table(rows):
    bt = by_budget(rows)
    table = []
    for T in sorted(bt):
        rs = bt[T]
        c = [r for r in rs if r["is_timeout"] == 0]
        table.append({
            "T": T,
            "accuracy": mean([r["is_correct"] for r in rs]),
            "commit_pct": len(c) / len(rs),
            "acc_given_commit": mean([r["is_correct"] for r in c]),
            "avg_elapsed": mean([elapsed(r) for r in rs]),
            "elapsed_commit": mean([elapsed(r) for r in c]),
            "elapsed_over_T_commit": mean([r["elapsed_over_target"] for r in c]),
            "inbudget_pct": mean([1.0 if r["elapsed_over_target"] <= 1 else 0.0 for r in c]),
            "turns_commit": mean([r["num_turns"] for r in c]),
        })
    return table


def logistic_slope(rows):
    """Logistic regression is_correct ~ T (standardized). Returns (slope, se, z, p)."""
    xs = [r["info"]["target_s"] for r in rows]
    ys = [r["is_correct"] for r in rows]
    mx = sum(xs) / len(xs)
    sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    xn = [(x - mx) / sx for x in xs]
    b0 = b1 = 0.0
    for _ in range(5000):
        g0 = g1 = 0.0
        for x, y in zip(xn, ys):
            p = 1 / (1 + math.exp(-(b0 + b1 * x)))
            g0 += p - y
            g1 += (p - y) * x
        b0 -= 0.1 * g0 / len(xn)
        b1 -= 0.1 * g1 / len(xn)
    W = sum((1 / (1 + math.exp(-(b0 + b1 * x)))) * (1 - 1 / (1 + math.exp(-(b0 + b1 * x)))) * x * x for x in xn)
    se = 1 / W ** 0.5
    z = b1 / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return b1, se, z, p


def paired_analysis(rows, lo=15, hi=130):
    idx = {(r["example_id"], round(r["info"]["target_s"])): r for r in rows}
    ids = sorted(set(r["example_id"] for r in rows))
    pairs = [(idx.get((e, lo)), idx.get((e, hi))) for e in ids if idx.get((e, lo)) and idx.get((e, hi))]
    both_commit = [(a, b) for a, b in pairs if a["is_timeout"] == 0 and b["is_timeout"] == 0]
    turns_lo = mean([a["num_turns"] for a, b in both_commit])
    turns_hi = mean([b["num_turns"] for a, b in both_commit])
    both_c = sum(1 for a, b in pairs if a["is_correct"] and b["is_correct"])
    only_hi = sum(1 for a, b in pairs if b["is_correct"] and not a["is_correct"])
    only_lo = sum(1 for a, b in pairs if a["is_correct"] and not b["is_correct"])
    both_w = sum(1 for a, b in pairs if not a["is_correct"] and not b["is_correct"])
    chi = (abs(only_hi - only_lo) - 1) ** 2 / (only_hi + only_lo) if (only_hi + only_lo) else float("nan")
    p = 1 - math.erf(math.sqrt(chi / 2)) if chi == chi else float("nan")
    return {
        "n_both_commit": len(both_commit), "turns_lo": turns_lo, "turns_hi": turns_hi,
        "both_correct": both_c, "only_hi": only_hi, "only_lo": only_lo, "both_wrong": both_w,
        "mcnemar_chi2": chi, "mcnemar_p": p,
    }


def render_figures(rows):
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    tab = per_budget_table(rows)

    # PNG 1
    cols = ["T (s)", "accuracy", "commit\n%", "acc |\ncommit", "avg\nelapsed",
            "elapsed |\ncommit", "elapsed/T\n| commit", "in-budget %\n(of commits)", "turns |\ncommit"]
    data = [[f"{r['T']}", f"{r['accuracy']:.2f}", f"{r['commit_pct']*100:.0f}%", f"{r['acc_given_commit']:.2f}",
             f"{r['avg_elapsed']:.0f}s", f"{r['elapsed_commit']:.0f}s", f"{r['elapsed_over_T_commit']:.1f}x",
             f"{r['inbudget_pct']*100:.0f}%", f"{r['turns_commit']:.1f}"] for r in tab]
    fig, ax = plt.subplots(figsize=(14, 3.0)); ax.axis("off")
    t = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=[0.07, 0.11, 0.09, 0.10, 0.11, 0.12, 0.12, 0.14, 0.10])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 2.6)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#2c3e50"); t[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data) + 1):
        t[i, 1].set_facecolor("#eaf2f8"); t[i, 3].set_facecolor("#eaf2f8")
    ax.set_title("Qwen3-4B v2 — final checkpoint (step 200), held-out eval (n=100/cell, temp 0)", fontsize=12, pad=14)
    fig.tight_layout(); fig.savefig(FIG / "16_v2_per_budget_table.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # PNG 2
    pa = paired_analysis(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 2.8))
    for a in axes: a.axis("off")
    axes[0].set_title("Turns | commit  (T=15 vs T=130)", fontsize=11.5, pad=12)
    tA = [["all commits", "9.5", "7.0"],
          [f"same {pa['n_both_commit']} (commit at both)", f"{pa['turns_lo']:.1f}", f"{pa['turns_hi']:.1f}"]]
    ta = axes[0].table(cellText=tA, colLabels=["", "T=15", "T=130"], loc="center", cellLoc="center",
                       colWidths=[0.46, 0.22, 0.22])
    ta.auto_set_font_size(False); ta.set_fontsize(11); ta.scale(1, 2.8)
    for j in range(3): ta[0, j].set_facecolor("#2c3e50"); ta[0, j].set_text_props(color="white", fontweight="bold")
    axes[1].set_title("Correctness — same 100 problems, T=15 vs T=130", fontsize=11.5, pad=12)
    tB = [["T=130 correct", f"{pa['both_correct']}", f"{pa['only_hi']}", f"{pa['both_correct']+pa['only_hi']}"],
          ["T=130 wrong", f"{pa['only_lo']}", f"{pa['both_wrong']}", f"{pa['only_lo']+pa['both_wrong']}"],
          ["total", f"{pa['both_correct']+pa['only_lo']}", f"{pa['only_hi']+pa['both_wrong']}", "100"]]
    tb = axes[1].table(cellText=tB, colLabels=["", "T=15 correct", "T=15 wrong", "total"], loc="center",
                       cellLoc="center", colWidths=[0.26, 0.25, 0.24, 0.16])
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 2.8)
    for j in range(4): tb[0, j].set_facecolor("#2c3e50"); tb[0, j].set_text_props(color="white", fontweight="bold")
    tb[1, 2].set_facecolor("#d5f5e3"); tb[2, 1].set_facecolor("#fadbd8")
    fig.suptitle("Paired analysis — same 100 problems run at every budget", fontsize=12, y=1.04)
    fig.tight_layout(); fig.savefig(FIG / "17_v2_paired_analysis.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {FIG/'16_v2_per_budget_table.png'} and {FIG/'17_v2_paired_analysis.png'}")


def main():
    final = load(200)
    print("=== FINAL CHECKPOINT (step 200) per-budget table ===")
    print(f"{'T':>4} {'acc':>5} {'commit%':>8} {'acc|cm':>7} {'elapsed':>8} {'el|cm':>7} {'el/T|cm':>8} {'inbud%':>7} {'turns|cm':>9}")
    for r in per_budget_table(final):
        print(f"{r['T']:>4} {r['accuracy']:>5.2f} {r['commit_pct']*100:>7.0f}% {r['acc_given_commit']:>7.2f} "
              f"{r['avg_elapsed']:>7.0f}s {r['elapsed_commit']:>6.0f}s {r['elapsed_over_T_commit']:>7.1f}x "
              f"{r['inbudget_pct']*100:>6.0f}% {r['turns_commit']:>9.1f}")
    b1, se, z, p = logistic_slope(final)
    print(f"\nlogistic is_correct ~ T:  slope={b1:.3f} se={se:.3f} z={z:.2f} p={p:.4f}")
    pa = paired_analysis(final)
    print(f"\npaired turns (n={pa['n_both_commit']} commit-at-both): T15={pa['turns_lo']:.1f} T130={pa['turns_hi']:.1f}")
    print(f"paired correctness: both={pa['both_correct']} only_T130={pa['only_hi']} only_T15={pa['only_lo']} "
          f"both_wrong={pa['both_wrong']}  McNemar chi2={pa['mcnemar_chi2']:.1f} p={pa['mcnemar_p']:.4f}")

    if "--trajectory" in sys.argv:
        print("\n=== is_correct vs T across checkpoints ===")
        print(f"{'step':>5} " + " ".join(f"T={T:>3}" for T in T_CELLS))
        for step in [20, 40, 60, 80, 100, 200]:
            bt = by_budget(load(step))
            print(f"{step:>5} " + " ".join(f"{mean([r['is_correct'] for r in bt[T]]):>5.2f}" for T in T_CELLS))

    render_figures(final)


if __name__ == "__main__":
    main()
