"""Prompt-salience probe — does the `remaining_budget` prompt unlock T-conditioning?

For each of the 4 cells (base, long-500) × (base, remaining_budget) prompts:
  1. Parse each rollout's completion for elapsed_at_commit, num_turns, etc.
  2. Compute Pearson r(behavior, T) — the headline T-conditioning signal.
  3. Compare against the long-500 base-prompt reference (r ≈ -0.04).

Outputs:
  - analysis/figures/38_prompt_salience_commit_time_vs_T.png
  - analysis/figures/39_prompt_salience_correlations.png (bar chart)
  - printed table of r per cell per signal
"""
from __future__ import annotations
import json, math, pathlib, re, statistics as st

DATA = pathlib.Path("analysis/eval_rollouts/prompt_salience/prompt_salience")
CELLS = [
    ("base / base prompt",              "base_base.jsonl",                  "#7f8fa6"),
    ("base / remaining_budget",         "base_remaining_budget.jsonl",      "#27ae60"),
    ("long-500 / base prompt",          "long-500_base.jsonl",              "#2C3E50"),
    ("long-500 / remaining_budget",     "long-500_remaining_budget.jsonl",  "#C2185B"),
    ("long-strict / base prompt",       "long-strict_base.jsonl",           "#e67e22"),
    ("long-strict / remaining_budget",  "long-strict_remaining_budget.jsonl","#8e44ad"),
    ("long-additive / base prompt",         "long-additive_base.jsonl",             "#006D77"),
    ("long-additive / remaining_budget",    "long-additive_remaining_budget.jsonl", "#E63946"),
    ("v2-λ=0.10 / base prompt",             "long-additive-v2-l10_base.jsonl",             "#B0BEC5"),
    ("v2-λ=0.10 / remaining_budget",        "long-additive-v2-l10_remaining_budget.jsonl", "#1976D2"),
    ("v2-λ=0.15 / base prompt",             "long-additive-v2-l15_base.jsonl",             "#A5D6A7"),
    ("v2-λ=0.15 / remaining_budget",        "long-additive-v2-l15_remaining_budget.jsonl", "#2E7D32"),
    ("v2-λ=0.30 / base prompt",             "long-additive-v2-l30_base.jsonl",             "#FFAB91"),
    ("v2-λ=0.30 / remaining_budget",        "long-additive-v2-l30_remaining_budget.jsonl", "#BF360C"),
]
BINS = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 40.001)]

# The `remaining_budget` variant injects messages like "[8.5s elapsed, 3.5s remaining of your 12s budget]".
# The `base` variant injects "[8.5s elapsed]". Both start with `[N.Ns elapsed`.
ELAPSED_RE = re.compile(r"\[([\d.]+)s elapsed")


def parse_completion(text: str) -> dict:
    # eval_prompt_salience.py's chat_to_text includes the SYSTEM PROMPT, which
    # contains the literal string "<answer>...</answer>" as the answer-format
    # example. To find the MODEL's actual answer tag, anchor the search to text
    # AFTER the first assistant turn begins.
    body_start = text.find("<|im_start|>assistant")
    body = text[body_start:] if body_start != -1 else text
    body_off = body_start if body_start != -1 else 0

    matches = list(ELAPSED_RE.finditer(body))
    num_turns = len(matches)
    final_elapsed = float(matches[-1].group(1)) if matches else None
    ans_pos = body.find("<answer>")
    has_ans = ans_pos != -1
    eac = None
    if has_ans:
        prior = [float(m.group(1)) for m in matches if m.start() < ans_pos]
        eac = prior[-1] if prior else 0.0
    return {"comp_chars": len(text), "num_turns": num_turns, "has_answer": has_ans,
            "elapsed_at_commit": eac, "final_elapsed": final_elapsed}


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2: return float("nan"), float("nan")
    xs, ys = zip(*pairs); n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0: return float("nan"), float("nan")
    r = sxy / (sxx*syy)**0.5
    if abs(r) >= 1.0: return r, 0.0
    z = 0.5 * math.log((1+r)/(1-r)); se = 1 / (n-3)**0.5
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z/se) / math.sqrt(2))))
    return r, p


def load_cell(jsonl_path: pathlib.Path) -> list[dict]:
    recs = []
    for line in jsonl_path.open():
        r = json.loads(line)
        r.update(parse_completion(r["completion"]))
        recs.append(r)
    return recs


def main():
    data = {}
    for label, fname, _ in CELLS:
        recs = load_cell(DATA / fname)
        # Mark each rec with is_correct from reward sign (attempt_bonus=0 means reward>0 <=> correct)
        for r in recs:
            r["is_correct"] = 1 if r.get("reward", 0) > 0 else 0
        data[label] = recs

    # ---- Printed table: Pearson r per cell per signal ----
    print("=" * 95)
    print("Pearson r (behavior ~ T) — does prompt salience unlock T-conditioning?")
    print("=" * 95)
    print(f"  {'cell':<32} {'comp_chars':>11} {'num_turns':>10} {'commit %':>10} "
          f"{'final_elap':>11} {'elap@commit':>12} {'n_commit':>9}")
    for label, _, _ in CELLS:
        recs = data[label]
        xs = [r["target_s"] for r in recs]
        committers = [r for r in recs if r["has_answer"] and r["elapsed_at_commit"] is not None]
        r_cc, _ = pearson(xs, [r["comp_chars"]    for r in recs])
        r_nt, _ = pearson(xs, [r["num_turns"]     for r in recs])
        r_co, _ = pearson(xs, [r["has_answer"]    for r in recs])
        r_fe, _ = pearson(xs, [r["final_elapsed"] for r in recs])
        r_ec, _ = pearson([r["target_s"] for r in committers],
                          [r["elapsed_at_commit"] for r in committers])
        n_c = len(committers)
        print(f"  {label:<32} {r_cc:>+11.3f} {r_nt:>+10.3f} {r_co:>+10.3f} "
              f"{r_fe:>+11.3f} {r_ec:>+12.3f} {n_c:>9}")
    print()

    # ---- Per-bin commit time table ----
    print("=" * 95)
    print("Mean elapsed_at_commit by T-bin (committers only)")
    print("=" * 95)
    header = "  " + f"{'cell':<32}" + "".join(f"  T={lo:>2}-{int(hi):<3}" for lo, hi in BINS) + "   mean"
    print(header)
    for label, _, _ in CELLS:
        recs = data[label]
        bins_means = []
        all_eac = []
        for lo, hi in BINS:
            committers = [r for r in recs if r["has_answer"]
                          and r["elapsed_at_commit"] is not None
                          and lo <= r["target_s"] < hi]
            if committers:
                m = st.mean(r["elapsed_at_commit"] for r in committers)
                bins_means.append(f"   {m:>5.1f}s ")
                all_eac.extend(r["elapsed_at_commit"] for r in committers)
            else:
                bins_means.append("       -- ")
        overall = st.mean(all_eac) if all_eac else float("nan")
        print(f"  {label:<32}" + "".join(bins_means) + f"  {overall:>5.1f}s")
    print()

    # ---- Figures ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)

    # Figure 38: elapsed_at_commit vs T, all 4 cells (bin means only — no scatter)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for label, _, color in CELLS:
        committers = [r for r in data[label] if r["has_answer"]
                      and r["elapsed_at_commit"] is not None]
        bx, by = [], []
        for lo, hi in BINS:
            b = [r for r in committers if lo <= r["target_s"] < hi]
            if not b: continue
            bx.append(st.mean(r["target_s"] for r in b))
            by.append(st.mean(r["elapsed_at_commit"] for r in b))
        ax.plot(bx, by, marker="o", ms=10, color=color, lw=2.8,
                label=f"{label} (n_commit={len(committers)})")
    ax.plot([0, 40], [0, 40], color="#999", lw=1.5, ls=":", label="t = T (on-budget)")
    ax.set_xlabel("Budget T (s)", fontsize=12)
    ax.set_ylabel("Elapsed at commit (s)", fontsize=12)
    ax.set_xlim(0, 40); ax.set_ylim(0, 45)
    ax.set_title("Commit time vs budget T", fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    out = FIG / "38_prompt_salience_commit_time_vs_T.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # Figure 39: bar chart of |Pearson r| for the headline signal (elapsed_at_commit)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    labels = [l for l, _, _ in CELLS]
    colors = [c for _, _, c in CELLS]
    rs = []
    for label, _, _ in CELLS:
        committers = [r for r in data[label] if r["has_answer"]
                      and r["elapsed_at_commit"] is not None]
        r_, _ = pearson([r["target_s"] for r in committers],
                        [r["elapsed_at_commit"] for r in committers])
        rs.append(r_)
    bars = ax.bar(labels, rs, color=colors, alpha=0.85)
    ax.axhline(0, color="#666", lw=0.8)
    ax.axhline(0.2, color="#c66", lw=0.7, ls="--", label="|r|=0.2 (rough threshold for real effect)")
    ax.axhline(-0.2, color="#c66", lw=0.7, ls="--")
    for bar, r_ in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.03),
                f"{r_:+.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Pearson r(elapsed_at_commit, T)", fontsize=11)
    ax.set_ylim(-0.35, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    ax.set_title("T-conditioning signal: does commit time track T?", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=8, ha="right")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    out2 = FIG / "39_prompt_salience_correlations.png"
    fig.tight_layout(); fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
