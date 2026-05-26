"""Analyze the 3-way f(t,T) control ablation (ctrl0 treatment vs ctrlA vs ctrlB).

Pulls the held-out eval sample tables from wandb (uniform T~U(15,130), temp 1.0,
dataset_seed=777 -> identical (problem,T) pairs across all three), derives per-rollout
target_s from the seed, and uses the fact that with attempt_bonus=0:
    reward > 0  <=>  is_correct      (all conditions)
    reward      ==  c * f(t,T)       (ctrl0 only; so reward == f among correct)
So is_correct and (for the treatment) the pacing factor f come straight from reward —
no completion re-scoring needed (the env already scored with the fixed solver).

Primary contrast: treatment (ctrl0) vs ctrlA — both multi-turn, differ ONLY in the time
reward. ctrlB is single-turn (max_turns=1), so it differs from A on TWO axes (signal +
turn structure); reported alongside with that caveat.

Outputs: per-rollout cache (analysis/eval_rollouts/controls/), is_correct~T binned
(base step0 vs final step200), logistic is_correct~T slope, and ctrl0 pacing f~T.

Usage: python scripts/dev/analyze_controls.py
"""
from __future__ import annotations
import json, math, os, pathlib, random, tempfile, statistics as st

ENTITY = "singhh5050-stanford-university/interoception"
RUNS = {
    "ctrl0": "ctrl0-qwen3-4b-treatment",   # reward c*f  (treatment)
    "ctrlA": "ctrlA-qwen3-4b-noTimeReward", # reward c, multi-turn, elapsed injected
    "ctrlB": "ctrlB-qwen3-4b-noTimeSignal", # reward c, single turn, no elapsed
}
EVAL_SEED, T_LO, T_HI = 777, 15.0, 130.0
CACHE = pathlib.Path("analysis/eval_rollouts/controls")
BASE_STEP_MAX, FINAL_STEP_MIN = 1, 199   # step 0 eval ~ "0"; final eval ~ "200"
BINS = [(15, 38), (38, 61), (61, 84), (84, 107), (107, 130.001)]


def target_s_for(example_id: int) -> float:
    """Replicates env _load_dataset uniform branch for dataset_seed=777."""
    rng = random.Random(EVAL_SEED ^ (example_id * 2654435761 & 0xFFFFFFFF))
    return rng.uniform(T_LO, T_HI)


def pull(cond: str, wandb_name: str) -> list[dict]:
    """Pull all eval-sample-table rows for a run -> per-rollout records (cached)."""
    out = CACHE / f"{cond}.jsonl"
    if out.exists():
        return [json.loads(l) for l in out.open()]
    import wandb
    api = wandb.Api()
    run = sorted(api.runs(ENTITY, filters={"display_name": wandb_name}),
                 key=lambda r: r.created_at, reverse=True)[0]
    tbls = [f for f in run.files() if "table/eval" in f.name and f.name.endswith(".json")]
    d = tempfile.mkdtemp()
    recs = []
    for f in tbls:
        f.download(root=d, replace=True)
        j = json.load(open(os.path.join(d, f.name)))
        c = j["columns"]; si, ri, ei = c.index("step"), c.index("reward"), c.index("example_id")
        for row in j["data"]:
            ex = int(row[ei]); rew = float(row[ri])
            recs.append({"condition": cond, "step": int(row[si]), "example_id": ex,
                         "target_s": target_s_for(ex), "reward": rew,
                         "is_correct": 1 if rew > 0 else 0,
                         "f": rew if rew > 0 else None})  # f only defined among correct (ctrl0)
    CACHE.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def logistic_slope(rows):
    xs = [r["target_s"] for r in rows]; ys = [r["is_correct"] for r in rows]
    mx = sum(xs) / len(xs); sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    xn = [(x - mx) / sx for x in xs]; b0 = b1 = 0.0
    for _ in range(4000):
        g0 = g1 = 0.0
        for x, y in zip(xn, ys):
            p = 1 / (1 + math.exp(-(b0 + b1 * x))); g0 += p - y; g1 += (p - y) * x
        b0 -= 0.1 * g0 / len(xn); b1 -= 0.1 * g1 / len(xn)
    W = sum((1 / (1 + math.exp(-(b0 + b1 * x)))) * (1 - 1 / (1 + math.exp(-(b0 + b1 * x)))) * x * x for x in xn)
    se = 1 / W ** 0.5; z = b1 / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return b1, se, z, p


def binned(rows, field="is_correct"):
    out = []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["target_s"] < hi]
        vals = [r[field] for r in b if r.get(field) is not None]
        out.append((lo, hi, len(b), st.mean(vals) if vals else float("nan")))
    return out


def _binned_points(recs, which):
    """(mean_T, accuracy, binomial_SE) per bin for the base/final eval step."""
    s = min(r["step"] for r in recs) if which == "base" else max(r["step"] for r in recs)
    rows = [r for r in recs if r["step"] == s]
    xs, ys, es = [], [], []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["target_s"] < hi]
        if not b:
            continue
        p = sum(r["is_correct"] for r in b) / len(b)
        xs.append(sum(r["target_s"] for r in b) / len(b)); ys.append(p)
        es.append((p * (1 - p) / len(b)) ** 0.5)
    return xs, ys, es


def render_acc_vs_T(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, GRAY, FAINT = "#C2185B", "#2C3E50", "#8E9BA8", "#C9CDD2"
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # treatment base (untrained) — flat reference, dashed
    x, y, e = _binned_points(data["ctrl0"], "base")
    ax.plot(x, y, marker="o", ms=5, color=FAINT, lw=1.4, ls="--", label="treatment — base (untrained)")
    # final checkpoints
    x, y, e = _binned_points(data["ctrlB"], "final")
    ax.errorbar(x, y, yerr=e, marker="^", ms=6, color=GRAY, lw=1.6, capsize=3, label="ctrlB (c, single-turn) — final")
    x, y, e = _binned_points(data["ctrlA"], "final")
    ax.errorbar(x, y, yerr=e, marker="s", ms=6, color=NAVY, lw=1.9, capsize=3, label="ctrlA (c, no time reward) — final")
    x, y, e = _binned_points(data["ctrl0"], "final")
    ax.errorbar(x, y, yerr=e, marker="o", ms=7, color=PINK, lw=2.6, capsize=3, label="treatment (c·f) — final")
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Accuracy (held-out eval)")
    ax.set_xticks([15, 45, 75, 105, 130]); ax.set_ylim(0, 0.65)
    ax.set_title("Accuracy vs budget — Qwen3-4B, held-out eval")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "22_controls_acc_vs_T.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", FIG / "22_controls_acc_vs_T.png")


def render_table(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    labels = {"ctrl0": "treatment (c·f)", "ctrlA": "ctrlA — no time reward", "ctrlB": "ctrlB — single-turn"}
    cols = ["condition", "stage", "15–38", "38–61", "61–84", "84–107", "107–130", "overall", "slope vs T (p)"]
    cw = [0.19, 0.11, 0.082, 0.082, 0.082, 0.082, 0.082, 0.09, 0.14]
    rows, star = [], None
    for c in ("ctrl0", "ctrlA", "ctrlB"):
        recs = data[c]
        for stage in ("base", "final"):
            s = min(r["step"] for r in recs) if stage == "base" else max(r["step"] for r in recs)
            rr = [r for r in recs if r["step"] == s]
            cells = [f"{m:.2f}" for *_, m in binned(rr, "is_correct")]
            overall = sum(r["is_correct"] for r in rr) / len(rr)
            b1, se, z, p = logistic_slope(rr)
            rows.append([labels[c], f"{stage} (s{s})", *cells, f"{overall:.2f}", f"{b1:+.2f} (p={p:.3f})"])
            if c == "ctrl0" and stage == "final":
                star = len(rows)
    fig, ax = plt.subplots(figsize=(12.5, 3.4)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colWidths=cw)
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.2)
    NAVY, PINK, SHADE = "#2C3E50", "#F7D5E0", "#EEF1F4"
    for j in range(len(cols)):
        t[0, j].set_facecolor(NAVY); t[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        t[i, len(cols) - 2].set_facecolor(SHADE); t[i, len(cols) - 1].set_facecolor(SHADE)
    if star:
        for j in range(len(cols)):
            t[star, j].set_facecolor(PINK)
        t[star, len(cols) - 1].set_text_props(fontweight="bold")
    ax.set_title("Accuracy by budget bin — Qwen3-4B held-out eval (base vs final)", fontsize=12, pad=12)
    fig.tight_layout(); fig.savefig(FIG / "23_controls_acc_table.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", FIG / "23_controls_acc_table.png")


def main():
    data = {c: pull(c, nm) for c, nm in RUNS.items()}
    render_acc_vs_T(data)
    render_table(data)
    for c, recs in data.items():
        steps = sorted(set(r["step"] for r in recs))
        print(f"{c}: {len(recs)} rollouts, eval steps {steps}, mean T={st.mean([r['target_s'] for r in recs]):.1f}")
    print()

    def step_rows(c, which):
        recs = data[c]
        if which == "base":
            s = min(r["step"] for r in recs)
        else:
            s = max(r["step"] for r in recs)
        return [r for r in recs if r["step"] == s], s

    # --- is_correct ~ T, base vs final, per condition ---
    print("=" * 78)
    print("is_correct by budget bin   (base = step0,  final = step~200)")
    print("=" * 78)
    hdr = "cond   stage   " + "".join(f"{lo:>3}-{int(hi):<3} " for lo, hi in BINS) + "  overall  logit-slope(p)"
    print(hdr)
    for c in RUNS:
        for which in ("base", "final"):
            rows, s = step_rows(c, which)
            bt = binned(rows, "is_correct")
            cells = "".join(f"{m:>7.2f} " for _, _, _, m in bt)
            overall = st.mean([r["is_correct"] for r in rows])
            if which == "final":
                b1, se, z, p = logistic_slope(rows)
                slope = f"  {b1:+.2f} (p={p:.3f})"
            else:
                slope = ""
            print(f"{c:5s}  {which:5s}({s:>3}) {cells}  {overall:>6.2f}{slope}")
        print()

    # --- treatment pacing: f(t,T) ~ T among correct (final) ---
    print("=" * 78)
    print("ctrl0 (treatment) pacing — mean f(t,T)=min(1,T/t) among CORRECT rollouts, final")
    print("(f<1 => over budget. Does in-budget-ness depend on T?)")
    print("=" * 78)
    rows, s = step_rows("ctrl0", "final")
    correct = [r for r in rows if r["is_correct"]]
    for lo, hi, n, _ in binned(rows):
        fb = [r["f"] for r in correct if lo <= r["target_s"] < hi]
        print(f"  T {lo:>3}-{int(hi):<3}: n_correct={len(fb):>3}  mean f={st.mean(fb):.3f}" if fb else f"  T {lo:>3}-{int(hi):<3}: n_correct=0")


if __name__ == "__main__":
    main()
