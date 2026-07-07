"""Predicted-vs-measured scatter for the hwprop simulator (slide version).

Every point is one measured (GPU, model, context) benchmark config from the
hardware-proprioception grid (full_cache rows). Two predictions per point:
naive spec-sheet roofline (red, collapses far below the diagonal) and the
calibrated hwprop model (colored by GPU, hugging the diagonal). A100-80GB —
the GPU every interoception RL run simulates — gets black-edged markers.
"""
from __future__ import annotations
import csv, pathlib, statistics as st
import matplotlib.pyplot as plt
from hwprop.simulator import simulate_latency
from hwprop.specs import get_model_configs
from hwprop.overhead import OverheadProfile

GRID = pathlib.Path("/Users/harshsingh/Development/hardware-proprioception/results/grid")
OUT = pathlib.Path("analysis/figures/65_sim_scatter.png")

GPU_COLORS = {"H200": "#1f77b4", "H100_SXM": "#17becf", "A100_80GB": "#e67e22",
              "L40S": "#9467bd", "A40": "#2ca02c"}
ROOFLINE = OverheadProfile(name="pure_roofline", roofline_efficiency=1.0,
                           launch_overhead_s=0.0, attn_scan_coeff=0.0,
                           kv_bandwidth_alpha=0.0, kv_bandwidth_beta=1.0)


def match_model(dirname, catalog):
    for cand in (dirname, dirname.replace("-Instruct", "").replace("-Base", "").replace("-it", "")):
        for key in catalog:
            if key.lower() == cand.lower():
                return key
    return None


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy)


def main():
    catalog = get_model_configs()
    pts = []  # (gpu, model, measured, calibrated, roofline)
    for mdir in sorted(GRID.iterdir()):
        # Scope to the Qwen family: the model class every interoception RL run
        # simulates (sim_model="Qwen3-4B"; A100 profile anchored on Qwen2.5-7B).
        # The universal equation is known-weak on <2B models (see hwprop deck).
        if not mdir.is_dir() or not mdir.name.startswith("Qwen"):
            continue
        key = match_model(mdir.name, catalog)
        if key is None:
            print(f"  (no catalog match for {mdir.name}, skipped)")
            continue
        for f in sorted(mdir.glob("*.csv")):
            for r in csv.DictReader(f.open()):
                if r["strategy"] != "full_cache" or int(r["batch_size"]) != 1:
                    continue
                gpu, ctx = r["hardware_key"], int(r["context_length"])
                steps = int(r["num_decode_steps"])
                meas = float(r["mean_ms_per_step"])
                cal = simulate_latency(gpu, key, prompt_len=ctx,
                                       decode_steps=steps).mean_per_token_ms
                rf = simulate_latency(gpu, key, prompt_len=ctx, decode_steps=steps,
                                      overhead=ROOFLINE).mean_per_token_ms
                pts.append((gpu, key, ctx, meas, cal, rf))

    meas = [p[3] for p in pts]
    cal = [p[4] for p in pts]
    rf = [p[5] for p in pts]
    mae_cal = st.mean(abs(c - m) / m for m, c in zip(meas, cal)) * 100
    mae_rf = st.mean(abs(x - m) / m for m, x in zip(meas, rf)) * 100
    rho_cal, rho_rf = spearman(meas, cal), spearman(meas, rf)

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 160,
    })
    # square scatter with the legend beside it; no title (caption lives on the slide)
    fig, ax = plt.subplots(figsize=(12.8, 5.25))
    ax.set_box_aspect(1)
    lo, hi = 2, 80
    ax.plot([lo, hi], [lo, hi], color="#666", lw=1.2, ls=":", zorder=1)

    # naive roofline cloud
    ax.scatter(meas, rf, s=42, color="#c0392b", alpha=0.35, marker="v",
               label=f"Spec-sheet roofline (MAE {mae_rf:.0f}%, ρ={rho_rf:.2f})", zorder=2)
    # calibrated, colored by GPU
    for gpu, color in GPU_COLORS.items():
        sub = [(m, c) for g, k, ctx, m, c, _ in pts if g == gpu]
        if not sub:
            continue
        a100 = gpu == "A100_80GB"
        ax.scatter([m for m, _ in sub], [c for _, c in sub], s=110 if a100 else 70,
                   color=color, alpha=0.9, edgecolors="black" if a100 else "none",
                   linewidths=1.3, zorder=4 if a100 else 3,
                   label=f"hwprop · {gpu}" + ("  ← RL config" if a100 else ""))
    ax.legend(frameon=False, fontsize=12.5, loc="center left",
              bbox_to_anchor=(1.04, 0.5), markerscale=1.2, labelspacing=0.7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured latency (ms/token)")
    ax.set_ylabel("Predicted latency (ms/token)")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"n={len(pts)}  calibrated MAE {mae_cal:.1f}% ρ={rho_cal:.3f}  |  "
          f"roofline MAE {mae_rf:.1f}% ρ={rho_rf:.3f}")
    a100 = [(m, c) for g, k, ctx, m, c, _ in pts if g == "A100_80GB"]
    print(f"A100-only MAE: {st.mean(abs(c-m)/m for m,c in a100)*100:.1f}%  (n={len(a100)})")


if __name__ == "__main__":
    main()
