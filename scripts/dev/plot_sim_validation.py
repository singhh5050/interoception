"""Simulator-validation hero figure for the deck.

One pane, one story: on the exact GPU our RL clock simulates (A100-80GB),
spec-sheet physics (pure roofline) predicts ~5x faster than reality, while the
calibrated hwprop simulator sits on top of the measurements (1.8% MAE).
The shaded gap is what naive physics misses: CUDA kernel-launch overhead and
KV-cache bandwidth degradation.

Measured data: hardware-proprioception benchmark (Qwen2.5-7B, the calibration
anchor model for this GPU), context sweep 1K-131K, bs=1, FA2.
"""
from __future__ import annotations
import csv, pathlib, statistics as st
import matplotlib.pyplot as plt
from hwprop.simulator import simulate_latency
from hwprop.overhead import OverheadProfile

HWPROP = pathlib.Path("/Users/harshsingh/Development/hardware-proprioception")
CSV = HWPROP / "results/benchmark_cross_model_a100/context_sweep_A100_80GB.csv"
OUT = pathlib.Path("analysis/figures/64_sim_validation.png")

ROOFLINE = OverheadProfile(name="pure_roofline", roofline_efficiency=1.0,
                           launch_overhead_s=0.0, attn_scan_coeff=0.0,
                           kv_bandwidth_alpha=0.0, kv_bandwidth_beta=1.0)


def main():
    rows = list(csv.DictReader(CSV.open()))
    ctx = [int(r["context_length"]) for r in rows]
    meas = [float(r["measured_per_step_ms"]) for r in rows]
    err = [float(r["measured_stdev_per_step_ms"]) for r in rows]
    calib, roof = [], []
    for c, r in zip(ctx, rows):
        steps = int(r["decode_steps"])
        calib.append(simulate_latency("A100_80GB", "Qwen2.5-7B",
                                      prompt_len=c, decode_steps=steps).mean_per_token_ms)
        roof.append(simulate_latency("A100_80GB", "Qwen2.5-7B", prompt_len=c,
                                     decode_steps=steps, overhead=ROOFLINE).mean_per_token_ms)
    mae = st.mean(abs(s - m) / m for s, m in zip(calib, meas)) * 100
    gap = st.mean(m / r for m, r in zip(meas, roof))

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 160,
    })
    # 12.8 x 5.25 in ~ the 2.45:1 body area of a 16:9 slide under its title
    fig, ax = plt.subplots(figsize=(12.8, 5.25))

    # the gap naive physics can't see
    ax.fill_between(ctx, roof, meas, color="#c0392b", alpha=0.08, zorder=0)

    ax.plot(ctx, roof, color="#c0392b", lw=2.2, ls="--", marker="v", ms=5,
            label="Spec-sheet physics (naive roofline)", zorder=2)
    ax.plot(ctx, calib, color="#2e74b5", lw=3.0, marker="s", ms=5,
            label="hwprop (calibrated)", zorder=3)
    ax.errorbar(ctx, meas, yerr=err, fmt="o", color="#111", ms=6.5, capsize=3,
                lw=1.4, label="Measured on real A100-80GB", zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode latency (ms/token)")
    ax.set_title("Is the simulated clock real?  (A100-80GB — the GPU every RL run simulates)")
    ax.set_ylim(0, 58)

    # annotations
    ax.annotate(f"calibrated sim: MAE {mae:.1f}%",
                xy=(ctx[3], calib[3]), xytext=(1500, 44),
                fontsize=13, fontweight="bold", color="#2e74b5",
                arrowprops=dict(arrowstyle="->", color="#2e74b5", lw=1.6))
    ax.annotate(f"spec sheets alone: {gap:.0f}× too optimistic",
                xy=(ctx[4], roof[4]), xytext=(4500, 14),
                fontsize=13, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    ax.text(ctx[2], (roof[2] + meas[2]) / 2,
            "what naive physics misses:\nkernel-launch overhead + KV cache misses",
            fontsize=9.5, style="italic", color="#8b2f23", ha="center", va="center")

    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"MAE calibrated: {mae:.1f}%  |  roofline underprediction: {gap:.1f}x")


if __name__ == "__main__":
    main()
