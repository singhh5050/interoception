"""Audit + compute true correctness per (cell, T) from saved eval rollouts.

The wandb "Avg@1" is the weighted rubric reward, which mixes:
  - correctness_with_time (= 1 if correct & in-budget, decays past T)
  - 0.05 * parseable_bonus (= 1 if model emitted parseable arithmetic)

For the writeup we want raw correctness (is_correct) too, plus the time-decayed
flavor (correctness_with_time). All three are saved per-rollout to the volume.
"""
from __future__ import annotations
import json
import pathlib
import statistics as st

ROLL = pathlib.Path("analysis/eval_rollouts")
CELLS = sorted([p.stem for p in ROLL.glob("*.jsonl")])
T_BUCKETS = [15.0, 30.0, 60.0, 120.0]


def bucket_for(target_s: float) -> str:
    for T in T_BUCKETS:
        if abs(target_s - T) < 0.1:
            return f"T={int(T)}"
    return f"T={target_s:.0f}"


def cell_stats(jsonl_path: pathlib.Path) -> dict:
    by_T: dict[str, list[dict]] = {}
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            T = r.get("info", {}).get("target_s")
            if T is None:
                continue
            by_T.setdefault(bucket_for(T), []).append(r)
    out = {}
    for k, rs in by_T.items():
        out[k] = {
            "n": len(rs),
            "is_correct": st.mean(r["is_correct"] for r in rs),
            "correctness_with_time": st.mean(r["correctness_with_time"] for r in rs),
            "parseable": st.mean(r["is_parseable"] for r in rs),
            "is_quit": st.mean(r["is_quit"] for r in rs),
            "is_timeout": st.mean(r["is_timeout"] for r in rs),
            "is_wrong": st.mean(r["is_wrong"] for r in rs),
            "wandb_avg_at_1": st.mean(r["reward"] for r in rs),
            "n_turns": st.mean(r["num_turns"] for r in rs),
            "truncated": st.mean(r["is_truncated"] for r in rs),
            "elapsed_over_target": st.mean(r["elapsed_over_target"] for r in rs),
        }
    return out


def cell_short(name: str) -> str:
    # phase1_qwen25_3b_hyp_s0 -> qwen25-3b-hyp-s0
    parts = name.split("_")
    parts = parts[1:]  # drop phase prefix
    # Reassemble: model parts + shape + seed
    return "-".join(parts).replace("hyp_s", "hyp-s").replace("exp_s", "exp-s").replace("25_3b", "25-3b").replace("3_4b", "3-4b").replace("4_e4b", "4-e4b")


def main():
    all_data = {}
    for cell in CELLS:
        all_data[cell] = cell_stats(ROLL / f"{cell}.jsonl")

    # Print a wide table: cells x T, with raw is_correct
    print("\n=== RAW IS_CORRECT (true correctness on the 100-example eval set) ===\n")
    print(f"{'cell':28s}  " + "  ".join(f"{T:>7s}" for T in ["T=15","T=30","T=60","T=120"]))
    print("-" * 78)
    for cell in CELLS:
        row = []
        for T in ["T=15","T=30","T=60","T=120"]:
            d = all_data[cell].get(T)
            row.append(f"{d['is_correct']:.3f}" if d else "   -   ")
        print(f"{cell_short(cell):28s}  " + "  ".join(f"{v:>7s}" for v in row))

    print("\n=== CORRECTNESS_WITH_TIME (correct AND mostly in-budget) ===\n")
    print(f"{'cell':28s}  " + "  ".join(f"{T:>7s}" for T in ["T=15","T=30","T=60","T=120"]))
    print("-" * 78)
    for cell in CELLS:
        row = []
        for T in ["T=15","T=30","T=60","T=120"]:
            d = all_data[cell].get(T)
            row.append(f"{d['correctness_with_time']:.3f}" if d else "   -   ")
        print(f"{cell_short(cell):28s}  " + "  ".join(f"{v:>7s}" for v in row))

    print("\n=== WANDB Avg@1 (weighted reward; for cross-check) ===\n")
    print(f"{'cell':28s}  " + "  ".join(f"{T:>7s}" for T in ["T=15","T=30","T=60","T=120"]))
    print("-" * 78)
    for cell in CELLS:
        row = []
        for T in ["T=15","T=30","T=60","T=120"]:
            d = all_data[cell].get(T)
            row.append(f"{d['wandb_avg_at_1']:.3f}" if d else "   -   ")
        print(f"{cell_short(cell):28s}  " + "  ".join(f"{v:>7s}" for v in row))

    print("\n=== TIMEOUT RATE (fraction of rollouts that never emitted <answer>) ===\n")
    print(f"{'cell':28s}  " + "  ".join(f"{T:>7s}" for T in ["T=15","T=30","T=60","T=120"]))
    print("-" * 78)
    for cell in CELLS:
        row = []
        for T in ["T=15","T=30","T=60","T=120"]:
            d = all_data[cell].get(T)
            row.append(f"{d['is_timeout']:.3f}" if d else "   -   ")
        print(f"{cell_short(cell):28s}  " + "  ".join(f"{v:>7s}" for v in row))

    print("\n=== AVG ELAPSED_OVER_TARGET (1.0 = at budget; >1 = over) ===\n")
    print(f"{'cell':28s}  " + "  ".join(f"{T:>7s}" for T in ["T=15","T=30","T=60","T=120"]))
    print("-" * 78)
    for cell in CELLS:
        row = []
        for T in ["T=15","T=30","T=60","T=120"]:
            d = all_data[cell].get(T)
            row.append(f"{d['elapsed_over_target']:.2f}" if d else "   -   ")
        print(f"{cell_short(cell):28s}  " + "  ".join(f"{v:>7s}" for v in row))

    # Save the structured data
    out_json = pathlib.Path("analysis/data/final_eval_stats.json")
    with out_json.open("w") as f:
        json.dump({cell_short(c): all_data[c] for c in CELLS}, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
