"""Plot training curves from rl_train.py's train_log.jsonl.

Renders four panels: correctness, timeout rate, mean reward, avg elapsed (vs T).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in args.log.read_text().splitlines() if l.strip()]
    train = [r for r in rows if r.get("phase") == "train"]
    eval_ = [r for r in rows if r.get("phase") in ("eval", "init_eval")]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Correctness
    if train:
        axs[0, 0].plot([r["step"] for r in train],
                       [100 * r["correct"] / r["total_rollouts"] for r in train],
                       label="train rollouts", marker="o", ms=4)
    if eval_:
        axs[0, 0].plot([r["step"] for r in eval_],
                       [r["correct_pct"] for r in eval_],
                       label="eval", marker="x", ms=8, linewidth=2)
    axs[0, 0].set_xlabel("step")
    axs[0, 0].set_ylabel("correct %")
    axs[0, 0].set_title("Correctness")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    # Timeout rate
    if train:
        axs[0, 1].plot([r["step"] for r in train],
                       [100 * r["timeout"] / r["total_rollouts"] for r in train],
                       label="train", marker="o", ms=4)
    if eval_:
        axs[0, 1].plot([r["step"] for r in eval_],
                       [r["timeout_pct"] for r in eval_],
                       label="eval", marker="x", ms=8)
    axs[0, 1].set_xlabel("step")
    axs[0, 1].set_ylabel("timeout %")
    axs[0, 1].set_title("Timeout rate (lower = more committal)")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)

    # Mean reward
    if train:
        axs[1, 0].plot([r["step"] for r in train],
                       [r["avg_reward"] for r in train],
                       label="train", marker="o", ms=4)
    if eval_:
        axs[1, 0].plot([r["step"] for r in eval_],
                       [r["avg_reward"] for r in eval_],
                       label="eval", marker="x", ms=8)
    axs[1, 0].set_xlabel("step")
    axs[1, 0].set_ylabel("avg reward")
    axs[1, 0].set_title("Avg reward")
    axs[1, 0].axhline(0, color="k", linewidth=0.5)
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)

    # Avg elapsed
    if train:
        axs[1, 1].plot([r["step"] for r in train],
                       [r["avg_elapsed_s"] for r in train],
                       label="train", marker="o", ms=4)
    if eval_:
        axs[1, 1].plot([r["step"] for r in eval_],
                       [r["avg_elapsed_s"] for r in eval_],
                       label="eval", marker="x", ms=8)
    axs[1, 1].set_xlabel("step")
    axs[1, 1].set_ylabel("avg elapsed (s)")
    axs[1, 1].set_title("Time used per rollout")
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out = args.out or args.log.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
