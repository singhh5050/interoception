"""Pull complete metric history from wandb for every sweep cell — into analysis/data/<name>.json.

Uses run.history(pandas=False, samples=N) which is one fast API call per run instead
of per-metric scan_history (which was ~30s × 19 keys × 8 runs = 60 min).
"""
from __future__ import annotations
import json
import pathlib
import sys
import wandb

PROJECT = "singhh5050-stanford-university/interoception"

CELLS = [
    "qwen25-3b-hyp-s0", "qwen25-3b-hyp-s1", "qwen25-3b-exp-s0", "qwen25-3b-exp-s1",
    "qwen3-4b-hyp-s0",  "qwen3-4b-hyp-s1",  "qwen3-4b-exp-s0",  "qwen3-4b-exp-s1",
]

# Hardcode the run IDs for cells where wandb dedup picks the wrong record.
# These two completed cleanly (ok=True in modal log) but wandb didn't update
# the run state to "finished" — multiple sibling crash records confuse the picker.
KNOWN_RUN_IDS = {
    "qwen25-3b-hyp-s1": "f0b966c732ec41dcabb811b387be0d37",
    "qwen25-3b-exp-s1": "9be10fe2a6d54a0aad94110d27adb371",
}


def pick_run(api: wandb.Api, name: str):
    """Pick the canonical run for a cell.

    Several wandb runs share each cell name (sweep_002 crash, sweep_003 crash,
    phase1_v1 crash, phase1_v2 finish). The "finished" state isn't reliable —
    some runs that completed cleanly in modal still show as "crashed" because
    the wandb.finish() handshake didn't complete on shutdown. And summary._step
    is None for all crashed runs.

    Strategy:
    1. If KNOWN_RUN_IDS overrides this cell, use that ID directly.
    2. Otherwise prefer a finished run (state-based).
    3. Otherwise probe history per candidate to find which has the most data.
    """
    if name in KNOWN_RUN_IDS:
        try:
            return api.run(f"{PROJECT}/{KNOWN_RUN_IDS[name]}")
        except Exception:
            pass  # fall through

    all_runs = list(api.runs(PROJECT, filters={"display_name": name}))
    if not all_runs:
        return None
    finished = [r for r in all_runs if r.state == "finished"]
    if finished:
        return finished[0]
    # Probe history for each: pick the run with most rows logged
    best, best_n = None, -1
    for r in all_runs:
        try:
            n = sum(1 for _ in r.scan_history(keys=["_step"], page_size=5000))
        except Exception:
            n = 0
        if n > best_n:
            best, best_n = r, n
    return best


def pull_run(r) -> dict:
    """One fast history() call instead of N scan_history()s. Returns dict keyed by metric."""
    # Pull as list of dicts via history(pandas=False); each row has all metrics logged at that step
    rows = list(r.history(pandas=False, samples=2000))
    metrics: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        step = row.get("_step")
        if step is None:
            continue
        for k, v in row.items():
            if k == "_step" or v is None:
                continue
            if not isinstance(v, (int, float)):
                continue
            metrics.setdefault(k, []).append((step, float(v)))
    return metrics


def main():
    api = wandb.Api()
    out = pathlib.Path("analysis/data")
    out.mkdir(parents=True, exist_ok=True)

    for name in CELLS:
        r = pick_run(api, name)
        if r is None:
            print(f"  {name:24s} -> NONE FOUND")
            continue
        try:
            metrics = pull_run(r)
        except Exception as e:
            print(f"  {name:24s} -> ERROR: {e}")
            continue
        n_keys = len(metrics)
        n_steps = max(len(v) for v in metrics.values()) if metrics else 0
        with open(out / f"{name}.json", "w") as f:
            json.dump({
                "name": name,
                "run_id": r.id,
                "state": r.state,
                "n_keys": n_keys,
                "n_steps": n_steps,
                "metrics": metrics,
            }, f)
        last_rwd = (metrics.get("reward/all/mean") or [(None, None)])[-1][1]
        last_t120 = (metrics.get("eval/eval-t120/avg@1") or [(None, None)])[-1][1]
        print(f"  {name:24s}  {r.state:10s}  {n_keys} keys, {n_steps} steps  "
              f"last_train_rwd={last_rwd if last_rwd is None else round(last_rwd, 3)}  "
              f"last_t120_avg@1={last_t120 if last_t120 is None else round(last_t120, 3)}")


if __name__ == "__main__":
    main()
