"""Build a held-out test set from the same Countdown source, guaranteed disjoint
from existing data/train.jsonl and data/eval.jsonl.

Pipeline:
  1. Load existing train.jsonl + eval.jsonl as the "forbidden" set (matched on (nums tuple, target)).
  2. Re-fetch the parquet, solve fresh rows, drop any that overlap with forbidden.
  3. Stratified-sample 500 fresh examples into data/test.jsonl using a NEW seed.

Usage:
    python scripts/build_test_set.py --seed 999 --size 500 --out data/test.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# Reuse the same source + solver as build_dataset.py
sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import (
    PARQUET_URL, DEFAULT_CACHE,
    download_parquet, load_4num_rows, bucket_key,
)


def load_forbidden(paths: list[Path]) -> set[tuple[tuple[int, ...], int]]:
    """Tuples of (nums tuple, target) that are off-limits — already in train or eval."""
    forbidden = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                forbidden.add((tuple(r["nums"]), int(r["target"])))
    return forbidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=999,
                    help="Different from train/eval (0). Don't reuse a published value.")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--max-solutions", type=int, default=50)
    ap.add_argument("--solve-limit", type=int, default=60_000)
    ap.add_argument("--exclude", type=Path, nargs="*",
                    default=[Path("data/train.jsonl"), Path("data/eval.jsonl")])
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    sys.path.insert(0, str(Path(__file__).parent.parent / "environments" / "interoception_countdown"))
    from _solver import solve

    forbidden = load_forbidden(args.exclude)
    print(f"forbidden set: {len(forbidden):,} examples (from {[str(p) for p in args.exclude]})")

    parquet_path = download_parquet(args.cache)
    rows = load_4num_rows(parquet_path)
    print(f"4-num pool: {len(rows):,}")

    # IMPORTANT: shuffle with the new seed BEFORE filtering by forbidden.
    # This way we sample over the full pool, not a contiguous slice.
    rng.shuffle(rows)
    fresh_rows = [(n, t) for (n, t) in rows if (n, t) not in forbidden]
    print(f"after dropping forbidden: {len(fresh_rows):,} ({len(rows) - len(fresh_rows):,} excluded)")

    fresh_rows = fresh_rows[: args.solve_limit]
    print(f"solving first {len(fresh_rows):,} fresh rows...")

    buckets: dict[str, list[dict]] = defaultdict(list)
    unsolvable = trivial = 0
    t0 = time.time()
    for i, (nums, target) in enumerate(fresh_rows, 1):
        r = solve(nums, target)
        if r.solution_count == 0:
            unsolvable += 1
            continue
        if r.solution_count > args.max_solutions:
            trivial += 1
            continue
        buckets[bucket_key(r.solution_count, r.has_integer_only_solution)].append({
            "nums": list(nums),
            "target": target,
            "solution_count": r.solution_count,
            "example_solution": r.example_solution,
            "has_integer_only_solution": r.has_integer_only_solution,
        })
        if i % 5000 == 0:
            print(f"  {i:6,}/{len(fresh_rows):,}  "
                  f"({time.time()-t0:.0f}s)  unsolvable={unsolvable} trivial={trivial}")

    bucket_names = sorted(buckets)
    n_per_bucket = args.size // len(bucket_names)
    test_set = []
    for k in bucket_names:
        items = buckets[k][:]
        rng.shuffle(items)
        take = min(n_per_bucket, len(items))
        test_set.extend(items[:take])
        if take < n_per_bucket:
            print(f"  WARNING bucket {k} short: have {len(items)}, wanted {n_per_bucket}")

    rng.shuffle(test_set)
    args.out.write_text("\n".join(json.dumps(r) for r in test_set) + "\n")
    print(f"\nwrote {len(test_set):,} → {args.out}")

    # Sanity check
    test_keys = set((tuple(r["nums"]), int(r["target"])) for r in test_set)
    overlap = test_keys & forbidden
    print(f"final overlap with forbidden: {len(overlap)} (must be 0)")
    assert len(overlap) == 0


if __name__ == "__main__":
    main()
