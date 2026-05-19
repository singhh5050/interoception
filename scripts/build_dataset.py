"""Build a stratified Countdown dataset from Jiayi-Pan/Countdown-Tasks-3to4.

Pipeline:
  1. Download the single parquet file (no `datasets` lib required).
  2. Filter to 4-number problems.
  3. Solve each with our solver to get `solution_count` and an example.
  4. Drop unsolvable (count == 0).
  5. Drop trivial: anything with solution_count above a configurable threshold
     (default 50) — these problems have so many solutions the model can stumble
     into one without thinking.
  6. Bucket by solution_count (small / med / large) and integer-only flag.
  7. Sample uniformly across buckets for train and eval.

Usage:
    python scripts/build_dataset.py --out-dir data/ --train-size 10000 --eval-size 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# This is the only fixed external URL the script depends on. If the dataset
# is ever moved or renamed, update this single line.
PARQUET_URL = "https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4/resolve/main/data/train-00000-of-00001.parquet"

# Local cache so re-runs don't re-download.
DEFAULT_CACHE = Path.home() / ".cache" / "interoception" / "countdown-3to4.parquet"


def download_parquet(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading {PARQUET_URL} → {dest}")
    urllib.request.urlretrieve(PARQUET_URL, dest)
    return dest


def load_4num_rows(parquet_path: Path) -> list[tuple[tuple[int, ...], int]]:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    rows = []
    for _, row in df.iterrows():
        nums = tuple(int(x) for x in row["nums"])
        if len(nums) == 4:
            rows.append((nums, int(row["target"])))
    return rows


def bucket_key(solution_count: int, has_int: bool) -> str:
    """Bucket on solution count.

    Bands chosen so each is meaningful for an RL training mix:
      - "rare"   (1-3 solutions):    model must search to find one
      - "med"    (4-12 solutions):   moderate; a few good paths
      - "common" (13-50 solutions):  many paths; should be fast wins

    Anything above max_solutions is dropped (trivial). The Jiayi-Pan dataset
    is pre-filtered to integer-solvable problems, so has_int is True for ~all
    rows and not useful for stratification — we ignore it here.
    """
    if solution_count <= 3:
        return "rare"
    if solution_count <= 12:
        return "med"
    return "common"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--train-size", type=int, default=10_000)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--max-solutions", type=int, default=50,
                   help="drop problems with more solutions than this (too easy)")
    p.add_argument("--solve-limit", type=int, default=60_000,
                   help="solve at most this many 4-num problems (keeps wall time bounded)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Import solver after argparse so --help is fast. The solver lives inside
    # the env package as `_solver.py`; this script reaches in directly rather
    # than depending on the env package install.
    sys.path.insert(
        0, str(Path(__file__).parent.parent / "environments" / "interoception_countdown")
    )
    from _solver import solve

    parquet_path = download_parquet(args.cache)
    print(f"loading 4-num rows from {parquet_path}")
    rows = load_4num_rows(parquet_path)
    print(f"  {len(rows):,} 4-num problems available")

    rng.shuffle(rows)
    rows = rows[: args.solve_limit]
    print(f"solving first {len(rows):,} (cap = --solve-limit)")

    buckets: dict[str, list[dict]] = defaultdict(list)
    unsolvable = 0
    trivial = 0
    t0 = time.time()

    for i, (nums, target) in enumerate(rows, 1):
        r = solve(nums, target)
        if r.solution_count == 0:
            unsolvable += 1
            continue
        if r.solution_count > args.max_solutions:
            trivial += 1
            continue
        rec = {
            "nums": list(nums),
            "target": target,
            "solution_count": r.solution_count,
            "example_solution": r.example_solution,
            "has_integer_only_solution": r.has_integer_only_solution,
        }
        buckets[bucket_key(r.solution_count, r.has_integer_only_solution)].append(rec)
        if i % 5000 == 0:
            print(f"  {i:6,}/{len(rows):,}  ({time.time()-t0:.0f}s)  "
                  f"unsolvable={unsolvable} trivial={trivial}  "
                  f"buckets={ {k: len(v) for k, v in buckets.items()} }")

    print(f"\nsolved {len(rows):,} in {time.time()-t0:.0f}s "
          f"({unsolvable} unsolvable, {trivial} trivial-dropped)")
    print("per-bucket counts:")
    for k in sorted(buckets):
        print(f"  {k:>12}  {len(buckets[k])}")

    # Stratified sample: uniform-per-bucket, with replacement only if a bucket is too small.
    bucket_names = sorted(buckets)
    n_per_bucket_train = args.train_size // len(bucket_names)
    n_per_bucket_eval = args.eval_size // len(bucket_names)
    train, eval_set = [], []
    for k in bucket_names:
        items = buckets[k][:]
        rng.shuffle(items)
        need_train = min(n_per_bucket_train, len(items))
        need_eval = min(n_per_bucket_eval, len(items) - need_train)
        eval_chunk = items[:need_eval]
        train_chunk = items[need_eval : need_eval + need_train]
        train.extend(train_chunk)
        eval_set.extend(eval_chunk)
        if need_train < n_per_bucket_train or need_eval < n_per_bucket_eval:
            print(f"  WARNING bucket {k} short: have {len(items)}, wanted {n_per_bucket_train}+{n_per_bucket_eval}")

    rng.shuffle(train)
    rng.shuffle(eval_set)

    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train) + "\n")
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_set) + "\n")
    print(f"\nwrote {len(train):,} → {train_path}")
    print(f"wrote {len(eval_set):,} → {eval_path}")


if __name__ == "__main__":
    main()
