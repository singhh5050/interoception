"""Countdown problem definitions and dataset loading.

`PROBLEMS` is a tiny hardcoded demo set used by the original exploration
runs. For training and the full sweeps you should build a stratified
dataset with `scripts/build_dataset.py` and load it via `load_problems`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CountdownProblem:
    numbers: tuple[int, ...]
    target: int
    # Optional metadata populated by the dataset builder. Hand-written PROBLEMS
    # below leave these at default; loaded JSONL records fill them in.
    solution_count: int = 0
    example_solution: str | None = None

    def to_prompt(self) -> str:
        return (
            f"Using the numbers {list(self.numbers)} and the operators +, -, *, / "
            f"(each number used exactly once), find an expression that equals {self.target}. "
            "Show your reasoning, then put the final expression inside <answer>...</answer>."
        )


PROBLEMS = [
    CountdownProblem(numbers=(2, 3, 4, 5), target=24),
    CountdownProblem(numbers=(1, 5, 6, 7), target=21),
    CountdownProblem(numbers=(3, 5, 7, 8), target=24),
    CountdownProblem(numbers=(2, 4, 6, 9), target=24),
]


def load_problems(path: str | Path) -> list[CountdownProblem]:
    """Load a JSONL file produced by scripts/build_dataset.py."""
    path = Path(path)
    problems: list[CountdownProblem] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            problems.append(CountdownProblem(
                numbers=tuple(rec["nums"]),
                target=rec["target"],
                solution_count=rec.get("solution_count", 0),
                example_solution=rec.get("example_solution"),
            ))
    return problems
