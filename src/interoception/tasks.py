"""Small set of Countdown-style problems for exploratory rollouts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountdownProblem:
    numbers: tuple[int, ...]
    target: int

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
