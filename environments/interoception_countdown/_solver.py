"""Countdown solver: enumerate all expression trees over a multiset of numbers.

For 4-number Countdown, every solution uses exactly 3 binary ops (4 → 3 → 2 → 1).
We enumerate by recursive pair-combine: pick any two values, apply each of
+, -, *, /, recurse on the resulting multiset.

Difficulty signal: `solution_count` — how many (path, op-assignment) tuples
reach the target. Commutative ops double-count by design (we don't canonicalize
across `a+b` and `b+a`), which is fine for relative difficulty ranking — what
matters is that problems with many ways to reach the target rank higher than
problems with few. Lower count = harder for a model to find any solution.

Float arithmetic with tolerance is used (TinyZero-style — fractional
intermediates allowed). This matches the convention in the Jiayi-Pan dataset.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Sequence

OPS: tuple[tuple[str, callable], ...] = (
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / b),
)

_TOL = 1e-9


@dataclass(frozen=True)
class SolveResult:
    solution_count: int
    example_solution: str | None
    has_integer_only_solution: bool


def _enumerate(items: list[tuple[float, str, bool]]) -> Iterable[tuple[float, str, bool]]:
    """Yield (value, expr, int_only) for every reachable combination.

    int_only is True iff every intermediate value in this expression is an
    integer (within tolerance). The Countdown game-show rules require this;
    most RL setups (Jiayi Pan / TinyZero) do not.
    """
    if len(items) == 1:
        yield items[0]
        return
    n = len(items)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a_val, a_str, a_int = items[i]
            b_val, b_str, b_int = items[j]
            rest = [items[k] for k in range(n) if k != i and k != j]
            for op_sym, op_fn in OPS:
                if op_sym == "/" and abs(b_val) < _TOL:
                    continue
                new_val = op_fn(a_val, b_val)
                new_int = a_int and b_int and abs(new_val - round(new_val)) < _TOL
                new_str = f"({a_str} {op_sym} {b_str})"
                yield from _enumerate(rest + [(new_val, new_str, new_int)])


def solve(nums: tuple[int, ...] | list[int], target: int) -> SolveResult:
    items = [(float(n), str(n), True) for n in nums]
    target_f = float(target)
    count = 0
    example: str | None = None
    has_int = False
    for value, expr, int_only in _enumerate(items):
        if abs(value - target_f) < _TOL:
            count += 1
            if example is None:
                example = expr
            if int_only:
                has_int = True
    return SolveResult(solution_count=count, example_solution=example, has_integer_only_solution=has_int)


_VALIDATOR_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def validate_solution(expr_str: str | None, nums: Sequence[int], target: int) -> bool | None:
    """Check a candidate solution string against the Countdown rules.

    Returns:
        True  — expression uses each number in `nums` exactly once (as a
                multiset of integer literals) AND evaluates to `target`.
        False — expression parses cleanly but fails either check.
        None  — expression is missing or unparseable (model failure to produce
                a valid expression; distinct from "produced one but wrong").
    """
    if expr_str is None:
        return None
    expr = expr_str.strip().strip("`").strip()
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    literals: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, _VALIDATOR_ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            if isinstance(v, float) and not v.is_integer():
                return None
            literals.append(int(v))

    if sorted(literals) != sorted(nums):
        return False

    try:
        value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(value, (int, float)):
        return None
    return abs(value - float(target)) < _TOL
