"""Aggregate checker results (IMPLEMENTATION_PLAN.md 3.4). Standard library only."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .checker import CATEGORIES


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _ratio(a: int, b: int) -> Optional[float]:
    return a / b if b else None


def summarize(results: List[dict]) -> dict:
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_format = sum(1 for r in results if r["format_ok"])
    n_hard = sum(1 for r in results if r["hard_ok"])
    lo, hi = wilson(n_success, n)
    dist = {c: (sum(1 for r in results if r["category"] == c) / n if n else 0.0) for c in CATEGORIES}
    return {
        "n": n,
        "success_rate": _ratio(n_success, n),
        "success_ci95": [lo, hi],
        "mean_reward": (sum(r["reward"] for r in results) / n) if n else None,
        "format_fail_rate": _ratio(n - n_format, n),
        "hard_violation_rate": _ratio(n_format - n_hard, n),
        "category_dist": dist,
        "p_format_ok": _ratio(n_format, n),
        "p_hard_ok_given_format": _ratio(n_hard, n_format),
        "p_success_given_hard": _ratio(n_success, n_hard),
    }
