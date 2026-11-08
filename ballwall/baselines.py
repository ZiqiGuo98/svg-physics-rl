"""Dumb strategies and hand-written wrong answers for the checker self-test (IMPLEMENTATION_PLAN.md 3.3).

Each strategy is f(task, rng) -> xs (ball-centre x per frame); render with task.render_svg.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List

from .task import contact_time, contact_x, frame_times, truth_xs, x_at


def oracle(task: dict, rng: random.Random) -> List[float]:
    return truth_xs(task)


def slight_e(task: dict, rng: random.Random) -> List[float]:
    return [x_at(task, t, e=task["e"] * 1.04) for t in frame_times(task)]


def jitter(task: dict, rng: random.Random) -> List[float]:
    a = 0.03 * task["v"] * task["dt"]
    return [x + rng.uniform(-a, a) for x in truth_xs(task)]


def mirror_e1(task: dict, rng: random.Random) -> List[float]:
    return [x_at(task, t, e=1.0) for t in frame_times(task)]


def fixed_e05(task: dict, rng: random.Random) -> List[float]:
    return [x_at(task, t, e=0.5) for t in frame_times(task)]


def early_turn(task: dict, rng: random.Random) -> List[float]:
    """Turn around after 60% of the distance to the contact point, then rebound with the right e."""
    x_turn = task["x0"] + 0.6 * (contact_x(task) - task["x0"])
    t_turn = 0.6 * contact_time(task)
    v, e = task["v"], task["e"]
    return [task["x0"] + v * t if t <= t_turn else x_turn - e * v * (t - t_turn) for t in frame_times(task)]


def small_penetration(task: dict, rng: random.Random) -> List[float]:
    """Correct answer, except the frame closest to the wall pokes the ball 5 px into it."""
    xs = truth_xs(task)
    k = max(range(len(xs)), key=lambda i: xs[i])
    xs[k] = contact_x(task) + 5.0
    return xs


def pass_through(task: dict, rng: random.Random) -> List[float]:
    return [task["x0"] + task["v"] * t for t in frame_times(task)]


def static(task: dict, rng: random.Random) -> List[float]:
    return [float(task["x0"])] * task["n_frames"]


def random_x(task: dict, rng: random.Random) -> List[float]:
    r = task["r"]
    return [rng.uniform(r, task["canvas"] - r) for _ in range(task["n_frames"])]


STRATEGIES: Dict[str, Callable[[dict, random.Random], List[float]]] = {
    "oracle": oracle,
    "slight_e": slight_e,
    "jitter": jitter,
    "mirror_e1": mirror_e1,
    "fixed_e05": fixed_e05,
    "early_turn": early_turn,
    "small_penetration": small_penetration,
    "pass_through": pass_through,
    "static": static,
    "random": random_x,
}
GOOD = ("oracle", "slight_e", "jitter")
VIOLATING = ("early_turn", "small_penetration", "pass_through", "static", "random")
PRINT_ONLY = ("mirror_e1", "fixed_e05")
