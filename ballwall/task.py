"""Task generation, ground truth, prompts, SVG rendering and jsonl IO for the ball-wall scene.

Standard library only (IMPLEMENTATION_PLAN.md section 3.1). Units are pixels and seconds; motion is 1-D.
Canvas 1000 x 1000, x to the right, y downward, ball centre y fixed at 500. The wall's inner (left) face
is at wall_x. Contact happens when the ball's rightmost point reaches the face: centre x = wall_x - r.
"""
from __future__ import annotations

import json
import math
import random
from typing import List, Optional

CANVAS = 1000
N_FRAMES = 16
BALL_Y = 500

SPLIT_DT = {
    "train": (0.05, 0.1, 0.2),
    "eval_in": (0.05, 0.1, 0.2),
    "heldout": (0.15, 0.25),
}
SPLIT_E = {
    "train": (0.30, 0.85),
    "eval_in": (0.30, 0.85),
    "heldout": (0.86, 0.95),
}

SYSTEM_PROMPT = """You draw physical processes as SVG animation frames. You must NOT write code. Output exactly one <svg> element and nothing else (no explanation, no markdown).

Format rules:
- Start with <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">.
- Draw static objects once, outside the frames, with exactly the ids and coordinates given in the task.
- Then one <g id="fK"> per frame, K = 0, 1, ..., N-1, in order. Each frame contains exactly one <circle class="ball" cx="..." cy="..." r="..."/>.
- Coordinates are plain numbers in pixels (decimals allowed). No units, no transform, no animation, no script.
- x increases to the right, y increases downward.

Example of the format (a different scene: a ball rolling left at a constant 200 px/s, no wall, frame interval 0.5 s, 4 frames):
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
<g id="f0"><circle class="ball" cx="800" cy="300" r="15"/></g>
<g id="f1"><circle class="ball" cx="700" cy="300" r="15"/></g>
<g id="f2"><circle class="ball" cx="600" cy="300" r="15"/></g>
<g id="f3"><circle class="ball" cx="500" cy="300" r="15"/></g>
</svg>"""

USER_TEMPLATE = """Scene: a ball moves horizontally to the right, collides with a vertical wall, and bounces back.
- Wall: its left (inner) face is at x = {wall_x}. Draw it as <rect id="wall" x="{wall_x}" y="0" width="{wall_w}" height="1000"/>.
- Ball: radius {r} px. Its center starts at x = {x0}, y = {y}. Initial velocity: {v} px/s to the right. No gravity, no friction.
- The ball touches the wall when its rightmost point reaches the wall's inner face. The collision has coefficient of restitution e = {e}.
- Frame K shows the scene at time t = K * {dt} s, for K = 0..{n_last} ({n} frames).

Output the {n} frames as one SVG."""


def contact_x(task: dict) -> float:
    """Ball-centre x at contact."""
    return task["wall_x"] - task["r"]


def contact_time(task: dict) -> float:
    return (contact_x(task) - task["x0"]) / task["v"]


def x_at(task: dict, t: float, e: Optional[float] = None) -> float:
    """Ground-truth centre x at time t; `e` overrides the restitution (used by the dumb baselines)."""
    e = task["e"] if e is None else e
    tc = contact_time(task)
    if t <= tc:
        return task["x0"] + task["v"] * t
    return contact_x(task) - e * task["v"] * (t - tc)


def frame_times(task: dict) -> List[float]:
    return [k * task["dt"] for k in range(task["n_frames"])]


def truth_xs(task: dict) -> List[float]:
    return [x_at(task, t) for t in frame_times(task)]


def sample_task(rng: random.Random, split: str, n_frames: int = N_FRAMES) -> dict:
    """Rejection-sample one task (IMPLEMENTATION_PLAN.md 3.1, constraints 1-3)."""
    lo_e, hi_e = SPLIT_E[split]
    for _ in range(100000):
        r = rng.randint(12, 30)
        x0 = rng.randint(60, 220)
        wall_x = rng.randint(700, 950)
        e = round(rng.uniform(lo_e, hi_e), 2)
        dt = rng.choice(SPLIT_DT[split])
        fc_target = rng.uniform(4.3, n_frames - 5.3)
        v = int(round((wall_x - r - x0) / (fc_target * dt)))
        if v <= 0:
            continue
        task = {
            "split": split, "n_frames": n_frames, "dt": dt, "x0": x0, "y": BALL_Y, "r": r,
            "wall_x": wall_x, "e": e, "v": v, "canvas": CANVAS,
        }
        fc = contact_time(task) / dt  # recomputed with the rounded v
        if not (4.3 <= fc <= n_frames - 5.3):
            continue
        frac = fc - math.floor(fc)
        if not (0.1 <= frac <= 0.9):
            continue
        if truth_xs(task)[-1] < r + 10:
            continue
        return task
    raise RuntimeError(f"could not sample a valid task for split={split}")


def _num(v: float) -> str:
    """Plain decimal for prompts and SVG: at most 2 decimals, no trailing zeros, no exponent."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def build_messages(task: dict) -> List[dict]:
    n = task["n_frames"]
    user = USER_TEMPLATE.format(
        wall_x=task["wall_x"], wall_w=task["canvas"] - task["wall_x"], r=task["r"], x0=task["x0"], y=task["y"],
        v=task["v"], e=_num(task["e"]), dt=_num(task["dt"]), n_last=n - 1, n=n,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def render_svg(task: dict, xs: List[float], ys: Optional[List[float]] = None, r: Optional[float] = None) -> str:
    """Render a list of ball-centre x positions as a format-compliant SVG answer."""
    ys = [task["y"]] * len(xs) if ys is None else ys
    r = task["r"] if r is None else r
    w = task["wall_x"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">',
        f'<rect id="wall" x="{w}" y="0" width="{task["canvas"] - w}" height="1000"/>',
    ]
    for k, (x, y) in enumerate(zip(xs, ys)):
        lines.append(f'<g id="f{k}"><circle class="ball" cx="{_num(x)}" cy="{_num(y)}" r="{_num(r)}"/></g>')
    lines.append("</svg>")
    return "\n".join(lines)


def write_jsonl(path, tasks: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for task in tasks:
            f.write(json.dumps({"task": task, "prompt": build_messages(task)}, ensure_ascii=False) + "\n")


def read_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
