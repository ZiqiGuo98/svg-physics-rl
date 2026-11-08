"""Parse a model's SVG answer and score it (IMPLEMENTATION_PLAN.md section 3.2). Standard library only.

score(text, task) -> dict with reward, success, category (one of CATEGORIES), format_ok, hard_ok, reason,
soft, components, diag. Wall position and ball radius are ALWAYS taken from the task parameters; the
values drawn in the SVG only have to agree with them within a pixel.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from .task import contact_time, frame_times, truth_xs

DEFAULT_TOL: Dict[str, float] = {
    "wall_px": 1.0,      # SVG wall x vs task wall_x
    "r_px": 1.0,         # SVG radius vs task r
    "y_px": 2.0,         # allowed drift of the ball's y
    "pen_px": 3.0,       # penetration tolerance
    "teleport": 1.5,     # max frame-to-frame displacement = teleport * v * dt
    "static_frac": 0.25, # x range below this fraction of the truth range -> "static"
    "reach": 0.5,        # max x below truth max by more than reach * v * dt -> "did not reach the wall"
    "energy": 1.05,      # v_out > energy * v_in -> "energy gain"
    "t_c": 0.30,         # contact-time error tolerance, in units of dt
    "e_rel": 0.08,       # relative error tolerance of v_out / v_in vs e
    "v_rel": 0.05,       # relative error tolerance of v_in vs v
    "resid": 0.10,       # fit residual RMS tolerance, in units of v * dt
}

CATEGORIES: List[str] = [
    "format",
    "teleport_out_static",
    "penetration",
    "no_reach_or_no_rebound",
    "energy_gain",
    "contact_time",
    "restitution",
    "trajectory",
    "success",
]
CATEGORY_LABEL = {
    "format": "Format failure",
    "teleport_out_static": "Teleport / out of bounds / static",
    "penetration": "Wall penetration",
    "no_reach_or_no_rebound": "Did not reach wall or no rebound",
    "energy_gain": "Energy gain",
    "contact_time": "Contact-time error",
    "restitution": "Restitution error",
    "trajectory": "Trajectory error",
    "success": "Success",
}

FORBIDDEN_TAGS = {"animate", "animatemotion", "animatetransform", "set", "script", "use", "foreignobject", "style"}
_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*")
_SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S)
_NUM_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")
_FRAME_ID_RE = re.compile(r"^f\d+$")


class FormatError(Exception):
    pass


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _plain_number(s: Optional[str], what: str) -> float:
    if s is None or not _NUM_RE.match(s):
        raise FormatError(f"{what} is not a plain number: {s!r}")
    return float(s)


def _is_ball(el) -> bool:
    return _local(el.tag) == "circle" and "ball" in (el.get("class") or "").split()


def parse_svg(text: str, task: dict, tol: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """Return (xs, ys) of the ball centre per frame, or raise FormatError."""
    if not isinstance(text, str):
        raise FormatError("completion is not text")
    m = _SVG_RE.search(_FENCE_RE.sub("", text))
    if not m:
        raise FormatError("no <svg>...</svg> element")
    try:
        root = ET.fromstring(m.group(0))
    except ET.ParseError as ex:
        raise FormatError(f"XML parse error: {ex}")
    elems = list(root.iter())
    for el in elems:
        tag = _local(el.tag)
        if tag in FORBIDDEN_TAGS:
            raise FormatError(f"forbidden element <{tag}>")
        if any(_local(k) == "transform" for k in el.attrib):
            raise FormatError(f"transform attribute on <{tag}>")

    walls = [el for el in elems if el.get("id") == "wall"]
    if len(walls) != 1:
        raise FormatError(f"expected exactly one id=wall element, found {len(walls)}")
    if _local(walls[0].tag) != "rect":
        raise FormatError("id=wall is not a <rect>")
    wall_x_svg = _plain_number(walls[0].get("x"), "wall x")
    if abs(wall_x_svg - task["wall_x"]) > tol["wall_px"]:
        raise FormatError(f"wall drawn at x={wall_x_svg}, task says {task['wall_x']}")

    n = task["n_frames"]
    frames: Dict[str, object] = {}
    for el in elems:
        fid = el.get("id")
        if _local(el.tag) == "g" and fid and _FRAME_ID_RE.match(fid):
            if fid in frames:
                raise FormatError(f"duplicate frame id {fid}")
            frames[fid] = el
    expected = {f"f{k}" for k in range(n)}
    if set(frames) != expected:
        missing = sorted(expected - set(frames), key=lambda s: int(s[1:]))
        extra = sorted(set(frames) - expected, key=lambda s: int(s[1:]))
        raise FormatError(f"frame ids wrong: missing {missing[:5]}, extra {extra[:5]}")

    total_balls = sum(1 for el in elems if _is_ball(el))
    if total_balls != n:
        raise FormatError(f"{total_balls} ball circles in the SVG, expected {n}")

    xs, ys = [], []
    for k in range(n):
        balls = [el for el in frames[f"f{k}"].iter() if _is_ball(el)]
        if len(balls) != 1:
            raise FormatError(f"frame f{k} has {len(balls)} ball circles")
        b = balls[0]
        cx = _plain_number(b.get("cx"), f"f{k} cx")
        cy = _plain_number(b.get("cy"), f"f{k} cy")
        rr = _plain_number(b.get("r"), f"f{k} r")
        if abs(rr - task["r"]) > tol["r_px"]:
            raise FormatError(f"f{k} radius {rr}, task says {task['r']}")
        xs.append(cx)
        ys.append(cy)
    return xs, ys


def _fit(ts: List[float], xs: List[float]) -> Tuple[float, float, float]:
    """Least-squares line x = a + b t; returns (a, b, residual RMS)."""
    n = len(ts)
    mt = sum(ts) / n
    mx = sum(xs) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    b = sum((t - mt) * (x - mx) for t, x in zip(ts, xs)) / sxx
    a = mx - b * mt
    rms = math.sqrt(sum((x - (a + b * t)) ** 2 for t, x in zip(ts, xs)) / n)
    return a, b, rms


def _result(**kw) -> dict:
    base = {
        "reward": 0.0, "success": False, "category": "format", "format_ok": False, "hard_ok": False,
        "reason": "", "soft": None, "components": None, "diag": None,
    }
    base.update(kw)
    return base


def score(text: str, task: dict, violation_mode: str = "low", tol: Optional[Dict[str, float]] = None) -> dict:
    if violation_mode not in ("low", "zero"):
        raise ValueError(f"violation_mode must be 'low' or 'zero', got {violation_mode!r}")
    T = dict(DEFAULT_TOL)
    if tol:
        T.update(tol)

    # Step 1: parse
    try:
        xs, ys = parse_svg(text, task, T)
    except FormatError as ex:
        return _result(category="format", reason=str(ex))

    n, dt, v, r = task["n_frames"], task["dt"], task["v"], task["r"]
    wall_x, e, y0 = task["wall_x"], task["e"], task["y"]
    vdt = v * dt
    ts = frame_times(task)
    truth = truth_xs(task)
    tc = contact_time(task)

    k_post = next((k for k, t in enumerate(ts) if t > tc), None)
    diag = {
        "tc": tc,
        "traj_err_mean": sum(abs(x - g) for x, g in zip(xs, truth)) / n / vdt,
        "first_post_err": abs(xs[k_post] - truth[k_post]) / vdt if k_post is not None else None,
        "v_in": None, "v_out": None, "e_model": None, "t_cross": None,
    }

    # Two-segment fit around the model's own peak frame (the peak frame itself is excluded).
    kp = max(range(n), key=lambda k: xs[k])
    fit = None
    if 2 <= kp <= n - 3:
        a1, b1, rms1 = _fit(ts[:kp], xs[:kp])
        a2, b2, rms2 = _fit(ts[kp + 1:], xs[kp + 1:])
        fit = (a1, b1, rms1, a2, b2, rms2)

    components, soft, comp_scores = None, None, None
    tols = {"contact_time": T["t_c"], "restitution": T["e_rel"], "v_in": T["v_rel"], "resid": T["resid"]}
    if fit is not None:
        a1, b1, rms1, a2, b2, rms2 = fit
        v_in, v_out = b1, -b2
        diag["v_in"], diag["v_out"] = v_in, v_out
        if v_in > 0 and b1 != b2:
            t_cross = (a2 - a1) / (b1 - b2)
            diag["t_cross"] = t_cross
            diag["e_model"] = v_out / v_in
            components = {
                "contact_time": abs(t_cross - tc) / dt,
                "restitution": abs(v_out / v_in - e) / e,
                "v_in": abs(v_in - v) / v,
                "resid": max(rms1, rms2) / vdt,
            }
            comp_scores = {
                k: 1.0 if components[k] <= tols[k] else math.exp(-(components[k] - tols[k]) / tols[k])
                for k in components
            }
            soft = math.exp(sum(math.log(max(s, 1e-300)) for s in comp_scores.values()) / len(comp_scores))

    # Step 2: hard constraints, first hit wins
    category, reason = None, ""
    steps = [abs(xs[k + 1] - xs[k]) for k in range(n - 1)]
    if max(abs(y - y0) for y in ys) > T["y_px"]:
        category, reason = "teleport_out_static", f"y drifts by {max(abs(y - y0) for y in ys):.1f} px"
    elif min(xs) - r < -T["pen_px"]:
        category, reason = "teleport_out_static", f"ball leaves the canvas on the left (min x {min(xs):.1f})"
    elif max(xs) - min(xs) < T["static_frac"] * (max(truth) - min(truth)):
        category, reason = "teleport_out_static", f"ball barely moves (range {max(xs) - min(xs):.1f} px)"
    elif max(steps) > T["teleport"] * vdt:
        category, reason = "teleport_out_static", f"teleport: frame step {max(steps):.1f} px > {T['teleport']} v dt"
    elif max(xs) + r > wall_x + T["pen_px"]:
        category, reason = "penetration", f"ball surface at {max(xs) + r:.1f}, wall face at {wall_x}"
    elif max(xs) < max(truth) - T["reach"] * vdt:
        category, reason = "no_reach_or_no_rebound", f"turns before the wall (max x {max(xs):.1f}, truth {max(truth):.1f})"
    elif fit is None:
        category, reason = "no_reach_or_no_rebound", f"peak at frame {kp}: no usable incoming or rebound segment"
    elif fit[1] <= 0 or -fit[4] <= 0:
        category, reason = "no_reach_or_no_rebound", f"no reversal (v_in {fit[1]:.1f}, v_out {-fit[4]:.1f})"
    elif -fit[4] > T["energy"] * fit[1]:
        category, reason = "energy_gain", f"rebound speed {-fit[4]:.1f} > {T['energy']} x incoming {fit[1]:.1f}"

    if category is not None:
        reward = 0.0 if violation_mode == "zero" else 0.2 * (soft or 0.0)
        return _result(reward=reward, category=category, format_ok=True, hard_ok=False, reason=reason,
                       soft=soft, components=components, diag=diag)

    # Step 3/4: soft accuracy (fit exists and v_in > 0 here, so soft is defined)
    over = {k: components[k] > tols[k] for k in components}
    if not any(over.values()):
        return _result(reward=1.0, success=True, category="success", format_ok=True, hard_ok=True,
                       reason="ok", soft=soft, components=components, diag=diag)
    if over["contact_time"]:
        category = "contact_time"
        reason = f"contact at {diag['t_cross']:.3f} s vs {tc:.3f} s ({components['contact_time']:.2f} dt)"
    elif over["restitution"]:
        category = "restitution"
        reason = f"rebound ratio {diag['e_model']:.3f} vs e={e}"
    else:
        category = "trajectory"
        reason = f"v_in error {components['v_in']:.3f}, residual {components['resid']:.3f} v dt"
    return _result(reward=0.3 + 0.7 * soft, category=category, format_ok=True, hard_ok=True, reason=reason,
                   soft=soft, components=components, diag=diag)
