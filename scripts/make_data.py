"""Generate data/train.jsonl, data/eval_in.jsonl and data/eval_heldout.jsonl (IMPLEMENTATION_PLAN.md 4.1).

Usage: python scripts/make_data.py --seed 0 --n_train 2000 --n_eval 200 --out data/
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballwall.task import contact_time, sample_task, write_jsonl  # noqa: E402


def describe(name: str, tasks: list) -> None:
    def rng_of(key):
        vals = [t[key] for t in tasks]
        return f"{min(vals)}..{max(vals)}"
    fcs = [contact_time(t) / t["dt"] for t in tasks]
    fracs = [fc - math.floor(fc) for fc in fcs]
    dts = sorted({t["dt"] for t in tasks})
    print(f"{name:14s} n={len(tasks):5d}  dt={dts}  e={rng_of('e')}  v={rng_of('v')}  r={rng_of('r')}  "
          f"x0={rng_of('x0')}  wall_x={rng_of('wall_x')}  contact frame={min(fcs):.2f}..{max(fcs):.2f}  "
          f"frac={min(fracs):.2f}..{max(fracs):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [
        ("train", "train", args.n_train, 1),
        ("eval_in", "eval_in", args.n_eval, 2),
        ("eval_heldout", "heldout", args.n_eval, 3),
    ]
    for fname, split, n, offset in plan:
        rng = random.Random(args.seed * 1000 + offset)  # a different seed per set
        tasks = []
        for i in range(n):
            t = sample_task(rng, split)
            t["id"] = f"{fname}-{i:05d}"
            tasks.append(t)
        write_jsonl(out / f"{fname}.jsonl", tasks)
        describe(fname, tasks)
    print("wrote", ", ".join(str(out / f"{p[0]}.jsonl") for p in plan))


if __name__ == "__main__":
    main()
