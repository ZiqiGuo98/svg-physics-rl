"""Collect per-checkpoint eval results into outputs/curve.json and training logs into outputs/train_log.json
(IMPLEMENTATION_PLAN.md 4.5).

Usage: python scripts/collect_curve.py [--eval_dir outputs/eval] [--run_dir outputs/grpo/run1]
Only eval files whose name starts with --prefix (default "step") are used for the curve, so that one-off
evaluations (other models, probes) in the same folder do not end up on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballwall.checker import CATEGORIES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--prefix", default="step")
    ap.add_argument("--run_dir", default=str(ROOT / "outputs/grpo/run1"))
    ap.add_argument("--out", default=str(ROOT / "outputs/curve.json"))
    ap.add_argument("--train_log_out", default=str(ROOT / "outputs/train_log.json"))
    args = ap.parse_args()

    reports = []
    for p in sorted(Path(args.eval_dir).glob(f"{args.prefix}*.json")):
        if p.name.endswith(".samples.jsonl"):
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        if not rep.get("datasets"):
            continue
        try:
            step = int(rep["tag"])
        except (KeyError, ValueError):
            print(f"skip {p.name}: tag {rep.get('tag')!r} is not a step number")
            continue
        reports.append((step, p.name, rep))
    reports.sort()
    curve = {"steps": [s for s, _, _ in reports], "files": [n for _, n, _ in reports], "categories": CATEGORIES}
    for ds in ("eval_in", "eval_heldout"):
        block = {"success": [], "ci_low": [], "ci_high": [], "mean_reward": [], "format_fail": [], "category_dist": []}
        for _, _, rep in reports:
            s = rep["datasets"].get(ds)
            if s is None:
                for k in block:
                    block[k].append(None)
                continue
            block["success"].append(s["success_rate"])
            block["ci_low"].append(s["success_ci95"][0])
            block["ci_high"].append(s["success_ci95"][1])
            block["mean_reward"].append(s["mean_reward"])
            block["format_fail"].append(s["format_fail_rate"])
            block["category_dist"].append([s["category_dist"][c] for c in CATEGORIES])
        curve[ds] = block
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(curve, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: steps {curve['steps']}")
    for ds in ("eval_in", "eval_heldout"):
        print(f"  {ds} success: {curve[ds]['success']}")

    log = Path(args.run_dir) / "train_log.jsonl"
    if log.exists():
        keep = ("step", "wall_s", "reward", "reward_std", "frac_reward_zero_std", "kl", "loss", "entropy",
                "completions/mean_length", "completions/clipped_ratio",
                "rewards/physics_reward/mean", "rewards/metric_success/mean", "rewards/metric_format_ok/mean",
                "rewards/metric_hard_ok/mean")
        recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs = [{k: r[k] for k in keep if k in r} for r in recs if "reward" in r]
        Path(args.train_log_out).write_text(json.dumps(recs, indent=1), encoding="utf-8")
        print(f"wrote {args.train_log_out}: {len(recs)} logged steps")
    else:
        print(f"no training log at {log}")


if __name__ == "__main__":
    main()
