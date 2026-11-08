"""Checker self-test; must pass before any RL (IMPLEMENTATION_PLAN.md 4.2).

1. oracle answer -> success with reward 1.0 on every task of every data file
2. all dumb strategies on eval_in and eval_heldout -> table (strategy, success rate, mean reward, top category)
3. assertions: violating strategies never succeed; their mean reward < every good strategy's;
   slight_e and jitter succeed >= 80%
4. warning if mirror_e1 or fixed_e05 succeed > 10% on eval_in
5. parser robustness cases must all be format failures
6. table written to outputs/selftest.md
Exit code 0 only if everything passes.
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballwall.baselines import GOOD, PRINT_ONLY, STRATEGIES, VIOLATING  # noqa: E402
from ballwall.checker import CATEGORY_ZH, score  # noqa: E402
from ballwall.metrics import summarize  # noqa: E402
from ballwall.task import read_jsonl, render_svg, truth_xs  # noqa: E402

FAILURES = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def robustness_cases(task: dict) -> dict:
    good = render_svg(task, truth_xs(task))
    n = task["n_frames"]
    cases = {}
    cases["missing frame"] = good.replace(f'<g id="f{n - 1}">', '<g id="x">')
    cases["duplicate frame id"] = good.replace('<g id="f3">', '<g id="f2">')
    first_cx = good.split('cx="', 1)[1].split('"', 1)[0]
    cases["cx with unit (880px)"] = good.replace(f'cx="{first_cx}"', 'cx="880px"', 1)
    cases["transform attribute"] = good.replace('<g id="f0">', '<g id="f0" transform="translate(1,0)">')
    cases["<animate> element"] = good.replace("</svg>", '<animate attributeName="cx" from="0" to="1" dur="1s"/></svg>')
    cases["wall x differs from task"] = good.replace(f'x="{task["wall_x"]}"', f'x="{task["wall_x"] + 50}"', 1)
    r = task["r"]
    cases["radius differs from task"] = good.replace(f'r="{r}"/>', f'r="{r + 5}"/>', 1)
    cases["extra ball outside frames"] = good.replace("</svg>", f'<circle class="ball" cx="10" cy="500" r="{r}"/></svg>')
    cases["no svg in output"] = "The ball moves right, hits the wall and bounces back."
    cases["scientific notation"] = good.replace(f'cx="{first_cx}"', 'cx="1e2"', 1)
    cases["svg truncated (no </svg>)"] = good[: len(good) // 2]
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "selftest.md"))
    ap.add_argument("--violation_mode", default="low", choices=["low", "zero"])
    args = ap.parse_args()
    data = Path(args.data)
    files = {"train": data / "train.jsonl", "eval_in": data / "eval_in.jsonl", "eval_heldout": data / "eval_heldout.jsonl"}
    sets = {k: [row["task"] for row in read_jsonl(p)] for k, p in files.items()}

    # 1. oracle on every task
    print("== 1. oracle answer on every task")
    for name, tasks in sets.items():
        bad = []
        for t in tasks:
            res = score(render_svg(t, truth_xs(t)), t, args.violation_mode)
            if not (res["success"] and res["reward"] == 1.0):
                bad.append((t, res))
        check(not bad, f"oracle succeeds with reward 1.0 on all {len(tasks)} {name} tasks")
        for t, res in bad[:5]:
            print("   task:", t)
            print("   result:", res["category"], res["reason"], res["components"])

    # 2. strategies table
    print("\n== 2. dumb strategies")
    md = ["# 评分器自检结果", "", f"violation_mode = `{args.violation_mode}`", ""]
    table = {}
    js = {"violation_mode": args.violation_mode, "strategies": {}, "robustness": [], "failures": FAILURES}
    for name in ("eval_in", "eval_heldout"):
        tasks = sets[name]
        md += [f"## {name}({len(tasks)} 题)", "", "| 策略 | 成功率 | 平均 reward | 最常见类别 | 硬约束通过率 |", "|---|---:|---:|---|---:|"]
        print(f"\n{name}: {'strategy':18s} {'success':>8s} {'reward':>8s}  {'hard_ok':>7s}  top category")
        for sname, fn in STRATEGIES.items():
            rng = random.Random(12345)
            results = [score(render_svg(t, fn(t, rng)), t, args.violation_mode) for t in tasks]
            s = summarize(results)
            top, cnt = collections.Counter(r["category"] for r in results).most_common(1)[0]
            table[(name, sname)] = (s, results)
            js["strategies"].setdefault(name, {})[sname] = {
                "success_rate": s["success_rate"], "mean_reward": s["mean_reward"], "top_category": top,
                "top_count": cnt, "hard_ok_rate": 1 - s["format_fail_rate"] - s["hard_violation_rate"],
                "category_dist": s["category_dist"]}
            print(f"{'':10s}{sname:18s} {s['success_rate']:8.1%} {s['mean_reward']:8.3f}  {1 - s['format_fail_rate'] - s['hard_violation_rate']:7.1%}  {top} ({cnt})")
            md.append(f"| `{sname}` | {s['success_rate']:.1%} | {s['mean_reward']:.3f} | {CATEGORY_ZH[top]}({cnt}/{len(tasks)}) | "
                      f"{1 - s['format_fail_rate'] - s['hard_violation_rate']:.1%} |")
        md.append("")

    # 3. assertions
    print("\n== 3. assertions")
    for name in ("eval_in", "eval_heldout"):
        for sname in VIOLATING:
            check(table[(name, sname)][0]["success_rate"] == 0.0, f"{name}: {sname} never succeeds")
        worst_good = min(table[(name, g)][0]["mean_reward"] for g in GOOD)
        best_bad = max(table[(name, b)][0]["mean_reward"] for b in VIOLATING)
        check(best_bad < worst_good, f"{name}: best violating mean reward {best_bad:.3f} < worst good mean reward {worst_good:.3f}")
        for sname in ("slight_e", "jitter"):
            sr = table[(name, sname)][0]["success_rate"]
            check(sr >= 0.80, f"{name}: {sname} success {sr:.1%} >= 80%")
        # lexicographic order holds sample by sample
        all_res = [r for (nm, _), (_, rs) in table.items() if nm == name for r in rs]
        hard_ok_min = min((r["reward"] for r in all_res if r["hard_ok"]), default=1.0)
        viol_max = max((r["reward"] for r in all_res if r["format_ok"] and not r["hard_ok"]), default=0.0)
        check(viol_max <= 0.2 < 0.3 <= hard_ok_min, f"{name}: every violating sample <= 0.2 < 0.3 <= every hard-ok sample "
              f"(max violating {viol_max:.3f}, min hard-ok {hard_ok_min:.3f})")

    # 4. warnings
    print("\n== 4. loose-tolerance warnings")
    for sname in PRINT_ONLY:
        sr = table[("eval_in", sname)][0]["success_rate"]
        if sr > 0.10:
            print(f"WARN  eval_in: {sname} succeeds {sr:.1%} > 10%: tolerance may be too loose")
        else:
            print(f"ok    eval_in: {sname} succeeds {sr:.1%}")

    # 5. parser robustness
    print("\n== 5. parser robustness (each must be a format failure)")
    t0 = sets["eval_in"][0]
    md += ["## 解析健壮性用例(都必须判为格式失败)", "", "| 用例 | 结果 | 原因 |", "|---|---|---|"]
    for cname, text in robustness_cases(t0).items():
        res = score(text, t0, args.violation_mode)
        ok = res["category"] == "format" and res["reward"] == 0.0
        check(ok, f"{cname}: {res['category']} ({res['reason']})")
        js["robustness"].append({"case": cname, "blocked": ok, "reason": res["reason"]})
        md.append(f"| {cname} | {'格式失败' if ok else '**未拦截:' + res['category'] + '**'} | {res['reason'].replace('|', '/')} |")
    # sanity: the untouched answer, fenced in markdown, still parses
    fenced = "```svg\n" + render_svg(t0, truth_xs(t0)) + "\n```"
    check(score(fenced, t0)["success"], "oracle answer wrapped in a ```svg fence still succeeds")

    md += ["", f"自检结论:{'全部通过' if not FAILURES else '有 %d 项未通过' % len(FAILURES)}", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    import json
    out.with_suffix(".json").write_text(json.dumps(js, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    if FAILURES:
        print(f"\nSELFTEST FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
