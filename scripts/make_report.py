"""Build report.html from real outputs (IMPLEMENTATION_PLAN.md section 9).

Reads outputs/selftest.json, outputs/eval/step*.json (+ .samples.jsonl), outputs/curve.json,
outputs/train_log.json and <run_dir>/run_config.json; reuses the <style> block of report_mock.html.
Usage: python scripts/make_report.py [--run_dir outputs/grpo/run1] [--out report.html]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballwall.checker import CATEGORIES, CATEGORY_LABEL, DEFAULT_TOL, FormatError, parse_svg  # noqa: E402
from ballwall.task import truth_xs  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def samples(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def pt(a, b):
    if a is None or b is None:
        return "—", ""
    d = (b - a) * 100
    return f"{d:+.1f} pt", ("good" if d > 0 else "bad" if d < 0 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(ROOT / "outputs/grpo/run1"))
    ap.add_argument("--eval_dir", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--out", default=str(ROOT / "report.html"))
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    ev = Path(args.eval_dir)

    selftest = load(out_dir / "selftest.json")
    curve = load(out_dir / "curve.json") or {"steps": []}
    train_log = load(out_dir / "train_log.json") or []
    run_cfg = load(Path(args.run_dir) / "run_config.json") or {}
    steps = curve.get("steps", [])
    files = curve.get("files", [])
    if not steps:
        sys.exit("no checkpoint evaluations in outputs/curve.json; run collect_curve.py first")
    first = load(ev / files[0])
    last = load(ev / files[-1])
    s0 = samples(ev / files[0].replace(".json", ".samples.jsonl"))
    s1 = samples(ev / files[-1].replace(".json", ".samples.jsonl"))

    # reward histograms (20 bins on [0, 1]) per dataset, before and after
    def hist(rows, ds):
        h = [0] * 20
        rs = [r["score"]["reward"] for r in rows if r["dataset"] == ds]
        for v in rs:
            h[min(19, int(v * 20))] += 1
        return [x / len(rs) if rs else 0 for x in h]

    # example trajectories: the eval_in task where the last checkpoint succeeds and the first did not, else any
    def xs_of(row):
        try:
            return parse_svg(row["completion"], row["task"], DEFAULT_TOL)[0]
        except FormatError:
            return None
    by_id0 = {r["task"]["id"]: r for r in s0 if r["dataset"] == "eval_in"}
    example = None
    for r1 in (r for r in s1 if r["dataset"] == "eval_in"):
        r0 = by_id0.get(r1["task"]["id"])
        if r0 and r1["score"]["success"] and not r0["score"]["success"] and xs_of(r0):
            example = (r0, r1)
            break
    if example is None:
        for r1 in (r for r in s1 if r["dataset"] == "eval_in"):
            r0 = by_id0.get(r1["task"]["id"])
            if r0 and xs_of(r1) and xs_of(r0):
                example = (r0, r1)
                break
    ex = None
    if example:
        r0, r1 = example
        t = r1["task"]
        ex = {"task": t, "truth": truth_xs(t), "before": xs_of(r0), "after": xs_of(r1),
              "before_cat": CATEGORY_LABEL[r0["score"]["category"]], "after_cat": CATEGORY_LABEL[r1["score"]["category"]],
              "before_reason": r0["score"]["reason"], "after_reason": r1["score"]["reason"]}

    # failing examples after training (one per category, eval_in)
    fails = []
    seen = set()
    for r in s1:
        c = r["score"]["category"]
        if r["dataset"] == "eval_in" and c not in ("success",) and c not in seen:
            seen.add(c)
            fails.append({"cat": CATEGORY_LABEL[c], "reason": r["score"]["reason"], "id": r["task"]["id"],
                          "head": r["completion"][:220]})

    data = {
        "cats": [CATEGORY_LABEL[c] for c in CATEGORIES], "steps": steps, "curve": curve, "train": train_log,
        "hist": {"before": {ds: hist(s0, ds) for ds in ("eval_in", "eval_heldout")},
                 "after": {ds: hist(s1, ds) for ds in ("eval_in", "eval_heldout")}},
        "example": ex,
    }

    style = re.search(r"<style>.*?</style>", (ROOT / "report_mock.html").read_text(encoding="utf-8"), re.S).group(0)
    model = run_cfg.get("args", {}).get("model") or (last or {}).get("model", "?")
    d0, d1 = first["datasets"], last["datasets"]

    def row(label, key, ds):
        a, b = d0[ds].get(key), d1[ds].get(key)
        delta, cls = pt(a, b)
        if key in ("format_fail_rate", "hard_violation_rate") and cls:
            cls = "good" if cls == "bad" else "bad"
        return f'<tr><td class="l">{label}</td><td>{pct(a)}</td><td>{pct(b)}</td><td class="{cls}">{delta}</td></tr>'

    t1 = []
    for ds, name in (("eval_in", "In-distribution"), ("eval_heldout", "Held-out parameters")):
        t1.append(f'<tr><th class="l" colspan="4">{name} ({d0[ds]["n"]} tasks, greedy decoding)</th></tr>')
        t1.append(row("Success rate", "success_rate", ds))
        lo0, hi0 = d0[ds]["success_ci95"]
        lo1, hi1 = d1[ds]["success_ci95"]
        t1.append(f'<tr><td class="l">Success rate 95% confidence interval (Wilson)</td><td>{pct(lo0)}–{pct(hi0)}</td><td>{pct(lo1)}–{pct(hi1)}</td><td></td></tr>')
        a, b = d0[ds]["mean_reward"], d1[ds]["mean_reward"]
        t1.append(f'<tr><td class="l">Mean reward</td><td>{a:.3f}</td><td>{b:.3f}</td><td class="{"good" if b > a else "bad"}">{b - a:+.3f}</td></tr>')
        t1.append(row("Format failure rate", "format_fail_rate", ds))
        t1.append(row("Hard-constraint violation rate (of all samples, excluding format failures)", "hard_violation_rate", ds))
        t1.append(row("P(format pass)", "p_format_ok", ds))
        t1.append(row("P(hard constraints pass | format pass)", "p_hard_ok_given_format", ds))
        t1.append(row("P(success | hard constraints pass)", "p_success_given_hard", ds))

    st_rows = []
    if selftest:
        for sname, v in selftest["strategies"]["eval_in"].items():
            h = selftest["strategies"]["eval_heldout"][sname]
            st_rows.append(f'<tr><td class="l"><code>{sname}</code></td><td>{pct(v["success_rate"])}</td><td>{v["mean_reward"]:.3f}</td>'
                           f'<td class="l">{CATEGORY_LABEL[v["top_category"]]}</td><td>{pct(h["success_rate"])}</td><td>{h["mean_reward"]:.3f}</td></tr>')
        rb = "".join(f'<tr><td class="l">{x["case"]}</td><td class="l">{"Blocked" if x["blocked"] else "<b>Not blocked</b>"}</td><td class="l">{x["reason"]}</td></tr>'
                     for x in selftest["robustness"])
    else:
        rb = ""

    probe = (first or {}).get("sample_k")
    probe_html = ""
    if probe:
        probe_html = (f"<p>Baseline probe: {probe['n_prompts']} training tasks, {probe['k']} samples per task at temperature 1.0. "
                      f"Share of zero-variance groups (all rewards in the group identical, no GRPO gradient): {pct(probe['low']['frac_zero_std_groups'])} with violation_mode=low, "
                      f"{pct(probe['zero']['frac_zero_std_groups'])} with zero; sampled format pass rate {pct(probe['low']['summary']['p_format_ok'])}.</p>")

    cfg_rows = ""
    if run_cfg:
        g = run_cfg.get("grpo_config", {})
        items = [("Trained model", model), ("Device", f"{run_cfg.get('device')} ({run_cfg.get('dtype')})"),
                 ("Per step", f"{run_cfg.get('prompts_per_step')} tasks × G={run_cfg.get('num_generations')} = {run_cfg.get('completions_per_step')} completions"),
                 ("micro-batch × gradient accumulation", f"{run_cfg.get('per_device_bs')} × {run_cfg.get('grad_accum')}"),
                 ("Learning rate / KL β", f"{g.get('learning_rate')} / {g.get('beta')}"),
                 ("Max completion length", g.get("max_completion_length")), ("Training steps", g.get("max_steps")),
                 ("LoRA", "r=16, alpha=32, dropout=0.05, all attention and MLP projections"),
                 ("violation_mode", run_cfg.get("args", {}).get("violation_mode")),
                 ("Library versions", ", ".join(f"{k} {v}" for k, v in run_cfg.get("versions", {}).items()))]
        cfg_rows = "".join(f'<tr><td class="l">{k}</td><td class="l">{v}</td></tr>' for k, v in items)

    fail_rows = "".join(f'<tr><td class="l">{f["cat"]}</td><td class="l">{f["reason"]}</td><td class="l"><code>{f["id"]}</code></td></tr>' for f in fails)
    s_in0, s_in1 = d0["eval_in"]["success_rate"], d1["eval_in"]["success_rate"]
    s_ho0, s_ho1 = d0["eval_heldout"]["success_rate"], d1["eval_heldout"]["success_rate"]
    f0, f1 = d0["eval_in"]["format_fail_rate"], d1["eval_in"]["format_fail_rate"]
    h0, h1 = d0["eval_in"]["hard_violation_rate"], d1["eval_in"]["hard_violation_rate"]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ball-Wall Collision Evaluation Report</title>
{style}
</head><body><div class="wrap">
<p class="sub">Hackathon project · Results report (real data, generated by <code>scripts/make_report.py</code> from <code>outputs/</code>)</p>
<h1>Ball-wall collision: can a model draw the correct collision frame by frame, without writing code?</h1>
<p class="sub">The model directly outputs 16 static SVG frames; the grader extracts the ball center coordinates by id, first checks format and hard constraints (wall penetration etc.), then compares soft accuracy (contact time, restitution ratio, trajectory). The same score is used as the GRPO reward.</p>

<h2>1. Summary</h2>
<div class="kpis">
  <div class="kpi"><div class="n">{pct(s_in0)} → {pct(s_in1)}</div><div class="l">Success rate, in-distribution (step {steps[0]} → step {steps[-1]})</div></div>
  <div class="kpi"><div class="n">{pct(s_ho0)} → {pct(s_ho1)}</div><div class="l">Success rate, held-out parameters</div></div>
  <div class="kpi"><div class="n">{pct(f0)} → {pct(f1)}</div><div class="l">Format failure rate, in-distribution</div></div>
  <div class="kpi"><div class="n">{pct(h0)} → {pct(h1)}</div><div class="l">Hard-constraint violation rate, in-distribution (excluding format failures)</div></div>
</div>
<ul>
  <li>Trained model: <code>{model}</code>, LoRA + GRPO, the grader is the only reward. All evaluation uses greedy decoding, {d0['eval_in']['n']} tasks per evaluation set.</li>
  <li>Success is equivalent to reward exactly 1.0; the success rate is a thresholding of reward, not independent evidence.</li>
  <li>Scope: the prompt gives v, e, x₀, Δt and the answer is a closed-form formula. A rising curve only shows the model executes collision kinematics better in this format; it does not show "improved physics understanding".</li>
</ul>

<h2>2. Training setup (as actually used)</h2>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">Item</th><th class="l">Setting</th></tr>{cfg_rows}</table></div>
{probe_html}

<h2>3. Training curves</h2>
<h3>Figure 1: Evaluation-set success rate and mean reward vs training step (greedy decoding, with 95% confidence interval)</h3>
<div class="card"><figure><div id="legend1" class="legend"></div><div id="chart1"></div>
<figcaption>Each point is one checkpoint's greedy-decoding result on the fixed evaluation set; the shaded band is the Wilson 95% confidence interval of the success rate.</figcaption></figure></div>
<h3>Figure 2: Online metrics during training (sampling results of each step's training batch)</h3>
<div class="card"><figure><div id="legend2" class="legend"></div><div id="chart2"></div>
<figcaption>From online sampling of the training batch (temperature 1.0); noisy, and measured differently from the greedy evaluation in Figure 1. A high share of zero-variance groups means rewards within a group do not differ, so there is no gradient.</figcaption></figure></div>
<h3>Figure 3: Failure category shares vs training step (in-distribution evaluation set)</h3>
<div class="card"><figure><div id="legend3" class="legend"></div><div id="chart3"></div>
<figcaption>Warm colors are format and hard-constraint failures, blue/purple are soft-accuracy failures, green is success.</figcaption></figure></div>

<h2>4. Before vs after training</h2>
<h3>Table 1: Step {steps[0]} vs step {steps[-1]}</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">Metric</th><th>Before training</th><th>After training</th><th>Change</th></tr>{''.join(t1)}</table></div>
<p class="tag">The last three rows are a chain decomposition; their product equals the success rate. The denominators of the conditional probabilities are different sample sets before and after training, so there is selection bias; one can only say the result is consistent with "learning collision kinematics".</p>
<h3>Figure 4: Reward distribution (in-distribution, before vs after training)</h3>
<div class="card"><figure><div id="legend4" class="legend"></div><div id="chart4"></div>
<figcaption>Successful samples have reward exactly 1.0 and fall in the rightmost bin. The absence of samples between 0.2 and 0.3 is a construction of the reward formula, not an experimental finding.</figcaption></figure></div>
<h3>Figure 5: Ball center x trajectory for the same task before and after training</h3>
<div class="card"><figure><div id="legend5" class="legend"></div><div id="chart5"></div>
<figcaption id="cap5"></figcaption></figure></div>
<h3>Table 2: Samples still failing after training (one per category, in-distribution)</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">Category</th><th class="l">Reason given by the grader</th><th class="l">Task</th></tr>{fail_rows or '<tr><td class="l" colspan="3">None</td></tr>'}</table></div>

<h2>5. Grader self-check (must pass before training)</h2>
<h3>Table 3: Dumb strategies (no model), measured</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">Strategy</th><th>Success rate (in-distribution)</th><th>Mean reward</th><th class="l">Most common category</th><th>Success rate (held-out)</th><th>Mean reward</th></tr>{''.join(st_rows)}</table></div>
<p class="tag"><code>fixed_e05</code> succeeds on 12.5% in-distribution: about 15% of tasks have a true e within the 8% tolerance of 0.5; this is what the tolerance means, not a loophole. <code>mirror_e1</code> succeeds on 25.5% of held-out parameters: for e ≥ 0.926 the gap to e=1 is within tolerance.</p>
<h3>Table 4: Parser robustness cases (all must be judged as format failure)</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">Case</th><th class="l">Result</th><th class="l">Reason</th></tr>{rb}</table></div>

<h2>6. Limitations</h2>
<ul>
  <li>One-dimensional single-ball scene, closed-form answer; the SVG vs coordinate-list (JSON) comparison was not run.</li>
  <li>The success rate is a thresholding of reward; no independent manual or VLM spot check.</li>
  <li>Training scale is limited by a 4 GB laptop GPU; steps and tasks per step are far below the planned CUDA defaults (see Section 2).</li>
</ul>
<footer><p>Data files: <code>outputs/eval/*.json</code>, <code>outputs/curve.json</code>, <code>outputs/train_log.json</code>, <code>outputs/selftest.json</code>.</p></footer>
</div>
<script>
const D = {json.dumps(data, ensure_ascii=False)};
const NS="http://www.w3.org/2000/svg";
function el(t,a,x){{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(x!=null)e.textContent=x;return e;}}
function legend(id,names,colors){{const b=document.getElementById(id);names.forEach((n,i)=>{{const s=document.createElement("span");s.innerHTML=`<i class="sw" style="background:${{colors[i]}}"></i>${{n}}`;b.appendChild(s);}});}}
function frame(w,h,m){{return {{svg:el("svg",{{viewBox:`0 0 ${{w}} ${{h}}`,role:"img"}}),W:w-m.l-m.r,H:h-m.t-m.b,m}};}}
function axes(f,xs,xlab,ymin,ymax,ystep,ylab,fmt,xmax){{const {{svg,W,H,m}}=f,g=el("g",{{class:"axis"}});svg.appendChild(g);
  for(let y=ymin;y<=ymax+1e-9;y+=ystep){{const py=m.t+H-(y-ymin)/(ymax-ymin)*H;g.appendChild(el("line",{{x1:m.l,x2:m.l+W,y1:py,y2:py,stroke:"var(--grid)"}}));g.appendChild(el("text",{{x:m.l-6,y:py+4,"text-anchor":"end"}},fmt(y)));}}
  xs.forEach(x=>{{const px=m.l+x/xmax*W;g.appendChild(el("text",{{x:px,y:m.t+H+18,"text-anchor":"middle"}},x));}});
  g.appendChild(el("text",{{x:m.l+W/2,y:m.t+H+40,"text-anchor":"middle"}},xlab));
  g.appendChild(el("text",{{x:14,y:m.t+H/2,"text-anchor":"middle",transform:`rotate(-90 14 ${{m.t+H/2}})`}},ylab));}}
const pc=v=>Math.round(v*100)+"%";
const C=["var(--c1)","var(--c2)","var(--c3)","var(--c4)","var(--c5)","var(--c6)"];
const K=D.cats.map((_,i)=>`var(--k${{i+1}})`);
const mount=(id,s)=>document.getElementById(id).appendChild(s);
function path(pts){{return pts.filter(p=>p[1]!=null).map((p,i)=>`${{i?"L":"M"}}${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(" ");}}
// chart 1
(()=>{{const f=frame(760,320,{{l:56,r:16,t:16,b:56}}),st=D.steps,xmax=Math.max(1,st[st.length-1]);
 axes(f,st,"Training step",0,1,.2,"Success rate / mean reward",pc,xmax);
 const xp=s=>f.m.l+s/xmax*f.W,yp=v=>f.m.t+f.H-v*f.H;
 const S=[["eval_in","success","Success rate (in-distribution)",0,""],["eval_heldout","success","Success rate (held-out)",1,""],["eval_in","mean_reward","Mean reward (in-distribution)",0,"5 4"],["eval_heldout","mean_reward","Mean reward (held-out)",1,"5 4"]];
 S.forEach(([ds,k,n,c,dash])=>{{const b=D.curve[ds];if(k=="success"){{const up=st.map((s,i)=>[xp(s),yp(b.ci_high[i])]),lo=st.map((s,i)=>[xp(s),yp(b.ci_low[i])]).reverse();
   f.svg.appendChild(el("path",{{d:path(up)+" "+path(lo).replace("M","L")+"Z",fill:C[c],opacity:.12}}));}}
   f.svg.appendChild(el("path",{{d:path(st.map((s,i)=>[xp(s),b[k][i]==null?null:yp(b[k][i])])),fill:"none",stroke:C[c],"stroke-width":2.2,"stroke-dasharray":dash}}));
   st.forEach((s,i)=>b[k][i]!=null&&f.svg.appendChild(el("circle",{{cx:xp(s),cy:yp(b[k][i]),r:3,fill:C[c]}})));}});
 mount("chart1",f.svg);legend("legend1",S.map(s=>s[2]),S.map(s=>C[s[3]]));}})();
// chart 2
(()=>{{const T=D.train;if(!T.length){{document.getElementById("chart2").textContent="No training log";return;}}
 const f=frame(760,300,{{l:56,r:16,t:16,b:56}}),xmax=Math.max(...T.map(r=>r.step));
 const ticks=[...new Set([0,Math.round(xmax/4),Math.round(xmax/2),Math.round(3*xmax/4),xmax])];
 axes(f,ticks,"Training step",0,1,.2,"Online metrics",pc,xmax);const xp=s=>f.m.l+s/xmax*f.W,yp=v=>f.m.t+f.H-Math.max(0,Math.min(1,v))*f.H;
 const S=[["reward","Mean reward"],["rewards/metric_success/mean","Success rate"],["rewards/metric_format_ok/mean","Format pass rate"],["rewards/metric_hard_ok/mean","Hard-constraint pass rate"],["frac_reward_zero_std","Zero-variance group share"]];
 S.forEach(([k,n],c)=>f.svg.appendChild(el("path",{{d:path(T.filter(r=>r[k]!=null).map(r=>[xp(r.step),yp(r[k])])),fill:"none",stroke:C[c],"stroke-width":1.8,"stroke-dasharray":c==4?"4 3":""}})));
 mount("chart2",f.svg);legend("legend2",S.map(s=>s[1]),S.map((_,i)=>C[i]));}})();
// chart 3
(()=>{{const st=D.steps,cd=D.curve.eval_in.category_dist,f=frame(760,300,{{l:56,r:16,t:16,b:56}}),n=st.length,gw=f.W/n,bw=Math.min(60,gw*.6);
 axes(f,[],"",0,1,.25,"Share",pc,1);
 st.forEach((s,i)=>{{let acc=0;const x=f.m.l+gw*(i+.5)-bw/2;cd[i].forEach((v,k)=>{{const h=v*f.H,y=f.m.t+f.H-(acc+v)*f.H;f.svg.appendChild(el("rect",{{x,y,width:bw,height:h,fill:K[k]}}));acc+=v;}});
  f.svg.appendChild(el("text",{{x:x+bw/2,y:f.m.t+f.H+18,"text-anchor":"middle",class:"axis"}},"Step "+s));}});
 mount("chart3",f.svg);legend("legend3",D.cats,K);}})();
// chart 4
(()=>{{const f=frame(760,280,{{l:56,r:16,t:16,b:56}}),H0=D.hist.before.eval_in,H1=D.hist.after.eval_in,ymax=Math.max(...H0,...H1,.05);
 axes(f,[0,0.2,0.4,0.6,0.8,1],"reward",0,ymax,ymax/4,"Share of samples",v=>Math.round(v*100)+"%",1);const bw=f.W/20;
 [[H0,0],[H1,2]].forEach(([H,c],j)=>H.forEach((v,i)=>{{const h=v/ymax*f.H;f.svg.appendChild(el("rect",{{x:f.m.l+i*bw+j*bw/2,y:f.m.t+f.H-h,width:bw/2-1,height:h,fill:C[c]}}));}}));
 mount("chart4",f.svg);legend("legend4",["Before training","After training"],[C[0],C[2]]);}})();
// chart 5
(()=>{{const E=D.example;if(!E){{document.getElementById("chart5").textContent="No parsable example";return;}}
 const t=E.task,N=t.n_frames,f=frame(760,320,{{l:56,r:16,t:16,b:56}}),all=[...E.truth,...E.before,...E.after,t.wall_x],ymax=Math.ceil(Math.max(...all)/100)*100+50,ymin=Math.max(0,Math.floor(Math.min(...all)/100)*100-50);
 axes(f,[0,3,6,9,12,15],"Frame index",ymin,ymax,(ymax-ymin)/5,"Ball center x (px)",v=>Math.round(v),N-1);
 const xp=k=>f.m.l+k/(N-1)*f.W,yp=v=>f.m.t+f.H-(v-ymin)/(ymax-ymin)*f.H;
 f.svg.appendChild(el("line",{{x1:f.m.l,x2:f.m.l+f.W,y1:yp(t.wall_x-t.r),y2:yp(t.wall_x-t.r),stroke:"var(--k3)","stroke-dasharray":"6 4"}}));
 [[E.truth,5,""],[E.before,1,"5 4"],[E.after,2,""]].forEach(([xs,c,dash])=>{{f.svg.appendChild(el("path",{{d:path(xs.map((v,k)=>[xp(k),yp(v)])),fill:"none",stroke:C[c],"stroke-width":2,"stroke-dasharray":dash}}));xs.forEach((v,k)=>f.svg.appendChild(el("circle",{{cx:xp(k),cy:yp(v),r:2.6,fill:C[c]}})));}});
 mount("chart5",f.svg);legend("legend5",["Ground truth","Before training ("+E.before_cat+")","After training ("+E.after_cat+")","Contact line x = wall − r"],[C[5],C[1],C[2],"var(--k3)"]);
 document.getElementById("cap5").textContent=`Task ${{t.id}}: v=${{t.v}} px/s, e=${{t.e}}, r=${{t.r}}, wall inner face x=${{t.wall_x}}, Δt=${{t.dt}} s. Before training: ${{E.before_reason}}. After training: ${{E.after_reason}}. Both model curves are real greedy-decoding outputs.`;}})();
</script></body></html>"""
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
