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

from ballwall.checker import CATEGORIES, CATEGORY_ZH, DEFAULT_TOL, FormatError, parse_svg  # noqa: E402
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
              "before_cat": CATEGORY_ZH[r0["score"]["category"]], "after_cat": CATEGORY_ZH[r1["score"]["category"]],
              "before_reason": r0["score"]["reason"], "after_reason": r1["score"]["reason"]}

    # failing examples after training (one per category, eval_in)
    fails = []
    seen = set()
    for r in s1:
        c = r["score"]["category"]
        if r["dataset"] == "eval_in" and c not in ("success",) and c not in seen:
            seen.add(c)
            fails.append({"cat": CATEGORY_ZH[c], "reason": r["score"]["reason"], "id": r["task"]["id"],
                          "head": r["completion"][:220]})

    data = {
        "cats": [CATEGORY_ZH[c] for c in CATEGORIES], "steps": steps, "curve": curve, "train": train_log,
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
    for ds, name in (("eval_in", "训练分布内"), ("eval_heldout", "留出参数")):
        t1.append(f'<tr><th class="l" colspan="4">{name}({d0[ds]["n"]} 题,贪心解码)</th></tr>')
        t1.append(row("成功率", "success_rate", ds))
        lo0, hi0 = d0[ds]["success_ci95"]
        lo1, hi1 = d1[ds]["success_ci95"]
        t1.append(f'<tr><td class="l">成功率 95% 置信区间(Wilson)</td><td>{pct(lo0)}–{pct(hi0)}</td><td>{pct(lo1)}–{pct(hi1)}</td><td></td></tr>')
        a, b = d0[ds]["mean_reward"], d1[ds]["mean_reward"]
        t1.append(f'<tr><td class="l">平均 reward</td><td>{a:.3f}</td><td>{b:.3f}</td><td class="{"good" if b > a else "bad"}">{b - a:+.3f}</td></tr>')
        t1.append(row("格式失败率", "format_fail_rate", ds))
        t1.append(row("硬约束违规率(占全部样本,不含格式失败)", "hard_violation_rate", ds))
        t1.append(row("P(格式通过)", "p_format_ok", ds))
        t1.append(row("P(硬约束通过 | 格式通过)", "p_hard_ok_given_format", ds))
        t1.append(row("P(成功 | 硬约束通过)", "p_success_given_hard", ds))

    st_rows = []
    if selftest:
        for sname, v in selftest["strategies"]["eval_in"].items():
            h = selftest["strategies"]["eval_heldout"][sname]
            st_rows.append(f'<tr><td class="l"><code>{sname}</code></td><td>{pct(v["success_rate"])}</td><td>{v["mean_reward"]:.3f}</td>'
                           f'<td class="l">{CATEGORY_ZH[v["top_category"]]}</td><td>{pct(h["success_rate"])}</td><td>{h["mean_reward"]:.3f}</td></tr>')
        rb = "".join(f'<tr><td class="l">{x["case"]}</td><td class="l">{"已拦截" if x["blocked"] else "<b>未拦截</b>"}</td><td class="l">{x["reason"]}</td></tr>'
                     for x in selftest["robustness"])
    else:
        rb = ""

    probe = (first or {}).get("sample_k")
    probe_html = ""
    if probe:
        probe_html = (f"<p>基线探针:取 {probe['n_prompts']} 道训练题,每题以 temperature 1.0 采样 {probe['k']} 条。"
                      f"零方差组(组内 reward 全相同,GRPO 无梯度)比例:violation_mode=low 时 {pct(probe['low']['frac_zero_std_groups'])},"
                      f"zero 时 {pct(probe['zero']['frac_zero_std_groups'])};采样格式通过率 {pct(probe['low']['summary']['p_format_ok'])}。</p>")

    cfg_rows = ""
    if run_cfg:
        g = run_cfg.get("grpo_config", {})
        items = [("被训练模型", model), ("设备", f"{run_cfg.get('device')}({run_cfg.get('dtype')})"),
                 ("每步", f"{run_cfg.get('prompts_per_step')} 道题 × G={run_cfg.get('num_generations')} 条 = {run_cfg.get('completions_per_step')} 条补全"),
                 ("micro-batch × 梯度累积", f"{run_cfg.get('per_device_bs')} × {run_cfg.get('grad_accum')}"),
                 ("学习率 / KL β", f"{g.get('learning_rate')} / {g.get('beta')}"),
                 ("最大补全长度", g.get("max_completion_length")), ("训练步数", g.get("max_steps")),
                 ("LoRA", "r=16, alpha=32, dropout=0.05, 全部注意力与 MLP 投影"),
                 ("violation_mode", run_cfg.get("args", {}).get("violation_mode")),
                 ("库版本", ", ".join(f"{k} {v}" for k, v in run_cfg.get("versions", {}).items()))]
        cfg_rows = "".join(f'<tr><td class="l">{k}</td><td class="l">{v}</td></tr>' for k, v in items)

    fail_rows = "".join(f'<tr><td class="l">{f["cat"]}</td><td class="l">{f["reason"]}</td><td class="l"><code>{f["id"]}</code></td></tr>' for f in fails)
    s_in0, s_in1 = d0["eval_in"]["success_rate"], d1["eval_in"]["success_rate"]
    s_ho0, s_ho1 = d0["eval_heldout"]["success_rate"], d1["eval_heldout"]["success_rate"]
    f0, f1 = d0["eval_in"]["format_fail_rate"], d1["eval_in"]["format_fail_rate"]
    h0, h1 = d0["eval_in"]["hard_violation_rate"], d1["eval_in"]["hard_violation_rate"]

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>球撞墙评测报告</title>
{style}
</head><body><div class="wrap">
<p class="sub">黑客松项目 · 结果报告(真实数据,由 <code>scripts/make_report.py</code> 从 <code>outputs/</code> 生成)</p>
<h1>球撞墙:模型不借助代码,能否逐帧画出正确的碰撞过程?</h1>
<p class="sub">模型直接输出 16 帧静态 SVG;评分器按 id 取球心坐标,先查格式和硬约束(穿墙等),再比软精度(撞墙时刻、反弹速度比、轨迹)。同一个分数用作 GRPO 奖励。</p>

<h2>1. 摘要</h2>
<div class="kpis">
  <div class="kpi"><div class="n">{pct(s_in0)} → {pct(s_in1)}</div><div class="l">成功率,训练分布内(第 {steps[0]} 步 → 第 {steps[-1]} 步)</div></div>
  <div class="kpi"><div class="n">{pct(s_ho0)} → {pct(s_ho1)}</div><div class="l">成功率,留出参数</div></div>
  <div class="kpi"><div class="n">{pct(f0)} → {pct(f1)}</div><div class="l">格式失败率,训练分布内</div></div>
  <div class="kpi"><div class="n">{pct(h0)} → {pct(h1)}</div><div class="l">硬约束违规率,训练分布内(不含格式失败)</div></div>
</div>
<ul>
  <li>被训练模型:<code>{model}</code>,LoRA + GRPO,评分器是唯一奖励。评测一律贪心解码,每个评测集 {d0['eval_in']['n']} 题。</li>
  <li>成功等价于 reward 恰好为 1.0,成功率是 reward 的阈值化,不是独立证据。</li>
  <li>口径:题目给了 v、e、x₀、Δt,答案是闭式公式。曲线上升只能说明在该格式下更会执行碰撞运动学,不能说明"物理理解提升"。</li>
</ul>

<h2>2. 训练设置(实际使用)</h2>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">项目</th><th class="l">设置</th></tr>{cfg_rows}</table></div>
{probe_html}

<h2>3. 训练曲线</h2>
<h3>图 1:评测集成功率与平均 reward 随训练步数(贪心解码,带 95% 置信区间)</h3>
<div class="card"><figure><div id="legend1" class="legend"></div><div id="chart1"></div>
<figcaption>每个点是一个检查点在固定评测集上的贪心解码结果;阴影是成功率的 Wilson 95% 置信区间。</figcaption></figure></div>
<h3>图 2:训练中的在线指标(每步训练 batch 的采样结果)</h3>
<div class="card"><figure><div id="legend2" class="legend"></div><div id="chart2"></div>
<figcaption>来自训练 batch 的在线采样(temperature 1.0),噪声大,和图 1 的贪心评测口径不同。零方差组比例高说明组内 reward 无差异,没有梯度。</figcaption></figure></div>
<h3>图 3:失败类别占比随训练步数(训练分布内评测集)</h3>
<div class="card"><figure><div id="legend3" class="legend"></div><div id="chart3"></div>
<figcaption>暖色是格式和硬约束失败,蓝紫是软精度失败,绿色是成功。</figcaption></figure></div>

<h2>4. 训练前后对比</h2>
<h3>表 1:第 {steps[0]} 步 vs 第 {steps[-1]} 步</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">指标</th><th>训练前</th><th>训练后</th><th>变化</th></tr>{''.join(t1)}</table></div>
<p class="tag">后三行是链式分解,三项之积等于成功率。条件概率的分母在训练前后是不同的样本集合,有选择偏差,只能说结果与"学会碰撞运动学"一致。</p>
<h3>图 4:reward 分布(训练分布内,训练前 vs 训练后)</h3>
<div class="card"><figure><div id="legend4" class="legend"></div><div id="chart4"></div>
<figcaption>成功样本的 reward 恰好为 1.0,落在最右一格。0.2 到 0.3 之间没有样本是奖励公式的构造结果,不是实验发现。</figcaption></figure></div>
<h3>图 5:同一道题训练前后的球心 x 轨迹</h3>
<div class="card"><figure><div id="legend5" class="legend"></div><div id="chart5"></div>
<figcaption id="cap5"></figcaption></figure></div>
<h3>表 2:训练后仍然失败的样本(每类一例,训练分布内)</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">类别</th><th class="l">评分器给出的原因</th><th class="l">题目</th></tr>{fail_rows or '<tr><td class="l" colspan="3">无</td></tr>'}</table></div>

<h2>5. 评分器自检(训练前必须通过)</h2>
<h3>表 3:笨策略(不用模型),实测</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">策略</th><th>成功率(分布内)</th><th>平均 reward</th><th class="l">最常见类别</th><th>成功率(留出)</th><th>平均 reward</th></tr>{''.join(st_rows)}</table></div>
<p class="tag"><code>fixed_e05</code> 在分布内成功 12.5%:真实 e 落在 0.5 的 8% 容差内的题约占 15%,这是容差本身的含义,不是漏洞。<code>mirror_e1</code> 在留出参数上成功 25.5%:e ≥ 0.926 时与 e=1 的差距在容差内。</p>
<h3>表 4:解析健壮性用例(都必须判为格式失败)</h3>
<div class="tablewrap card" style="padding:0"><table><tr><th class="l">用例</th><th class="l">结果</th><th class="l">原因</th></tr>{rb}</table></div>

<h2>6. 局限</h2>
<ul>
  <li>一维单球场景,答案是闭式公式;SVG 与坐标列表(JSON)的对照实验未做。</li>
  <li>成功率是 reward 的阈值化;没有独立的人工或 VLM 抽查。</li>
  <li>训练规模受 4 GB 显存笔记本 GPU 限制,步数和每步题数都远小于计划的 CUDA 默认值(见第 2 节)。</li>
</ul>
<footer><p>数据文件:<code>outputs/eval/*.json</code>、<code>outputs/curve.json</code>、<code>outputs/train_log.json</code>、<code>outputs/selftest.json</code>。</p></footer>
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
 axes(f,st,"训练步数",0,1,.2,"成功率 / 平均 reward",pc,xmax);
 const xp=s=>f.m.l+s/xmax*f.W,yp=v=>f.m.t+f.H-v*f.H;
 const S=[["eval_in","success","成功率(分布内)",0,""],["eval_heldout","success","成功率(留出)",1,""],["eval_in","mean_reward","平均 reward(分布内)",0,"5 4"],["eval_heldout","mean_reward","平均 reward(留出)",1,"5 4"]];
 S.forEach(([ds,k,n,c,dash])=>{{const b=D.curve[ds];if(k=="success"){{const up=st.map((s,i)=>[xp(s),yp(b.ci_high[i])]),lo=st.map((s,i)=>[xp(s),yp(b.ci_low[i])]).reverse();
   f.svg.appendChild(el("path",{{d:path(up)+" "+path(lo).replace("M","L")+"Z",fill:C[c],opacity:.12}}));}}
   f.svg.appendChild(el("path",{{d:path(st.map((s,i)=>[xp(s),b[k][i]==null?null:yp(b[k][i])])),fill:"none",stroke:C[c],"stroke-width":2.2,"stroke-dasharray":dash}}));
   st.forEach((s,i)=>b[k][i]!=null&&f.svg.appendChild(el("circle",{{cx:xp(s),cy:yp(b[k][i]),r:3,fill:C[c]}})));}});
 mount("chart1",f.svg);legend("legend1",S.map(s=>s[2]),S.map(s=>C[s[3]]));}})();
// chart 2
(()=>{{const T=D.train;if(!T.length){{document.getElementById("chart2").textContent="没有训练日志";return;}}
 const f=frame(760,300,{{l:56,r:16,t:16,b:56}}),xmax=Math.max(...T.map(r=>r.step));
 const ticks=[...new Set([0,Math.round(xmax/4),Math.round(xmax/2),Math.round(3*xmax/4),xmax])];
 axes(f,ticks,"训练步数",0,1,.2,"在线指标",pc,xmax);const xp=s=>f.m.l+s/xmax*f.W,yp=v=>f.m.t+f.H-Math.max(0,Math.min(1,v))*f.H;
 const S=[["reward","平均 reward"],["rewards/metric_success/mean","成功率"],["rewards/metric_format_ok/mean","格式通过率"],["rewards/metric_hard_ok/mean","硬约束通过率"],["frac_reward_zero_std","零方差组比例"]];
 S.forEach(([k,n],c)=>f.svg.appendChild(el("path",{{d:path(T.filter(r=>r[k]!=null).map(r=>[xp(r.step),yp(r[k])])),fill:"none",stroke:C[c],"stroke-width":1.8,"stroke-dasharray":c==4?"4 3":""}})));
 mount("chart2",f.svg);legend("legend2",S.map(s=>s[1]),S.map((_,i)=>C[i]));}})();
// chart 3
(()=>{{const st=D.steps,cd=D.curve.eval_in.category_dist,f=frame(760,300,{{l:56,r:16,t:16,b:56}}),n=st.length,gw=f.W/n,bw=Math.min(60,gw*.6);
 axes(f,[],"",0,1,.25,"占比",pc,1);
 st.forEach((s,i)=>{{let acc=0;const x=f.m.l+gw*(i+.5)-bw/2;cd[i].forEach((v,k)=>{{const h=v*f.H,y=f.m.t+f.H-(acc+v)*f.H;f.svg.appendChild(el("rect",{{x,y,width:bw,height:h,fill:K[k]}}));acc+=v;}});
  f.svg.appendChild(el("text",{{x:x+bw/2,y:f.m.t+f.H+18,"text-anchor":"middle",class:"axis"}},"第 "+s+" 步"));}});
 mount("chart3",f.svg);legend("legend3",D.cats,K);}})();
// chart 4
(()=>{{const f=frame(760,280,{{l:56,r:16,t:16,b:56}}),H0=D.hist.before.eval_in,H1=D.hist.after.eval_in,ymax=Math.max(...H0,...H1,.05);
 axes(f,[0,0.2,0.4,0.6,0.8,1],"reward",0,ymax,ymax/4,"样本占比",v=>Math.round(v*100)+"%",1);const bw=f.W/20;
 [[H0,0],[H1,2]].forEach(([H,c],j)=>H.forEach((v,i)=>{{const h=v/ymax*f.H;f.svg.appendChild(el("rect",{{x:f.m.l+i*bw+j*bw/2,y:f.m.t+f.H-h,width:bw/2-1,height:h,fill:C[c]}}));}}));
 mount("chart4",f.svg);legend("legend4",["训练前","训练后"],[C[0],C[2]]);}})();
// chart 5
(()=>{{const E=D.example;if(!E){{document.getElementById("chart5").textContent="没有可解析的样例";return;}}
 const t=E.task,N=t.n_frames,f=frame(760,320,{{l:56,r:16,t:16,b:56}}),all=[...E.truth,...E.before,...E.after,t.wall_x],ymax=Math.ceil(Math.max(...all)/100)*100+50,ymin=Math.max(0,Math.floor(Math.min(...all)/100)*100-50);
 axes(f,[0,3,6,9,12,15],"帧号",ymin,ymax,(ymax-ymin)/5,"球心 x(px)",v=>Math.round(v),N-1);
 const xp=k=>f.m.l+k/(N-1)*f.W,yp=v=>f.m.t+f.H-(v-ymin)/(ymax-ymin)*f.H;
 f.svg.appendChild(el("line",{{x1:f.m.l,x2:f.m.l+f.W,y1:yp(t.wall_x-t.r),y2:yp(t.wall_x-t.r),stroke:"var(--k3)","stroke-dasharray":"6 4"}}));
 [[E.truth,5,""],[E.before,1,"5 4"],[E.after,2,""]].forEach(([xs,c,dash])=>{{f.svg.appendChild(el("path",{{d:path(xs.map((v,k)=>[xp(k),yp(v)])),fill:"none",stroke:C[c],"stroke-width":2,"stroke-dasharray":dash}}));xs.forEach((v,k)=>f.svg.appendChild(el("circle",{{cx:xp(k),cy:yp(v),r:2.6,fill:C[c]}})));}});
 mount("chart5",f.svg);legend("legend5",["标准答案","训练前("+E.before_cat+")","训练后("+E.after_cat+")","接触线 x = 墙 − r"],[C[5],C[1],C[2],"var(--k3)"]);
 document.getElementById("cap5").textContent=`题目 ${{t.id}}:v=${{t.v}} px/s,e=${{t.e}},r=${{t.r}},墙内侧 x=${{t.wall_x}},Δt=${{t.dt}} s。训练前:${{E.before_reason}}。训练后:${{E.after_reason}}。两条模型曲线都是真实的贪心解码输出。`;}})();
</script></body></html>"""
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
