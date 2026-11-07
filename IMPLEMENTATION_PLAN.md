# 实现方案:球撞墙 SVG 评分器 + TRL GRPO 训练

> 读者:负责实现的 coding agent 和队友。
> 背景和设计理由见 `plan_physics_svg_frames.md`;本文只写**要实现什么、怎么实现、怎么验收**。两份文档冲突时,以本文为准。
> 本文中的数值(容差、超参数)是起始值,允许在自检后调整,但调整要写进 `NOTES.md`。

---

## 0. 目标与范围

**要交付:**
1. 题目生成器:随机生成"球撞墙"题目、标准答案、提示文本。
2. 评分器:解析模型输出的 SVG,判定格式、硬约束、软精度,给出 reward、是否成功、失败类别。
3. 自检:标准答案、手写错误答案、笨策略,证明评分器正确且有区分度。
4. 评测脚本:对基座模型或 LoRA 检查点做贪心解码评测,输出成功率、置信区间、失败类别占比。
5. 训练脚本:TRL `GRPOTrainer` + LoRA,用评分器作为奖励。
6. 曲线汇总:把多个检查点的评测结果汇总成一份 JSON,供报告使用。

**不做:**
- 加重力和地面的"变体"场景。
- SVG 与坐标列表(JSON)的对照实验(已记为局限)。
- verl、多机训练。

**硬性要求:**
- `ballwall/` 下的核心模块(生成器、评分器、笨策略、指标)**只用 Python 标准库**,兼容 Python 3.9+,这样在任何机器上都能跑自检。
- 训练和评测脚本可以依赖 torch / transformers / trl / peft,要求**设备无关**:自动选择 CUDA、MPS(Mac)或 CPU。

---

## 1. 目录结构

```
svg-physics-rl/
├── ballwall/
│   ├── __init__.py
│   ├── task.py         # 题目生成、标准答案、提示、渲染、读写 jsonl
│   ├── checker.py      # 解析 + 打分(核心)
│   ├── baselines.py    # 笨策略和手写错误答案
│   └── metrics.py      # 汇总统计
├── scripts/
│   ├── make_data.py    # 生成数据集
│   ├── selftest.py     # 评分器自检(RL 之前必须通过)
│   ├── evaluate.py     # 模型评测
│   ├── train_grpo.py   # GRPO 训练
│   └── collect_curve.py# 汇总各检查点结果
├── data/               # 生成的 jsonl(提交到仓库,保证所有人用同一份题)
├── outputs/            # 检查点和评测结果(已在 .gitignore)
├── requirements.txt
├── NOTES.md            # 实现过程中的偏离、调参记录
└── IMPLEMENTATION_PLAN.md
```

脚本开头把仓库根目录加入 `sys.path`,保证 `python scripts/xxx.py` 能直接 `import ballwall`。

---

## 2. 环境

`requirements.txt`(训练和评测才需要;版本按实际能装上的为准,装好后在 `NOTES.md` 记录实际版本):

```
torch
transformers
trl
peft
datasets
accelerate
# 可选,仅 CUDA:vllm
```

- Python 用 3.10 或 3.11,新建虚拟环境(uv 或 conda)。系统自带的 3.9 可能装不上新版 TRL。
- Mac:没有 CUDA,不能用 vLLM 和 bitsandbytes;用 MPS,`float32`,不开 bf16。
- CUDA:`bf16=True`;可选 `use_vllm=True`。

---

## 3. 模块规格

### 3.1 `ballwall/task.py`

**坐标与物理(单位一律是像素和秒,一维运动):**
- 画布 1000 × 1000,x 向右,y 向下。球的 y 固定为 500。
- 墙是一个矩形,左边(内侧面)在 `wall_x`,延伸到画布右边缘。
- 接触条件:球最右点碰到墙内侧面,即球心 x = `wall_x - r`(记作 `xc`)。
- 接触时刻 `tc = (xc - x0) / v`。
- 标准答案:`t <= tc` 时 `x = x0 + v·t`;`t > tc` 时 `x = xc - e·v·(t - tc)`。

**题目参数(一道题是一个 dict):**

| 字段 | 含义 | 取值 |
|---|---|---|
| `split` | `train` / `eval_in` / `heldout` | |
| `n_frames` | 帧数 | 16 |
| `dt` | 帧间隔(秒) | train、eval_in:{0.05, 0.1, 0.2};heldout:{0.15, 0.25} |
| `x0` | 球心起点 | 整数 [60, 220] |
| `y` | 球心 y | 500 |
| `r` | 半径 | 整数 [12, 30] |
| `wall_x` | 墙内侧面 | 整数 [700, 950] |
| `e` | 恢复系数,保留两位小数 | train、eval_in:[0.30, 0.85];heldout:[0.86, 0.95] |
| `v` | 速度(px/s,整数) | 由下面的约束反推 |
| `canvas` | 1000 | |

**采样方法:**先随机 `r, x0, wall_x, e, dt`,再随机一个目标接触帧位置 `fc ∈ [4.3, n_frames - 5.3]`,反推 `v = round((xc - x0) / (fc·dt))`,然后**用取整后的 v 重新计算** `fc = tc / dt`,检查以下约束,不满足就重新采样:
1. `4.3 <= fc <= n_frames - 5.3`(接触前后各至少 5 帧)。
2. `fc` 的小数部分在 [0.1, 0.9](接触时刻不贴着采样帧,避免歧义)。
3. 最后一帧的标准答案 `x_end >= r + 10`(反弹后不出画布)。

**函数:**
- `sample_task(rng, split, n_frames=16) -> dict`
- `contact_time(task) -> float`
- `x_at(task, t, e=None) -> float`(`e` 可覆盖,供笨策略使用)
- `truth_xs(task) -> list[float]`
- `build_messages(task) -> list[{"role","content"}]`(system + user,内容见下)
- `render_svg(task, xs, ys=None, r=None) -> str`(把一组坐标渲染成合规 SVG,供自检和笨策略使用)
- `write_jsonl(path, tasks)` / `read_jsonl(path)`:每行 `{"task": {...}, "prompt": [messages]}`

**System prompt(英文,原样使用):**

```
You draw physical processes as SVG animation frames. You must NOT write code. Output exactly one <svg> element and nothing else (no explanation, no markdown).

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
</svg>
```

示例故意用**另一个场景**(向左匀速、无墙),只教格式,不泄露碰撞答案。

**User prompt 模板:**

```
Scene: a ball moves horizontally to the right, collides with a vertical wall, and bounces back.
- Wall: its left (inner) face is at x = {wall_x}. Draw it as <rect id="wall" x="{wall_x}" y="0" width="{1000-wall_x}" height="1000"/>.
- Ball: radius {r} px. Its center starts at x = {x0}, y = {y}. Initial velocity: {v} px/s to the right. No gravity, no friction.
- The ball touches the wall when its rightmost point reaches the wall's inner face. The collision has coefficient of restitution e = {e}.
- Frame K shows the scene at time t = K * {dt} s, for K = 0..{n-1} ({n} frames).

Output the {n} frames as one SVG.
```

提示里**不给公式、不给轨迹、不给接触时刻**。

### 3.2 `ballwall/checker.py`(核心)

入口:`score(text: str, task: dict, violation_mode: str = "low", tol: dict | None = None) -> dict`

#### 容差(默认值,集中放在 `DEFAULT_TOL`,可被参数覆盖)

| 键 | 默认 | 含义 |
|---|---|---|
| `wall_px` | 1.0 | SVG 里墙的 x 与题目参数允许的偏差 |
| `r_px` | 1.0 | SVG 里半径与题目参数允许的偏差 |
| `y_px` | 2.0 | 球 y 的允许漂移 |
| `pen_px` | 3.0 | 穿墙容差 |
| `teleport` | 1.5 | 帧间位移上限 = `teleport · v · dt` |
| `static_frac` | 0.25 | 球运动范围小于标准答案范围的这个比例,判为"不动" |
| `reach` | 0.5 | 最大 x 比标准答案最大 x 小超过 `reach · v · dt`,判为"未到墙" |
| `energy` | 1.05 | 反弹速度大于入射速度的这个倍数,判为"能量增加" |
| `t_c` | 0.30 | 接触时刻误差容差(以 dt 为单位) |
| `e_rel` | 0.08 | 反弹速度比相对 e 的误差容差 |
| `v_rel` | 0.05 | 入射速度相对 v 的误差容差 |
| `resid` | 0.10 | 拟合残差 RMS 容差(以 v·dt 为单位) |

#### 第一步:解析(任何一条不满足 → 格式失败)

1. 去掉可能的 markdown 代码块标记,用正则 `<svg\b.*?</svg>`(DOTALL)取**第一个** svg 元素;取不到 → 失败。
2. `xml.etree.ElementTree.fromstring` 解析;解析错误 → 失败。标签去掉命名空间前缀后再判断。
3. 禁止的元素:`animate, animateMotion, animateTransform, set, script, use, foreignObject, style`。任意元素带 `transform` 属性 → 失败。
4. 必须恰好有一个 `id="wall"` 的元素,且是 `rect`;它的 `x` 必须是纯数字,并与 `task["wall_x"]` 相差不超过 `wall_px`。**评分时一律使用题目参数 `wall_x`,不使用 SVG 里的值。**
5. 帧:所有 `id` 匹配 `^f\d+$` 的 `<g>`。id 集合必须**恰好**是 `f0 … f{n-1}`,不能重复、缺失或多余。
6. 每帧内部(含嵌套)恰好一个 `class` 中包含 `ball` 的 `<circle>`。整份 SVG 中这种 circle 的总数必须等于 `n`(防止在帧外藏额外的球)。
7. `cx, cy, r` 必须匹配 `^\s*-?\d+(\.\d+)?\s*$`(拒绝 `880px`、科学计数法等);`r` 与 `task["r"]` 相差不超过 `r_px`。**评分时使用题目参数 r。**

输出:`xs, ys`(长度 n)。

#### 第二步:硬约束(按顺序判定,命中第一条即停,记为该类别)

记 `vdt = v · dt`,`truth = truth_xs(task)`,`xc = wall_x - r`,`ts = [k·dt]`。

1. **`teleport_out_static`(瞬移/越界/不动)**,依次检查:
   - `max |y_k - task.y| > y_px`(y 漂移)
   - `min(xs) - r < -pen_px`(出画布左边)
   - `max(xs) - min(xs) < static_frac · (max(truth) - min(truth))`(几乎不动)
   - `max |x_{k+1} - x_k| > teleport · vdt`(瞬移)
2. **`penetration`(穿墙)**:`max(xs) + r > wall_x + pen_px`。
3. **`no_reach_or_no_rebound`(未到墙或未反弹)**:
   - `max(xs) < max(truth) - reach · vdt` → 未到墙(提前掉头)。
   - 令 `kp = argmax(xs)`(峰值帧)。若 `kp < 2` 或 `kp > n - 3` → 没有可用的反弹段。
   - 对 `0 … kp-1` 帧做最小二乘直线拟合 `x = a1 + b1·t`(**排除峰值帧**,因为它可能跨越接触时刻);对 `kp+1 … n-1` 帧拟合 `x = a2 + b2·t`。
   - `v_in = b1`,`v_out = -b2`。若 `v_in <= 0` 或 `v_out <= 0` → 没有反向。
   - 注意:是否反向按**模型自己的轨迹**判断,不按标准答案的接触时刻。
4. **`energy_gain`(能量增加)**:`v_out > energy · v_in`。

只要两段拟合能算出来(即使后面判了违规),就继续计算软精度,供 `violation_mode="low"` 使用。

#### 第三步:软精度(只用一维场景里真正独立的量)

| 分量 | 误差定义 | 容差 | 归入类别 |
|---|---|---|---|
| 接触时刻 | `|t_cross - tc| / dt`,`t_cross = (a2 - a1) / (b1 - b2)`(两条拟合线的交点,即模型隐含的掉头时刻) | `t_c` | `contact_time` |
| 恢复系数 | `|v_out / v_in - e| / e` | `e_rel` | `restitution` |
| 入射速度 | `|v_in - v| / v` | `v_rel` | `trajectory` |
| 匀速性 | `max(rms1, rms2) / vdt`(两段拟合的残差 RMS) | `resid` | `trajectory` |

- 每个分量的得分:`s = 1`(误差 ≤ 容差),否则 `s = exp(-(误差 - 容差) / 容差)`。
- `soft` = 四个分量得分的**几何平均**(不用算术平均,和"全部在容差内才算成功"的定义对齐)。
- 软精度失败的类别:按 `contact_time → restitution → trajectory` 的顺序,取第一个超出容差的分量。

不用"逐帧与标准答案的绝对误差"作为软分量,原因是:它和上面四个量高度冗余,而且它的容差很难和 e 的容差自洽(e 差 10% 时,反弹后期位置误差可以很大)。但把它作为**诊断量**记录下来(见返回值)。

#### 第四步:合成 reward 和成功

```
格式失败:                    reward = 0
违反任一硬约束:
    violation_mode == "zero": reward = 0
    violation_mode == "low":  reward = 0.2 × soft   (拟合算不出来时 soft = 0)
满足全部硬约束:              reward = 0.3 + 0.7 × soft
成功:满足全部硬约束 且 四个软分量都在容差内;此时 reward 显式设为 1.0
```

- 默认 `violation_mode="low"`。是否改成 `"zero"` 由基线的零方差组比例决定(见第 7 节)。
- 由于容差内记满分,**成功 ⇔ reward == 1.0**。报告里要如实说明成功率是 reward 的阈值化,不是独立证据。

#### 返回值(dict)

```python
{
  "reward": float,
  "success": bool,
  "category": str,        # 见下表,共 9 类
  "format_ok": bool,
  "hard_ok": bool,
  "reason": str,          # 人类可读的失败原因
  "soft": float | None,
  "components": {"contact_time": err, "restitution": err, "v_in": err, "resid": err},
  "diag": {
     "v_in": ..., "v_out": ..., "e_model": ..., "t_cross": ..., "tc": ...,
     "traj_err_mean": ...,      # mean |x_k - truth_k| / vdt
     "first_post_err": ...,     # 接触后第一帧的 |x - truth| / vdt,区分 e=1 镜像反射
  },
}
```

#### 失败类别(互斥,顺序固定)

| key | 中文 |
|---|---|
| `format` | 格式失败 |
| `teleport_out_static` | 瞬移/越界/不动 |
| `penetration` | 穿墙 |
| `no_reach_or_no_rebound` | 未到墙或未反弹 |
| `energy_gain` | 能量增加 |
| `contact_time` | 撞墙时刻偏差 |
| `restitution` | 反弹速度比偏差 |
| `trajectory` | 轨迹偏差(入射速度或匀速性) |
| `success` | 成功 |

这个顺序与 `report_mock.html` 中 `CATS` 数组一致,报告可以直接按这个顺序填数。

### 3.3 `ballwall/baselines.py`

每个策略是 `f(task, rng) -> xs`,再用 `render_svg` 渲染成文本交给评分器。至少实现:

| 名称 | 做法 | 预期 |
|---|---|---|
| `oracle` | 标准答案 | 100% 成功,reward 全部 = 1.0 |
| `slight_e` | 用 `e × 1.04` 计算反弹 | 大部分成功(在 `e_rel` 容差内) |
| `jitter` | 标准答案加均匀噪声 ±0.03·vdt | 大部分成功 |
| `mirror_e1` | 按 e = 1 镜像反射 | 训练分布内几乎不成功;heldout 中 e 接近 1 的题可能成功 |
| `fixed_e05` | 正确时刻掉头,但 e 固定 0.5 | 仅当真实 e 在 0.5 附近才成功 |
| `early_turn` | 走到 60% 距离就以 e 反弹 | 硬约束违规(未到墙) |
| `small_penetration` | 标准答案,但最靠近墙的那帧让球表面越过墙面 5px | 硬约束违规(穿墙) |
| `pass_through` | `x0 + v·t`,一直向右 | 硬约束违规(穿墙或瞬移) |
| `static` | 一直在 x0 | 硬约束违规(不动) |
| `random` | 每帧随机 x | 硬约束违规(瞬移) |

### 3.4 `ballwall/metrics.py`

`summarize(results: list[dict]) -> dict`,输出:
- `n`、`success_rate`、`success_ci95`(Wilson 区间)
- `mean_reward`
- `format_fail_rate`
- `hard_violation_rate`(占全部样本,不含格式失败)
- `category_dist`:9 类占比,和为 1
- 链式分解:`p_format_ok`、`p_hard_ok_given_format`、`p_success_given_hard`(分母为 0 时记 `null`)

---

## 4. 脚本

### 4.1 `scripts/make_data.py`

- 参数:`--seed 0 --n_train 2000 --n_eval 200 --out data/`
- 输出:`data/train.jsonl`、`data/eval_in.jsonl`、`data/eval_heldout.jsonl`
- 三个集合用不同的随机种子。生成后打印每个集合的参数范围摘要。
- **生成的数据提交到仓库**,保证训练机和评测机用同一份题。

### 4.2 `scripts/selftest.py`(RL 之前必须全部通过)

1. 对三个数据集的**每一道题**:`oracle` 必须判成功,reward == 1.0。不通过就打印题目参数并以非零码退出。
2. 对 `eval_in` 和 `eval_heldout`,跑第 3.3 节全部策略,打印表格:策略、成功率、平均 reward、最常见的类别。
3. 断言:
   - `early_turn`、`small_penetration`、`pass_through`、`static`、`random` 的成功率为 0。
   - 违规策略的平均 reward 低于任何"满足硬约束"策略的平均 reward(验证字典序)。
   - `slight_e`、`jitter` 的成功率 ≥ 80%(否则说明容差过严)。
4. 对 `mirror_e1` 和 `fixed_e05` 只打印,不断言,但如果它们在 `eval_in` 上的成功率超过 10%,打印警告:容差可能过松。
5. 解析健壮性用例(各一条,必须判为格式失败):缺少某一帧、重复帧 id、`cx="880px"`、带 `transform`、带 `<animate>`、墙的 x 与参数不一致、半径与参数不一致、帧外多一个 ball、输出里没有 svg。
6. 把表格写到 `outputs/selftest.md`,方便贴进报告。

### 4.3 `scripts/evaluate.py`

- 参数:`--model Qwen/Qwen2.5-1.5B-Instruct --adapter <可选,LoRA 路径> --data data/eval_in.jsonl data/eval_heldout.jsonl --limit <可选> --batch_size 8 --max_new_tokens 768 --violation_mode low --out outputs/eval/<name>.json --tag <检查点步数>`
- 设备:CUDA → `bfloat16`;MPS → `float32`;否则 CPU。
- 生成:`tokenizer.apply_chat_template(messages, add_generation_prompt=True)`,**左侧 padding**,贪心解码(`do_sample=False`)。
- 加载 LoRA:`PeftModel.from_pretrained(base, adapter)`。
- 每个样本用评分器打分;输出:
  - `outputs/eval/<name>.json`:每个数据集一份 `summarize` 结果,外加 `tag`、模型名、时间。
  - `outputs/eval/<name>.samples.jsonl`:每条的题目、模型原文、评分结果(方便人工抽查)。
- 可选 `--backend vllm`(仅 CUDA):用 vLLM + `LoRARequest` 加速;不是必须项。

### 4.4 `scripts/train_grpo.py`

**数据:**读 `data/train.jsonl`,构造 HF `Dataset`,两列:
- `prompt`:消息列表(system + user)
- `task_json`:`json.dumps(task)`

TRL 会把 `task_json` 作为关键字参数传给奖励函数。

**奖励函数:**
- `physics_reward(prompts, completions, task_json, **kw)`:**唯一参与训练的奖励**,返回 `score(...)["reward"]`。
- 用于日志、权重为 0 的指标函数:`metric_success`、`metric_format_ok`、`metric_hard_ok`。
- 用 `functools.lru_cache` 按 `(completion_text, task_json)` 缓存 `score` 的结果,避免重复计算。
- completion 的取法:对话格式时是 `completion[0]["content"]`,纯文本时就是字符串本身;两种都要兼容。

**不要**把 reward 拆成多个函数让 TRL 相加:那样会破坏字典序,违规答案可以靠其他分量的分数抵消穿墙。

**模型:**
- 默认 `Qwen/Qwen2.5-1.5B-Instruct`;显存紧张或在 Mac 上时用 `Qwen/Qwen2.5-0.5B-Instruct`。
- 若用 Qwen3,要关闭 thinking(否则会输出推理文本,违反"只输出 SVG"的设定 A)。不确定怎么在 TRL 里关的话,就不要用 Qwen3。
- `tokenizer.pad_token` 为空时设为 `eos_token`。

**LoRA:** `r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM"`

**GRPOConfig(起始值):**

| 参数 | CUDA | Mac(MPS) |
|---|---|---|
| `learning_rate` | 2e-5 | 2e-5 |
| `lr_scheduler_type` / `warmup_ratio` | `constant_with_warmup` / 0.05 | 同左 |
| `num_generations` | 8 | 4 |
| `per_device_train_batch_size` | 16 | 4 |
| `gradient_accumulation_steps` | 4 | 4 |
| `max_prompt_length` | 512 | 512 |
| `max_completion_length` | 768 | 768(16 帧约需 400 到 600 token,不能再小) |
| `temperature` | 1.0 | 1.0 |
| `beta`(KL 系数) | 0.02 | 0.02 |
| `max_steps` | 60(时间允许再加) | 先 1 步冒烟测试,再决定 |
| `bf16` | True | False |
| `use_vllm` | 可选 True | False |
| `logging_steps` / `save_steps` | 1 / 20 | 1 / 10 |
| `reward_weights` | `[1, 0, 0, 0]` | 同左 |
| `log_completions` | True(若支持) | 同左 |
| `report_to` | `wandb` 或 `none` | `none` |

- 每步的补全总数(`per_device_train_batch_size × gradient_accumulation_steps × GPU 数`)必须能被 `num_generations` 整除。
- **TRL 版本兼容**:不同版本的 `GRPOConfig` 参数名不同(例如 `max_prompt_length`、`use_vllm` 的模式参数、`reward_weights`、`log_completions`)。实现时先用 `dataclasses.fields(GRPOConfig)` 取出支持的参数名,过滤掉不支持的并打印警告。如果 `reward_weights` 不支持,就只传 `physics_reward`,指标改由 `evaluate.py` 计算。
- 命令行参数:`--model --out --max_steps --num_generations --lr --beta --violation_mode --use_vllm`,其余用上表默认值。
- 训练结束保存 LoRA 到 `<out>/final`;中间检查点按 `save_steps` 保存。
- 训练前打印:设备、dtype、TRL 版本、每步题数和补全数。

### 4.5 `scripts/collect_curve.py`

- 读 `outputs/eval/*.json`,按 `tag`(训练步数)排序。
- 输出 `outputs/curve.json`:

```json
{
  "steps": [0, 20, 40, 60],
  "eval_in":      {"success": [...], "ci_low": [...], "ci_high": [...], "mean_reward": [...], "category_dist": [[9 个占比], ...]},
  "eval_heldout": {"...同上": "..."}
}
```

- 同时把训练日志里的在线指标(`reward`、`reward_std`、零方差组比例、`metric_success` 等,若有)导出为 `outputs/train_log.json`。

---

## 5. 训练中要监控的指标

| 指标 | 健康的样子 | 异常时怎么办 |
|---|---|---|
| `reward` 均值 | 上升 | 不动:打印样本,先查评分器和格式 |
| `reward_std` | 保持大于 0 | 趋近 0:组内无差异,没有梯度 |
| 零方差组比例(若 TRL 有 `frac_reward_zero_std`) | 低于 30% 到 50% | 过高:补 few-shot,或改 `violation_mode` |
| `metric_success`(在线) | 上升 | reward 涨而它不涨:看失败类别 |
| `kl` | 缓慢增长 | 大于 0.5:降学习率 |
| 补全长度 | 稳定在 400 到 700 | 被截断(达到上限):格式会失败,检查 `max_completion_length` |
| loss | 从 0 附近上升 | 正常现象,不用于判断进度 |

每 10 步人工看 2 条生成样本,检查有没有钻空子。

---

## 6. 执行顺序与验收

| 步骤 | 内容 | 验收标准 | 依赖 GPU |
|---|---|---|---|
| 1 | `task.py`、`checker.py`、`baselines.py`、`metrics.py` | 能 import,`selftest.py` 可运行 | 否 |
| 2 | `make_data.py` 生成数据 | 三个 jsonl 生成,参数范围符合第 3.1 节 | 否 |
| 3 | `selftest.py` | 第 4.2 节全部断言通过;`outputs/selftest.md` 生成 | 否 |
| 4 | `evaluate.py` 跑基线(第 0 步) | 两个评测集各 200 题出结果;记录格式失败率、零方差组比例 | 是 |
| 5 | 根据基线定 `violation_mode`,必要时调 few-shot | 决定写进 `NOTES.md` | 否 |
| 6 | `train_grpo.py` 冒烟测试 1 步 | 不报错;记录每步耗时 | 是 |
| 7 | 正式训练 | 按时间跑满步数,保存检查点 | 是 |
| 8 | 对各检查点跑 `evaluate.py`,再跑 `collect_curve.py` | `outputs/curve.json` 生成 | 是 |

**第 3 步不通过,不要开始第 4 步以后的工作。**

基线零方差组比例的统计方法:取 50 道训练题,每题用 temperature 1.0 采样 `num_generations` 条,打分,统计 reward 完全相同的组的比例。可以作为 `evaluate.py` 的一个选项(`--sample_k 8`)。

---

## 7. 待决定事项(实现时做成开关,不要写死)

1. **`violation_mode`**:`low`(违规给 0.2×soft)还是 `zero`(违规给 0)。规则:基线零方差组比例低于 30% 用 `zero`;超过 50% 用 `low`;中间任选,但要记录。
2. **容差**:先用默认值;自检中 `slight_e`、`jitter` 成功率低于 80% 就放宽,笨策略成功率过高就收紧。改动写进 `NOTES.md`。
3. **模型大小**:由训练机显存和第 6 步的单步耗时决定。

---

## 8. 不要做的事

- 不要从 SVG 里读墙的位置和球半径来评分(模型可以把墙画远、把球画小来规避穿墙)。
- 不要把 reward 拆成多个函数让 TRL 相加。
- 不要用标准答案的轨迹做 SFT 数据(会把答案直接教给模型);冷启动最多只用 few-shot 或只教格式。
- 不要用"最靠近墙的那一帧"当接触时刻;用两条拟合线的交点。
- 不要在评测时用采样解码;评测一律贪心解码。
- 不要修改 `data/` 里已提交的题目;需要新题就换种子另存新文件。

---

## 9. 与报告的对接

`report_mock.html` 目前全是占位数据。真实结果出来后:
- 图 1、图 2 和表 1 用 `outputs/eval/*.json` 的 `summarize` 结果填。
- 图 3、图 4 目前是由 `RL.start` / `RL.end` 插值画出来的,要改成直接读 `outputs/curve.json` 的逐检查点数据。
- 表 2(评分器自检)和表 3(笨策略)用 `outputs/selftest.md` 的实际结果替换手算估计。
- 删除页面顶部的"示例数据"横幅前,确认每一个数字都来自实际结果。
