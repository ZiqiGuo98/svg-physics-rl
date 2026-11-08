# 实现记录(偏离、调参、实际环境)

按 `IMPLEMENTATION_PLAN.md` 第 0 节要求,所有偏离和调参都记在这里。日期 2026-11-07。

## 实际环境

| 项目 | 值 |
|---|---|
| 机器 | Windows 11 笔记本,NVIDIA RTX 3050 Ti Laptop GPU(4 GB 显存),i7-11370H,32 GB 内存 |
| Python | 3.14.0(上级目录 `aie-hackathon/.venv`,不是计划建议的 3.10/3.11;所需轮子都能装上) |
| 库版本 | torch 2.9.1+cu128,transformers 5.22.2,trl 1.17.1,peft 0.21.2 |
| vLLM | 未安装(Windows) |

核心模块 `ballwall/` 只用标准库,`python -I scripts/selftest.py` 用系统 Python 也能跑。

## 第 1–3 步:生成器、评分器、自检

- 数据:`python scripts/make_data.py`(seed 0;train 2000、eval_in 200、eval_heldout 200;三个集合种子分别为 1、2、3)。参数范围与计划第 3.1 节一致,接触帧 4.30–10.70,小数部分 0.10–0.90。
- 题目多了一个 `id` 字段(如 `eval_in-00007`),方便抽查;评分不使用它。
- 自检全部通过,结果在 `outputs/selftest.md`。
- 一条警告,未改容差:`fixed_e05` 在 eval_in 上成功率 12.5%,超过计划设的 10% 警戒线。原因不是漏洞:恢复系数容差 8% 意味着真实 e 落在 0.463–0.543 的题,固定猜 0.5 就会判成功;这个区间约占训练分布 [0.30, 0.85] 的 15%。收紧 `e_rel` 会让 `slight_e`(e×1.04)开始失败,所以保留默认值。同理,heldout 上 `mirror_e1` 成功 25.5% 是因为 e ≥ 0.926 的题与 e=1 的差距在 8% 以内,计划已预期。
- 额外加的解析用例:科学计数法 `1e2`、截断的 SVG(没有 `</svg>`)都判格式失败;包在 ```` ```svg ```` 代码块里的正确答案仍判成功。
- 帧里 ball 的识别规则:`class` 按空格切分后含 `ball`(不是子串匹配)。

## 评测脚本的实现细节

- 贪心解码,左侧 padding,`stop_strings=["</svg>"]`:生成到第一个 `</svg>` 就停,评分器本来也只读第一个 svg 元素,这只省时间,不改变得分。训练时 TRL 不传这个参数。
- LoRA 评测时先 `merge_and_unload()` 再生成,加快速度。
- `--sample_k` 探针同时算 `low` 和 `zero` 两种 violation_mode 下的零方差组比例,供第 7 节的决策用。
- `collect_curve.py` 只收 `outputs/eval/step*.json`,避免把其他模型的一次性评测画进训练曲线。

## 第 4 步以后

见下文追加的记录。
