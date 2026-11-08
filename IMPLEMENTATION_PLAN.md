# Implementation Plan: Ball-Hits-Wall SVG Grader + TRL GRPO Training

> Audience: the coding agent responsible for implementation, and teammates.
> Background and design rationale are in `plan_physics_svg_frames.md`; this document only covers **what to implement, how to implement it, and how to accept it**. If the two documents conflict, this one takes precedence.
> The numbers in this document (tolerances, hyperparameters) are starting values. They may be adjusted after the self-check, but any adjustment must be recorded in `NOTES.md`.

---

## 0. Goals and Scope

**Deliverables:**
1. Task generator: randomly generates "ball hits wall" tasks, reference answers, and prompt text.
2. Grader: parses the SVG output by the model, checks format, hard constraints, and soft accuracy, and outputs a reward, a success flag, and a failure category.
3. Self-check: reference answers, hand-written wrong answers, and dumb baselines, proving the grader is correct and discriminative.
4. Evaluation script: evaluates a base model or LoRA checkpoint with greedy decoding, outputs success rate, confidence interval, and failure-category proportions.
5. Training script: TRL `GRPOTrainer` + LoRA, using the grader as the reward.
6. Curve aggregation: aggregates evaluation results from multiple checkpoints into one JSON for the report.

**Out of scope:**
- The "variant" scene with gravity and a ground plane.
- The SVG vs. coordinate-list (JSON) comparison experiment (already recorded as a limitation).
- verl, multi-node training.

**Hard requirements:**
- The core modules under `ballwall/` (generator, grader, dumb baselines, metrics) **use only the Python standard library** and are compatible with Python 3.9+, so the self-check can run on any machine.
- The training and evaluation scripts may depend on torch / transformers / trl / peft, but must be **device-agnostic**: automatically select CUDA, MPS (Mac), or CPU.

---

## 1. Directory Layout

```
svg-physics-rl/
├── ballwall/
│   ├── __init__.py
│   ├── task.py         # task generation, reference answers, prompts, rendering, jsonl read/write
│   ├── checker.py      # parsing + scoring (core)
│   ├── baselines.py    # dumb baselines and hand-written wrong answers
│   └── metrics.py      # summary statistics
├── scripts/
│   ├── make_data.py    # generate datasets
│   ├── selftest.py     # grader self-check (must pass before RL)
│   ├── evaluate.py     # model evaluation
│   ├── train_grpo.py   # GRPO training
│   └── collect_curve.py# aggregate per-checkpoint results
├── data/               # generated jsonl (committed to the repo so everyone uses the same tasks)
├── outputs/            # checkpoints and evaluation results (already in .gitignore)
├── requirements.txt
├── NOTES.md            # deviations and tuning records during implementation
└── IMPLEMENTATION_PLAN.md
```

Each script adds the repo root to `sys.path` at the top, so `python scripts/xxx.py` can `import ballwall` directly.

---

## 2. Environment

`requirements.txt` (only needed for training and evaluation; use whatever versions actually install, and record the actual versions in `NOTES.md` after installation):

```
torch
transformers
trl
peft
datasets
accelerate
# optional, CUDA only: vllm
```

- Use Python 3.10 or 3.11 in a fresh virtual environment (uv or conda). The system-provided 3.9 may not be able to install recent TRL.
- Mac: no CUDA, so vLLM and bitsandbytes are unavailable; use MPS, `float32`, and do not enable bf16.
- CUDA: `bf16=True`; optionally `use_vllm=True`.

---

## 3. Module Specifications

### 3.1 `ballwall/task.py`

**Coordinates and physics (units are always pixels and seconds; one-dimensional motion):**
- Canvas 1000 × 1000, x increases to the right, y increases downward. The ball's y is fixed at 500.
- The wall is a rectangle whose left (inner) face is at `wall_x`, extending to the right edge of the canvas.
- Contact condition: the ball's rightmost point touches the inner face of the wall, i.e. ball center x = `wall_x - r` (denoted `xc`).
- Contact time `tc = (xc - x0) / v`.
- Reference answer: for `t <= tc`, `x = x0 + v·t`; for `t > tc`, `x = xc - e·v·(t - tc)`.

**Task parameters (one task is one dict):**

| Field | Meaning | Values |
|---|---|---|
| `split` | `train` / `eval_in` / `heldout` | |
| `n_frames` | number of frames | 16 |
| `dt` | frame interval (seconds) | train, eval_in: {0.05, 0.1, 0.2}; heldout: {0.15, 0.25} |
| `x0` | starting ball center | integer in [60, 220] |
| `y` | ball center y | 500 |
| `r` | radius | integer in [12, 30] |
| `wall_x` | inner face of the wall | integer in [700, 950] |
| `e` | coefficient of restitution, rounded to two decimals | train, eval_in: [0.30, 0.85]; heldout: [0.86, 0.95] |
| `v` | velocity (px/s, integer) | derived from the constraints below |
| `canvas` | 1000 | |

**Sampling method:** first sample `r, x0, wall_x, e, dt` at random, then sample a target contact frame position `fc ∈ [4.3, n_frames - 5.3]`, derive `v = round((xc - x0) / (fc·dt))`, then **recompute `fc = tc / dt` using the rounded v**, and check the following constraints; resample if any is not satisfied:
1. `4.3 <= fc <= n_frames - 5.3` (at least 5 frames before and after contact).
2. The fractional part of `fc` is in [0.1, 0.9] (the contact time does not sit right on a sampled frame, avoiding ambiguity).
3. The reference answer at the last frame satisfies `x_end >= r + 10` (the ball does not leave the canvas after the bounce).

**Functions:**
- `sample_task(rng, split, n_frames=16) -> dict`
- `contact_time(task) -> float`
- `x_at(task, t, e=None) -> float` (`e` may be overridden, for use by dumb baselines)
- `truth_xs(task) -> list[float]`
- `build_messages(task) -> list[{"role","content"}]` (system + user; content below)
- `render_svg(task, xs, ys=None, r=None) -> str` (renders a list of coordinates into a compliant SVG, for use by the self-check and dumb baselines)
- `write_jsonl(path, tasks)` / `read_jsonl(path)`: each line is `{"task": {...}, "prompt": [messages]}`

**System prompt (English, used verbatim):**

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

The example deliberately uses **a different scene** (constant velocity to the left, no wall): it only teaches the format and does not leak the collision answer.

**User prompt template:**

```
Scene: a ball moves horizontally to the right, collides with a vertical wall, and bounces back.
- Wall: its left (inner) face is at x = {wall_x}. Draw it as <rect id="wall" x="{wall_x}" y="0" width="{1000-wall_x}" height="1000"/>.
- Ball: radius {r} px. Its center starts at x = {x0}, y = {y}. Initial velocity: {v} px/s to the right. No gravity, no friction.
- The ball touches the wall when its rightmost point reaches the wall's inner face. The collision has coefficient of restitution e = {e}.
- Frame K shows the scene at time t = K * {dt} s, for K = 0..{n-1} ({n} frames).

Output the {n} frames as one SVG.
```

The prompt gives **no formula, no trajectory, and no contact time**.

### 3.2 `ballwall/checker.py` (core)

Entry point: `score(text: str, task: dict, violation_mode: str = "low", tol: dict | None = None) -> dict`

#### Tolerances (defaults, centralized in `DEFAULT_TOL`, overridable via the parameter)

| Key | Default | Meaning |
|---|---|---|
| `wall_px` | 1.0 | allowed deviation between the wall x in the SVG and the task parameter |
| `r_px` | 1.0 | allowed deviation between the radius in the SVG and the task parameter |
| `y_px` | 2.0 | allowed drift of the ball's y |
| `pen_px` | 3.0 | wall penetration tolerance |
| `teleport` | 1.5 | upper bound on inter-frame displacement = `teleport · v · dt` |
| `static_frac` | 0.25 | if the ball's range of motion is smaller than this fraction of the reference answer's range, classify as "static" |
| `reach` | 0.5 | if the maximum x is smaller than the reference maximum x by more than `reach · v · dt`, classify as "did not reach the wall" |
| `energy` | 1.05 | if the rebound speed exceeds this multiple of the incoming speed, classify as "energy gain" |
| `t_c` | 0.30 | contact-time error tolerance (in units of dt) |
| `e_rel` | 0.08 | relative error tolerance of the rebound speed ratio w.r.t. e |
| `v_rel` | 0.05 | relative error tolerance of the incoming speed w.r.t. v |
| `resid` | 0.10 | fit residual RMS tolerance (in units of v·dt) |

#### Step 1: Parsing (any unmet condition → format failure)

1. Strip any markdown code-fence markers, then take the **first** svg element with the regex `<svg\b.*?</svg>` (DOTALL); if none is found → fail.
2. Parse with `xml.etree.ElementTree.fromstring`; parse error → fail. Strip namespace prefixes from tags before checking them.
3. Forbidden elements: `animate, animateMotion, animateTransform, set, script, use, foreignObject, style`. Any element with a `transform` attribute → fail.
4. There must be exactly one element with `id="wall"`, and it must be a `rect`; its `x` must be a plain number and differ from `task["wall_x"]` by no more than `wall_px`. **Scoring always uses the task parameter `wall_x`, never the value from the SVG.**
5. Frames: all `<g>` elements whose `id` matches `^f\d+$`. The set of ids must be **exactly** `f0 … f{n-1}`, with no duplicates, gaps, or extras.
6. Each frame contains (including nested elements) exactly one `<circle>` whose `class` contains `ball`. The total number of such circles in the whole SVG must equal `n` (prevents hiding extra balls outside the frames).
7. `cx, cy, r` must match `^\s*-?\d+(\.\d+)?\s*$` (reject `880px`, scientific notation, etc.); `r` must differ from `task["r"]` by no more than `r_px`. **Scoring uses the task parameter r.**

Output: `xs, ys` (length n).

#### Step 2: Hard constraints (checked in order; stop at the first hit and record that category)

Let `vdt = v · dt`, `truth = truth_xs(task)`, `xc = wall_x - r`, `ts = [k·dt]`.

1. **`teleport_out_static` (teleport / out of bounds / static)**, checked in order:
   - `max |y_k - task.y| > y_px` (y drift)
   - `min(xs) - r < -pen_px` (leaves the canvas on the left)
   - `max(xs) - min(xs) < static_frac · (max(truth) - min(truth))` (almost no motion)
   - `max |x_{k+1} - x_k| > teleport · vdt` (teleport)
2. **`penetration` (wall penetration)**: `max(xs) + r > wall_x + pen_px`.
3. **`no_reach_or_no_rebound` (did not reach the wall or did not rebound)**:
   - `max(xs) < max(truth) - reach · vdt` → did not reach the wall (turned around early).
   - Let `kp = argmax(xs)` (peak frame). If `kp < 2` or `kp > n - 3` → no usable rebound segment.
   - Least-squares line fit `x = a1 + b1·t` on frames `0 … kp-1` (**excluding the peak frame**, since it may straddle the contact time); fit `x = a2 + b2·t` on frames `kp+1 … n-1`.
   - `v_in = b1`, `v_out = -b2`. If `v_in <= 0` or `v_out <= 0` → no reversal.
   - Note: reversal is judged from **the model's own trajectory**, not from the reference answer's contact time.
4. **`energy_gain` (energy gain)**: `v_out > energy · v_in`.

As long as both fits can be computed (even if a violation is recorded afterwards), continue computing soft accuracy so it is available for `violation_mode="low"`.

#### Step 3: Soft accuracy (uses only the quantities that are truly independent in the 1-D scene)

| Component | Error definition | Tolerance | Category |
|---|---|---|---|
| Contact time | `|t_cross - tc| / dt`, `t_cross = (a2 - a1) / (b1 - b2)` (intersection of the two fitted lines, i.e. the model's implied turnaround time) | `t_c` | `contact_time` |
| Coefficient of restitution | `|v_out / v_in - e| / e` | `e_rel` | `restitution` |
| Incoming speed | `|v_in - v| / v` | `v_rel` | `trajectory` |
| Uniformity of motion | `max(rms1, rms2) / vdt` (residual RMS of the two fits) | `resid` | `trajectory` |

- Score of each component: `s = 1` (error ≤ tolerance), otherwise `s = exp(-(error - tolerance) / tolerance)`.
- `soft` = the **geometric mean** of the four component scores (not the arithmetic mean, to align with the definition "success only when everything is within tolerance").
- Soft-accuracy failure category: in the order `contact_time → restitution → trajectory`, take the first component that exceeds its tolerance.

The "per-frame absolute error against the reference answer" is not used as a soft component, because it is highly redundant with the four quantities above, and its tolerance is hard to make consistent with the tolerance on e (when e is off by 10%, the position error late in the rebound can be large). It is, however, recorded as a **diagnostic** (see the return value).

#### Step 4: Composing the reward and success

```
Format failure:                    reward = 0
Any hard constraint violated:
    violation_mode == "zero": reward = 0
    violation_mode == "low":  reward = 0.2 × soft   (soft = 0 if the fits cannot be computed)
All hard constraints satisfied:    reward = 0.3 + 0.7 × soft
Success: all hard constraints satisfied AND all four soft components within tolerance; reward is then explicitly set to 1.0
```

- Default `violation_mode="low"`. Whether to switch to `"zero"` is decided by the baseline's zero-variance group proportion (see Section 7).
- Since full score is given within tolerance, **success ⇔ reward == 1.0**. The report must state plainly that the success rate is a thresholding of the reward, not independent evidence.

#### Return value (dict)

```python
{
  "reward": float,
  "success": bool,
  "category": str,        # see the table below, 9 categories in total
  "format_ok": bool,
  "hard_ok": bool,
  "reason": str,          # human-readable failure reason
  "soft": float | None,
  "components": {"contact_time": err, "restitution": err, "v_in": err, "resid": err},
  "diag": {
     "v_in": ..., "v_out": ..., "e_model": ..., "t_cross": ..., "tc": ...,
     "traj_err_mean": ...,      # mean |x_k - truth_k| / vdt
     "first_post_err": ...,     # |x - truth| / vdt at the first frame after contact; distinguishes the e=1 mirror reflection
  },
}
```

#### Failure categories (mutually exclusive, fixed order)

| key | Description |
|---|---|
| `format` | format failure |
| `teleport_out_static` | teleport / out of bounds / static |
| `penetration` | wall penetration |
| `no_reach_or_no_rebound` | did not reach the wall or did not rebound |
| `energy_gain` | energy gain |
| `contact_time` | contact-time deviation |
| `restitution` | rebound speed ratio deviation |
| `trajectory` | trajectory deviation (incoming speed or uniformity of motion) |
| `success` | success |

This order matches the `CATS` array in `report_mock.html`, so the report can be filled directly in this order.

### 3.3 `ballwall/baselines.py`

Each strategy is `f(task, rng) -> xs`, which is then rendered to text with `render_svg` and passed to the grader. Implement at least:

| Name | Method | Expectation |
|---|---|---|
| `oracle` | reference answer | 100% success, all rewards = 1.0 |
| `slight_e` | compute the rebound with `e × 1.04` | mostly successful (within the `e_rel` tolerance) |
| `jitter` | reference answer plus uniform noise ±0.03·vdt | mostly successful |
| `mirror_e1` | mirror reflection with e = 1 | almost never successful in-distribution; heldout tasks with e close to 1 may succeed |
| `fixed_e05` | turns around at the correct time, but e is fixed at 0.5 | succeeds only when the true e is near 0.5 |
| `early_turn` | rebounds with e after covering 60% of the distance | hard-constraint violation (did not reach the wall) |
| `small_penetration` | reference answer, but at the frame closest to the wall the ball's surface crosses the wall face by 5px | hard-constraint violation (wall penetration) |
| `pass_through` | `x0 + v·t`, keeps moving right | hard-constraint violation (wall penetration or teleport) |
| `static` | stays at x0 the whole time | hard-constraint violation (static) |
| `random` | random x in every frame | hard-constraint violation (teleport) |

### 3.4 `ballwall/metrics.py`

`summarize(results: list[dict]) -> dict`, outputs:
- `n`, `success_rate`, `success_ci95` (Wilson interval)
- `mean_reward`
- `format_fail_rate`
- `hard_violation_rate` (over all samples, excluding format failures)
- `category_dist`: proportions of the 9 categories, summing to 1
- Chain decomposition: `p_format_ok`, `p_hard_ok_given_format`, `p_success_given_hard` (record `null` when the denominator is 0)

---

## 4. Scripts

### 4.1 `scripts/make_data.py`

- Arguments: `--seed 0 --n_train 2000 --n_eval 200 --out data/`
- Output: `data/train.jsonl`, `data/eval_in.jsonl`, `data/eval_heldout.jsonl`
- The three sets use different random seeds. After generation, print a summary of the parameter ranges of each set.
- **The generated data is committed to the repo**, so the training machine and the evaluation machine use the same tasks.

### 4.2 `scripts/selftest.py` (everything must pass before RL)

1. For **every task** in the three datasets: `oracle` must be judged successful with reward == 1.0. On failure, print the task parameters and exit with a non-zero code.
2. On `eval_in` and `eval_heldout`, run all the strategies from Section 3.3 and print a table: strategy, success rate, mean reward, most common category.
3. Assertions:
   - The success rate of `early_turn`, `small_penetration`, `pass_through`, `static`, `random` is 0.
   - The mean reward of every violating strategy is lower than the mean reward of any strategy that "satisfies the hard constraints" (verifies the lexicographic ordering).
   - The success rate of `slight_e` and `jitter` is ≥ 80% (otherwise the tolerances are too strict).
4. For `mirror_e1` and `fixed_e05`, only print, do not assert; but if their success rate on `eval_in` exceeds 10%, print a warning: the tolerances may be too loose.
5. Parsing robustness cases (one each; all must be judged as format failures): a missing frame, duplicate frame id, `cx="880px"`, a `transform` attribute, an `<animate>` element, wall x inconsistent with the parameter, radius inconsistent with the parameter, an extra ball outside the frames, no svg in the output.
6. Write the table to `outputs/selftest.md` so it can be pasted into the report.

### 4.3 `scripts/evaluate.py`

- Arguments: `--model Qwen/Qwen2.5-1.5B-Instruct --adapter <optional, LoRA path> --data data/eval_in.jsonl data/eval_heldout.jsonl --limit <optional> --batch_size 8 --max_new_tokens 768 --violation_mode low --out outputs/eval/<name>.json --tag <checkpoint step>`
- Device: CUDA → `bfloat16`; MPS → `float32`; otherwise CPU.
- Generation: `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`, **left padding**, greedy decoding (`do_sample=False`).
- Loading LoRA: `PeftModel.from_pretrained(base, adapter)`.
- Score each sample with the grader; output:
  - `outputs/eval/<name>.json`: one `summarize` result per dataset, plus `tag`, model name, and time.
  - `outputs/eval/<name>.samples.jsonl`: the task, the model's raw output, and the scoring result for each item (for manual spot checks).
- Optional `--backend vllm` (CUDA only): speed up with vLLM + `LoRARequest`; not required.

### 4.4 `scripts/train_grpo.py`

**Data:** read `data/train.jsonl` and build an HF `Dataset` with two columns:
- `prompt`: the message list (system + user)
- `task_json`: `json.dumps(task)`

TRL passes `task_json` to the reward function as a keyword argument.

**Reward function:**
- `physics_reward(prompts, completions, task_json, **kw)`: **the only reward used for training**, returns `score(...)["reward"]`.
- Metric functions for logging only, with weight 0: `metric_success`, `metric_format_ok`, `metric_hard_ok`.
- Cache the result of `score` by `(completion_text, task_json)` with `functools.lru_cache` to avoid recomputation.
- Extracting the completion: in conversational format it is `completion[0]["content"]`; in plain-text format it is the string itself; both must be supported.

**Do not** split the reward into multiple functions for TRL to sum: that breaks the lexicographic ordering, and a violating answer could offset wall penetration with scores from other components.

**Model:**
- Default `Qwen/Qwen2.5-1.5B-Instruct`; when GPU memory is tight or on a Mac, use `Qwen/Qwen2.5-0.5B-Instruct`.
- If using Qwen3, disable thinking (otherwise it outputs reasoning text, violating setting A, "output only SVG"). If you are not sure how to disable it in TRL, do not use Qwen3.
- If `tokenizer.pad_token` is empty, set it to `eos_token`.

**LoRA:** `r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM"`

**GRPOConfig (starting values):**

| Parameter | CUDA | Mac (MPS) |
|---|---|---|
| `learning_rate` | 2e-5 | 2e-5 |
| `lr_scheduler_type` / `warmup_ratio` | `constant_with_warmup` / 0.05 | same as left |
| `num_generations` | 8 | 4 |
| `per_device_train_batch_size` | 16 | 4 |
| `gradient_accumulation_steps` | 4 | 4 |
| `max_prompt_length` | 512 | 512 |
| `max_completion_length` | 768 | 768 (16 frames need roughly 400 to 600 tokens; cannot go lower) |
| `temperature` | 1.0 | 1.0 |
| `beta` (KL coefficient) | 0.02 | 0.02 |
| `max_steps` | 60 (more if time allows) | 1-step smoke test first, then decide |
| `bf16` | True | False |
| `use_vllm` | optional True | False |
| `logging_steps` / `save_steps` | 1 / 20 | 1 / 10 |
| `reward_weights` | `[1, 0, 0, 0]` | same as left |
| `log_completions` | True (if supported) | same as left |
| `report_to` | `wandb` or `none` | `none` |

- The total number of completions per step (`per_device_train_batch_size × gradient_accumulation_steps × number of GPUs`) must be divisible by `num_generations`.
- **TRL version compatibility**: `GRPOConfig` parameter names differ across versions (e.g. `max_prompt_length`, the mode parameter of `use_vllm`, `reward_weights`, `log_completions`). At implementation time, first obtain the supported parameter names with `dataclasses.fields(GRPOConfig)`, filter out unsupported ones, and print a warning. If `reward_weights` is not supported, pass only `physics_reward` and have `evaluate.py` compute the metrics instead.
- Command-line arguments: `--model --out --max_steps --num_generations --lr --beta --violation_mode --use_vllm`; everything else uses the defaults in the table above.
- Save the LoRA to `<out>/final` at the end of training; intermediate checkpoints are saved according to `save_steps`.
- Before training, print: device, dtype, TRL version, number of tasks and completions per step.

### 4.5 `scripts/collect_curve.py`

- Read `outputs/eval/*.json` and sort by `tag` (training step).
- Output `outputs/curve.json`:

```json
{
  "steps": [0, 20, 40, 60],
  "eval_in":      {"success": [...], "ci_low": [...], "ci_high": [...], "mean_reward": [...], "category_dist": [[9 proportions], ...]},
  "eval_heldout": {"...same as above": "..."}
}
```

- Also export the online metrics from the training log (`reward`, `reward_std`, zero-variance group proportion, `metric_success`, etc., if available) to `outputs/train_log.json`.

---

## 5. Metrics to Monitor During Training

| Metric | Healthy behavior | What to do when abnormal |
|---|---|---|
| mean `reward` | increasing | flat: print samples, check the grader and the format first |
| `reward_std` | stays above 0 | approaches 0: no within-group differences, no gradient |
| zero-variance group proportion (if TRL has `frac_reward_zero_std`) | below 30% to 50% | too high: add few-shot examples, or change `violation_mode` |
| `metric_success` (online) | increasing | reward rises but this does not: look at the failure categories |
| `kl` | grows slowly | above 0.5: lower the learning rate |
| completion length | stable at 400 to 700 | truncated (hits the limit): format will fail, check `max_completion_length` |
| loss | rises from around 0 | normal; not used to judge progress |

Every 10 steps, manually inspect 2 generated samples to check for reward hacking.

---

## 6. Execution Order and Acceptance

| Step | Content | Acceptance criteria | Needs GPU |
|---|---|---|---|
| 1 | `task.py`, `checker.py`, `baselines.py`, `metrics.py` | importable; `selftest.py` runs | No |
| 2 | `make_data.py` generates data | three jsonl files generated; parameter ranges match Section 3.1 | No |
| 3 | `selftest.py` | all assertions in Section 4.2 pass; `outputs/selftest.md` generated | No |
| 4 | `evaluate.py` on the baseline (step 0) | results for 200 tasks in each of the two evaluation sets; record the format failure rate and zero-variance group proportion | Yes |
| 5 | Decide `violation_mode` from the baseline; adjust few-shot if needed | decision recorded in `NOTES.md` | No |
| 6 | `train_grpo.py` 1-step smoke test | no errors; record time per step | Yes |
| 7 | Full training | run as many steps as time allows, save checkpoints | Yes |
| 8 | Run `evaluate.py` on each checkpoint, then `collect_curve.py` | `outputs/curve.json` generated | Yes |

**If step 3 does not pass, do not start any work from step 4 onward.**

How to measure the baseline's zero-variance group proportion: take 50 training tasks, sample `num_generations` completions per task at temperature 1.0, score them, and count the proportion of groups whose rewards are all identical. This can be an option of `evaluate.py` (`--sample_k 8`).

---

## 7. Open Decisions (implement as switches, do not hard-code)

1. **`violation_mode`**: `low` (violations get 0.2×soft) or `zero` (violations get 0). Rule: if the baseline's zero-variance group proportion is below 30%, use `zero`; above 50%, use `low`; in between, either is fine, but record the choice.
2. **Tolerances**: start with the defaults; if the success rate of `slight_e` or `jitter` in the self-check is below 80%, loosen them; if the dumb baselines succeed too often, tighten them. Record changes in `NOTES.md`.
3. **Model size**: determined by the training machine's GPU memory and the per-step time measured in step 6.

---

## 8. Things Not to Do

- Do not read the wall position and ball radius from the SVG for scoring (the model could draw the wall farther away or the ball smaller to evade wall penetration).
- Do not split the reward into multiple functions for TRL to sum.
- Do not use reference-answer trajectories as SFT data (that would teach the model the answer directly); for cold start, use at most few-shot examples or teach only the format.
- Do not use "the frame closest to the wall" as the contact time; use the intersection of the two fitted lines.
- Do not use sampling decoding during evaluation; always use greedy decoding.
- Do not modify the committed tasks in `data/`; if new tasks are needed, change the seed and save them as a new file.

---

## 9. Integration with the Report

`report_mock.html` currently contains only placeholder data. Once real results are available:
- Fill Figure 1, Figure 2, and Table 1 with the `summarize` results from `outputs/eval/*.json`.
- Figure 3 and Figure 4 are currently drawn by interpolating `RL.start` / `RL.end`; change them to read the per-checkpoint data from `outputs/curve.json` directly.
- Replace the hand-computed estimates in Table 2 (grader self-check) and Table 3 (dumb baselines) with the actual results from `outputs/selftest.md`.
- Before removing the "sample data" banner at the top of the page, confirm that every number comes from actual results.
