# Implementation Notes (deviations, tuning, actual environment)

As required by section 0 of `IMPLEMENTATION_PLAN.md`, all deviations and parameter tuning are recorded here. Date: 2026-11-07.

## Actual environment

| Item | Value |
|---|---|
| Machine | Windows 11 laptop, NVIDIA RTX 3050 Ti Laptop GPU (4 GB VRAM), i7-11370H, 32 GB RAM |
| Python | 3.14.0 (parent directory `aie-hackathon/.venv`, not the 3.10/3.11 suggested by the plan; all required wheels install fine) |
| Library versions | torch 2.9.1+cu128, transformers 5.22.2, trl 1.17.1, peft 0.21.2 |
| vLLM | Not installed (Windows) |

The core module `ballwall/` uses only the standard library; `python -I scripts/selftest.py` also runs with the system Python.

## Steps 1–3: generator, grader, self-check

- Data: `python scripts/make_data.py` (seed 0; train 2000, eval_in 200, eval_heldout 200; the three sets use seeds 1, 2, and 3 respectively). Parameter ranges match plan section 3.1: contact frame 4.30–10.70, fractional part 0.10–0.90.
- Tasks gained an extra `id` field (e.g. `eval_in-00007`) for convenient spot checks; grading does not use it.
- All self-checks pass; results are in `outputs/selftest.md`.
- One warning, tolerance left unchanged: `fixed_e05` has a 12.5% success rate on eval_in, above the 10% alert line set in the plan. The cause is not a bug: the 8% coefficient-of-restitution tolerance means that for tasks whose true e falls in 0.463–0.543, a fixed guess of 0.5 is judged a success; that interval covers about 15% of the in-distribution range [0.30, 0.85]. Tightening `e_rel` would make `slight_e` (e×1.04) start failing, so the default is kept. Likewise, `mirror_e1` succeeding 25.5% on held-out is because tasks with e ≥ 0.926 are within 8% of e=1, which the plan already anticipated.
- Extra parsing test cases added: scientific notation `1e2` and a truncated SVG (no `</svg>`) are both judged format failures; a correct answer wrapped in a ```` ```svg ```` code block is still judged a success.
- Rule for identifying the ball in a frame: `class`, split on whitespace, contains `ball` (not substring matching).

## Implementation details of the evaluation script

- Greedy decoding, left padding, `stop_strings=["</svg>"]`: generation stops at the first `</svg>`. The grader only reads the first svg element anyway, so this only saves time and does not change the score. During training, TRL is not passed this parameter.
- For LoRA evaluation, `merge_and_unload()` is called before generation to speed it up.
- The `--sample_k` probe computes the fraction of zero-variance groups under both the `low` and `zero` violation_mode, for the decision in section 7.
- `collect_curve.py` collects only `outputs/eval/step*.json`, to avoid drawing one-off evaluations of other models into the training curve.

## Step 4 onward

See the records appended below.
