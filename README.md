# Ball Hits Wall: Physics Evaluation and RL for LLMs Drawing SVG Frame by Frame

The model writes no code and calls no physics engine; it directly outputs a sequence of static SVG frames depicting a "ball hitting a wall". The grader parses coordinates by id, first checks hard constraints (wall penetration, etc.), then compares soft accuracy. The same score serves as a verifiable reward for LoRA + GRPO training of a small Qwen model.

## Files

| File | Contents |
|---|---|
| `plan_physics_svg_frames.md` | Full plan (tasks, grader, reward, training setup, schedule, likely questions) |
| `report_mock.html` | Results report template. **All numbers are made-up placeholder data**, used only to preview what the final report will look like; they are not results of any experiment |

## Status

- Grader, task generator, and training scripts: **implemented** (`ballwall/`, `scripts/`, `data/`).
- For several related works only the abstract was read (see plan section 10); verify before citing.
- Open question: the controlled comparison between SVG and coordinate lists (JSON) has not been done.

## Viewing the report

Open `report_mock.html` directly in a browser; no build step needed.
