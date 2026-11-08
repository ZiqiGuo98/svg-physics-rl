# Project Plan: Ball Hits Wall — Evaluation and RL of an LLM Drawing SVG Frame by Frame (v2)

> v2 incorporates: difficulty tiers L0–L4 have been removed; the reward is now two-layered ("hard constraints + soft accuracy"); success rate and reward are defined separately; and the review comments from one Opus sub-agent pass have been merged in. Items marked "unverified" should be checked against the original source before being cited.
> Time remaining: about 2 hours (hackathon).

## 1. One Sentence

The model **writes no code and calls no physics engine**; it directly outputs a sequence of static SVG frames depicting a "ball hits wall" scene. The grader (which may use the analytical solution) extracts the ball's coordinates by id, first checks the hard constraints, then compares soft accuracy; the same score serves as a verifiable reward for LoRA + GRPO on a small Qwen model. The final deliverable is a plot of "success rate rising with training steps".

## 2. Task

A ball starts from the left, moves right at constant horizontal velocity, hits a wall on the right, and bounces back with coefficient of restitution e. Draw 16 frames.

- The problem gives: v, the x of the wall's inner face, ball radius r, e, Δt, canvas scale.
- Each problem randomizes v, e, wall position, r.
- Generator constraints: e ∈ [0.3, 0.95]; ≥4 frames both before and after contact; the end point after the bounce stays inside the canvas; the moment of impact does not fall within ±0.1Δt of a sampled frame (cases where it does land on a frame are tracked as a separate group).
- Reference answer: before the impact `x = x₀ + v·t`; when the ball's surface touches the wall the velocity becomes `−e·v`.
- A "variant" with gravity and a floor needs a separate generator and grader; **not done this time**.

  
### Scene 1: Rubber Ball Bouncing — Tests Energy Decay

**Prompt**
> Canvas 400×400. A rubber ball of radius 20 is released from rest at the top of the canvas (center y=20), falls vertically to the floor (y=400), and bounces 4 times, each bounce peaking lower than the previous one. No horizontal motion.

**Fixed parameters**: r=20, cx=200, initial cy=20, floor y=400, coefficient of restitution e ≈ 0.8

**Checks**
- Hard constraints: the ball does not pass through the floor (cy + r ≤ 400); no bounce peak is higher than the starting height; x does not drift; no teleport
- Soft accuracy: the ratio of consecutive bounce heights ≈ e², and strictly decreasing

**Common failures**: bounces get higher each time, every bounce reaches the same height, the ball sinks into the floor after landing, bounces once and then stops

---

### Scene 2: Ball Hits Wall — Tests Wall Penetration and Rebound

**Prompt**
> Canvas 400×200. A ball of radius 20, center starting at (60, 100), rolls right at a constant 100 px/s, hits a solid brick wall at x=300 on the right, and rolls back left at 0.8 times the speed. No gravity, no friction.

**Fixed parameters**: r=20, cy=100, x₀=60, v=100 px/s, wall face x=300, e=0.8

**Checks**
- Hard constraints: at every moment cx + r ≤ 300 (no wall penetration); the ball must reach the wall before turning around; it must reverse direction after hitting the wall; rebound speed ≤ incoming speed
- Soft accuracy: contact time ≈ 2.2 s; rebound speed / incoming speed ≈ 0.8

**Common failures**: passes straight through the wall, turns around before reaching the wall, stops at the wall without bouncing, rebounds faster than it came in

---

### Scene 3: Wooden Block Splits in Two — Tests Momentum Symmetry

**Prompt**
> Canvas 400×200. A 60×60 brown wooden block rests at the center of the frame (top-left corner at (170, 70)). At t = 1 s the block splits into left and right fragments of 30×60 each; the left fragment slides left and the right fragment slides right, each at a constant 60 px/s,
until the end of t = 3 s. No gravity; the fragments do not rotate.

**Fixed parameters**: block 60×60 @ (170, 70), fragments 30×60, split time t=1 s, v=60 px/s

**Checks**
- Hard constraints: the centers of the two fragments are always symmetric about x=200 (total momentum = 0); y unchanged; the fragments do not pass through each other; only the block exists before the split and only the fragments after it
- Soft accuracy: split time ≈ 1 s; fragment speed ≈ 60 px/s

**Common failures**: the two fragments have unequal speeds, both fly to the same side, the original block is still present after the split, the fragments are displayed from the very start, the fragments fly toward the center

---

### Scene 4: Two Balls Collide — Tests Momentum Transfer

**Prompt**
> Canvas 400×200. Two identical red balls, radius 20. The left ball's center starts at (60, 100) and moves right at a constant 100 px/s; the right ball's center is at rest at (260, 100). At t = 1.6
s the two balls touch (the left ball's center reaches (220, 100), center distance 40). After the collision the left ball stops immediately and the right ball moves right at a constant 100 px/s until the end of t = 3 s. No gravity, no friction.

**Fixed parameters**: r=20, cy=100, left ball x₀=60, right ball x₀=260, v=100 px/s, contact time t=1.6 s

**Checks**
- Hard constraints: center distance ≥ 40 (no overlap); the right ball does not move before contact; the left ball does not move after contact; the right ball's speed ≤ the left ball's incoming speed
- Soft accuracy: contact time ≈ 1.6 s; right ball speed after collision ≈ 100 px/s

**Common failures**: the two balls overlap and pass through each other, the right ball starts early, the left ball keeps following after the collision, the left ball stops before touching, the right ball is faster than the incoming ball

---

### Scoring Rules (shared by all scenes)

| Case | Score |
|---|---|
| Format error (specified id not found, values with units, forbidden constructs beyond animate/script) | 0 |
| Any hard constraint violated | 0.2 × soft accuracy |
| All hard constraints pass | 0.3 + 0.7 × soft accuracy |

Each soft-accuracy item has a tolerance band; full marks within the tolerance. Success = score exactly 1.0.
Only the highest-priority failure category is reported: format → teleport/out-of-bounds/stationary → hard constraints → soft accuracy.


## 3. Output Format and Parsing

- A single text containing 16 `<g id="f0">…<g id="f15">` groups sharing one `<defs>`, roughly 1–2k tokens.
- Canvas `viewBox="0 0 1000 1000"`; the prompt gives the scale and the y-axis direction.
- `<animate>`, `<script>`, `transform`, and `<use>` are forbidden; no formulas and no trajectory are given.
- **Parsing allowlist**: only `cx/cy/r` of `circle#ball` are recognized; duplicate ids, an `ellipse/path` posing as the ball, and values with units (e.g. `880px`) are all format failures.
- **Coordinate convention**: the ball is represented by its center cx; the problem gives the x of the wall's inner face; contact occurs when the distance from the center to the wall face = r.
- **The wall position and ball radius are always taken from the problem parameters**; the grader does not read the values in the SVG; SVG values inconsistent with the parameters (±1px) count as a format failure; the ball's y should be constant. (Review comment C1: otherwise the model could draw the wall at x=2000 or shrink r to 1px to evade the wall-penetration check.)

## 4. Estimators on Discrete Frames (Review Comment C3)

Take v=400 px/s, Δt=0.25 as an example: the impact time 1.95 s falls between frames 7 and 8 (x=800 and x=868), and no frame is exactly touching the wall. Therefore:

- Incoming speed: least-squares line fit over the frames before contact; take the slope.
- Rebound speed: the same fit over the frames after contact.
- The model's contact time: the intersection of the two fitted lines with `x = x_wall − r`. **Do not use "the frame closest to the wall".**
- Trajectory error: per-frame |x − x*|, normalized by v·Δt.
- Diagnostic: "position error of the first frame after the rebound", used to distinguish e=1 mirror reflection from a genuine rebound with e.
- Do not force the impact time onto the sampling grid (aligning it would remove the fractional-frame cases that are the most discriminative).

## 5. Reward: Two-Layer Lexicographic

```
Bad format:                      reward = 0
Any hard constraint violated:    reward = 0.2 × soft        (at most 0.2)
All hard constraints satisfied:  reward = 0.3 + 0.7 × soft  (at least 0.3)
```

**Hard constraints** (physically unacceptable):
- Wall penetration: in any frame the ball's surface crosses the wall face by more than the tolerance (about 3 px).
- **Turning around before reaching the wall**: the maximum x is smaller than the reference answer's maximum x on the sampled frames by more than 0.5·v·Δt (review comment C2: otherwise a conservative strategy such as "turn around early" reliably earns ≥0.3, scoring higher than "almost correct but penetrates the wall by 4px").
- No direction reversal after hitting the wall, judged by the direction of **its own trajectory** (not by the ground-truth contact time).
- Rebound speed greater than 1.05 times the incoming speed.
- Inter-frame displacement exceeding 1.5·v·Δt, y drift, leaving the canvas, stationary throughout.

**Soft accuracy** (has error but is physically plausible): impact time deviation, deviation of the rebound speed ratio from e, trajectory deviation. Each item has a **tolerance band** (full marks within the tolerance, decay only beyond it); soft is the geometric mean or the minimum (aligned with the success definition "all within tolerance"; no arithmetic mean).

**Order for setting tolerances**: first fix the physical tolerances (10% relative error on e, ±0.25Δt on contact time), then derive the trajectory tolerance from them, to avoid contradictions (review comment M2: e=0.65 vs 0.6 differs by only 20 px/s in rebound speed, but after 1.8 s the position differs by about 36 px, which may exceed an independently set trajectory tolerance).

Purpose of the two-layer structure: any answer satisfying all hard constraints scores higher than any answer violating one; among violating answers there is still a 0–0.2 gradient.

## 6. Success and Failure Categories

**Success** (fixed before training): no hard constraint violation, and impact time, rebound speed ratio, and trajectory are all within tolerance.

**Important note**: because full marks are given within tolerance, success is equivalent to reward exactly = 1.0. Success rate is a thresholding of the reward, **not independent evidence**. There are no samples with reward between 0.2 and 0.3; this is a construct of the formula, not a finding. Independent validation requires additional human/VLM spot checks of the videos, or a second independently implemented grader.

**Failure categories** (mutually exclusive, judged by priority):
1. Format failure
2. Teleport/out-of-bounds/stationary (placed early so that erratic outputs are not misclassified as "energy increase")
3. Wall penetration
4. Did not reach the wall or did not rebound
5. Energy increase
6. Impact time deviation
7. Rebound speed ratio deviation
8. Trajectory deviation
9. Success

## 7. Training Model and Environment

| Item | Setting |
|---|---|
| Model being trained | Small Qwen instruct model, 1.5B–4B class (e.g. Qwen2.5-1.5B/3B-Instruct, Qwen3-1.7B/4B, depending on downloadable versions and GPU memory) |
| Fine-tuning method | LoRA + GRPO (TRL `GRPOTrainer` or Unsloth), no full-parameter tuning |
| Sampling | G=8 per problem, temperature≈1.0, maximum output about 2k tokens |
| Inference setting | Default setting A: output SVG only (Qwen3 with thinking disabled); setting B (reasoning allowed first) as a later comparison |
| Cold start | Teach the format only (few-shot or a small amount of format SFT); trajectory values are not taken from the reference answer, to avoid leakage |
| Evaluation set | Fixed ≥200 problems, one in-distribution set and one with held-out parameters; evaluate with greedy decoding every 10–20 steps; draw confidence intervals on the plot |
| Scale | On the order of a few dozen steps within the 2-hour window |
| Prerequisite | GPU and TRL/verl/Unsloth environment already ready; **if the environment has to be built from scratch, it will most likely not be done in time**, in which case fall back to plotting "rejection-sampling rounds vs. success rate" and label it honestly as not RL |

GRPO caveats (review comment M5):
- When an entire group fails on format, advantage=0 and there is no signal. Countermeasure: give a complete format few-shot in the prompt; monitor the "fraction of zero-variance groups".
- Reward is flat within tolerance: once the success rate is high, accuracy no longer improves; report this honestly.
- The success-rate curve must state its measurement basis: online sampling on the training batch (noisy), or greedy decoding on a fixed evaluation set. With 100 problems the 95% confidence interval is about ±10 pt, so differences of a few points cannot be distinguished.

## 8. Grader Self-Check (must pass before RL)

1. **Reference answer self-check (review comment C4)**: for every problem the generator produces, run the reference answer through the grader and assert reward = 1 and a success verdict. Problems this exposes: with large v or large e the ball leaves the canvas after the rebound; with very small e it is judged "stationary"; with e=1 rounding errors trigger "energy increase"; a fixed-pixel teleport threshold wrongly penalizes correct answers.
2. **Hand-written wrong answers**: e=1 mirror reflection, final frame penetrating the wall by 4 px, passing straight through the wall, turning around early, rebounding faster, stationary ball. Confirm the score ordering matches expectations.
3. **Dumb baselines (fixed strategies without a model)**: always mirror with e=1; turn around at the correct time but with a fixed e=0.5; turn around early; stationary ball; wall penetration; random positions. The goal is to show that the grader is discriminative and has no exploitable loopholes. If some dumb baseline is counted as "success" in large numbers, fix the tolerances before training.
4. If this step does not pass, **do not start RL**.

## 9. Evaluation and Reporting

- Held-out parameters (e, speed outside the training range) serve as the generalization evaluation; the held-out variant (gravity + floor) is not done this time and is written up as "next step".
- Report success rates with confidence intervals and give the chain decomposition: P(format pass) × P(hard constraints pass | format pass) × P(success | hard constraints pass). Phrase it as "consistent with having learned the physics", not "proves" (the denominators of the conditional probabilities are different sample sets before and after RL, so there is selection bias).
- Plot three kinds of curves together: success rate, mean reward, and a stacked area of failure categories over training steps. When reward rises but success rate does not, look at the stacked plot to find the cause.
- Framing of the conclusion: the problem gives v, e, x₀, Δt, and the answer is a closed-form formula. A rising RL curve can only show "learned to correctly execute collision kinematics in this format", not "improved physical understanding".

## 10. Related Work and Positioning (partly unverified)

| Work | Content | Status |
|---|---|---|
| SimuScene (2602.10840) | Model writes code; VLM watches videos to judge qualitative questions; GRPO | Read part of the main text |
| VisPhyWorld (2602.13294) | Two key frames → write simulation code; compares four rendering backends: Three.js / P5.js / SVG / Manim | **Main text verified** (including the end of the appendix): SVG and Manim appear only in the qualitative examples of Figure 1 and in Section 3.4, where they are called "script-based backends". The paper **does not state** whether the model writes code that generates SVG or writes SVG markup directly; there are no quantitative SVG results, and the main experiments evaluate only Three.js and P5.js. The wording leans toward "writes code", but this is unconfirmed, so it cannot be used to judge whether "drawing SVG directly" has been done before |
| PhySimCode (2610.15722) | Video → infer simulation code | Read part of the main text |
| PhysBreak (GitHub) | 5 problem families, numerical verification, grader self-check | Read README-level information; no license declaration |
| RL with Verifiable Physics (2607.10474) | GRPO + physics verifier, output is solver code | **Search summary only**, possibly the closest work, must be verified first |
| SVG-LLMs multi-reward RL (2603.16189), GAUGE (2608.05948) | SVG generation RL / physical fidelity of video world models | **Search summary only** |

Positioning: a small-scale feasibility demonstration of RLVR with a deterministic grader in a no-code, direct state-output setting. **No claim of "first".**
Value of SVG: in a one-dimensional scene SVG is just a wrapper around coordinates, equivalent to outputting a list of coordinates (e.g. `{"ball_x":[…]}`). The defensible value is that it can be directly synthesized into a video for human/VLM cross-validation, and it can be extended to multiple objects.
**Open question (limitation, recorded as a next step)**: the difference between SVG and a coordinate list (JSON) has **not been tested in a controlled experiment**. Design: same problems, same grader, same model, change only the output format, and compare success rate and format failure rate. If the two groups are close, SVG adds no extra difficulty and its value lies only in visualization; if the SVG group is clearly worse, "drawing" itself adds difficulty.

## 11. Schedule (about 2 hours remaining)

| Time | Task |
|---|---|
| 0–35 min | Write the grader and generator: wall/radius from problem parameters, fitted estimators, the "must reach the wall" constraint, teleport threshold scaled by v·Δt, generator parameter constraints |
| 35–50 min | Self-check: reference answer reward=1, hand-written wrong answers, dumb baselines |
| 50–60 min | Small-model baseline: 200-problem evaluation set. If the format failure rate is >50%, add few-shot; no SFT |
| 60–105 min | GRPO (0.5–1.5B/small Qwen model, G=8, a few dozen steps); record evaluation-set success rate, fraction of zero-variance groups, failure category proportions |
| Last 15 min | Replace the placeholder data in the sample report with real data; verify related work |

## 12. Likely Questions and Answers

**Q1. Does this test physical understanding, or mental arithmetic / kinematics execution?**
Mostly the latter. The problem gives all parameters and the answer is a closed-form formula, so we only claim "whether the model can correctly execute collision kinematics in a no-code, direct state-output setting". RL gains can only be interpreted the same way.

**Q2. Why not let the model write code?**
Writing code outsources the computation to an interpreter, so the source cannot be told apart. This is a diagnostic setting, not an application setting; the code path is already covered by work such as SimuScene.

**Q3. SimuScene already did physics plus RL; what is new here?**
We do not claim "first physics plus RL". Differences: the output is per-frame state rather than code; the reward comes from a deterministic grader and does not depend on a noisy VLM (whose human agreement rate is about 88%, at about 579 seconds per step — from the paper, of which I read only part of the main text). This is a different setting, not "better".

**Q4. In a one-dimensional scene, what is the difference between SVG and outputting a coordinate list?**
Essentially none; this is a limitation. We acknowledge no controlled experiment was done; it is the next step: same problems, same grader, only swap the output for a coordinate list, and compare success rate and format failure rate. The value of SVG is that it can be rendered directly into a video for cross-validation and extended to multiple objects.

**Q5. Aren't success rate and reward the same thing?**
In our design, essentially yes: full marks within tolerance, so success ⇔ reward=1.0. Success rate is a thresholded metric for readability, not independent evidence. Independent validation relies on spot-checking videos or a second grader.

**Q6. Could the model exploit the grader?**
Known loopholes and countermeasures: changing the wall position/radius (taken from problem parameters); turning around early without touching the wall (hard constraint that the wall must be reached); wall penetration handled as a hard constraint rather than a weighted deduction; parsing allowlist. Dumb baselines are used for ongoing checks.

**Q7. Why is wall penetration a hard constraint while speed deviation is soft?**
Wall penetration is physically impossible; speed deviation is merely imprecise. If they were put into the same weighted sum, the model could offset wall penetration with high scores on other items. Hence the lexicographic order.

**Q8. Why a small Qwen model?**
Within 2 hours only LoRA + GRPO is feasible, and the 1.5B–4B class is workable on a single GPU. The exact model depends on downloadable versions and GPU memory.

**Q9. Can this generalize to real video or complex scenes?**
No. The scene is idealized (single ball, rigid wall, no noise).

**Q10. Why evaluate only held-out parameters and not gravity + floor?**
That is a different scene; the reference trajectory and hard constraints both differ, so it needs a separate generator and grader, which cannot be done in 2 hours. It is a next step.

## 13. To Do

- [ ] Confirm GPU/environment; decide between RL and rejection sampling
- [ ] Write the grader and generator (Sections 3–6)
- [ ] Three-step self-check (Section 8); proceed to training only after it passes
- [ ] Small-model baseline; add few-shot if the format failure rate is >50%
- [ ] GRPO training and record the three kinds of curves
- [ ] Replace `report_mock.html` with real data
- [ ] Verify 2607.10474, 2603.16189, 2608.05948 (VisPhyWorld's SVG backend already verified: the main text does not state the output form)
- [ ] (Next step) SVG vs. coordinate-list controlled experiment
- [ ] Contact the PhysBreak authors about the license before reusing code
- [ ] Scene library v1 (Scenes 1–4 in Section 2): first wire the fixed Scene 2 problem into the grader, then add Scene 4 (two balls) and Scene 3 (split); Scene 1 (gravity) spec only, not implemented

## 14. Reference Links

- SimuScene: https://arxiv.org/html/2602.10840v2
- VisPhyWorld: https://arxiv.org/html/2602.13294v2
- PhySimCode: https://arxiv.org/html/2610.15722v1
- PhysBreak: https://github.com/ashifirahman9999-lab/physbreak
- PhysCodeBench: https://arxiv.org/html/2604.23580
- RL with Verifiable Physics: https://arxiv.org/html/2607.10474v1
- SVG-LLMs multi-reward RL: https://arxiv.org/html/2603.16189
- GAUGE: https://arxiv.org/pdf/2608.05948
