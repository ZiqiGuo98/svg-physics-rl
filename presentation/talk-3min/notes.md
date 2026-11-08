# Speaker notes: 3-minute talk

About 350 words, roughly 3 minutes. The deck has 8 slides.

**1. Title (10 s)**
Hi everyone. Can recursive self-improvement, RSI, improve a language model's understanding of physics? That is our question, and we test it on a ball and a wall.

**2. Background (20 s)**
Something changed with GPT-6: creators are already using it to make videos. This YouTube tutorial says GPT-6 Astra can bulk-create videos. [Point at the screenshot.] People are using these models as video generators, so how physically correct the output is starts to matter.

**3. The problem (25 s)** — play clips
But the motion often breaks the physics. Here are four ways. Newton's laws: a ball drifts sideways though no force acts on it. Energy conservation: each bounce goes higher than the last. Contact: the ball stops at the wall instead of coming back. Momentum: the ball that hits keeps moving after the collision.

**4. Our idea (30 s)**
So we put an agent in the improvement loop. The agent, Claude, a stronger model, builds the training environment, generates the data, and tunes the reward. It fine-tunes the model automatically from the results, and explores on its own to raise the score. The model draws a scene, the agent finds its failures, writes data and rewards that target them, we train, and the loop repeats from the new model's failures.

**5. Model and training (20 s)**
The model is a small Qwen instruct model, about one and a half billion parameters. We train it with GRPO and LoRA using Hugging Face TRL. The reward is a physics score from a checker, written and revised by the agent each round, on scenes the agent generates.

**6. Correct examples (20 s)** — play clips
Four scenarios, each with a correct reference. A ball hits a wall and rebounds at 0.8 of its speed. A ball bounces and each bounce is lower. A block splits in two with total momentum staying zero. Two balls collide and the momentum passes on.

**7. Failure modes (25 s)** — play clips
And here are typical ways it goes wrong: the ball passes through the wall, turns back before touching it, or comes back faster than it went in. Each one is a rule in our checker and a term in the reward, so the model is pushed away from exactly these mistakes.

**8. Results (20 s)**
After training, success on scenes the model never saw goes from [base] to [after]. Success means no physics rule is broken and contact time, bounce speed, and trajectory are all accurate. With limited resources, a stronger model, Claude, did this for a smaller one, Qwen; the same loop is what a model could run on itself. Thank you.

---

## If asked: "is this recursive self-improvement?"

Not yet. Today a stronger model improves a weaker one, because of resource limits. What carries over is the loop itself: find failures, write targeted data and rewards, train, repeat. The claim is an automated loop that improves a model's physics ability, and a clear path to self-improvement. Do not say it "achieves RSI".

## If asked: "does spatial understanding really improve?"

Be careful here. The examples on slide 2 are demos and user reports, not benchmark results. Academic spatial-reasoning benchmarks still show clear weaknesses (for example, arXiv 2603.26779 finds frontier models reach at most 62.5% on a 3D rotation imagery task). So say "spatial output is getting impressive" and not "spatial understanding is solved". That also supports the story: looking spatially right does not mean obeying physics.

## Source behind slide 2

- YouTube result: "NEW ChatGPT Astra Can Make VIRAL Faceless Videos (FAST & UNLIMITED)" by Success With Sam, 3.4K views. It is a creator's own claim, and the channel promotes a product, so say "a tutorial says", not "GPT-6 can".
- Put your screenshot at `media/background_1.png`.

## Evidence to prepare before presenting

1. A round-by-round table: failure the agent found, reward change it made, effect on success rate. This is how you show the agent actually discovered something.
2. A record of what the human did and did not do. If you edited rewards by hand, say so.
3. At least two rounds. One round is a single fix, not a loop.

## Before presenting

- Slides 6 and 7 videos are the reference and single-error examples from `videos/` (made by miajin), not model outputs and not failures the agent discovered. Do not say "the agent found these" unless it is true. If you have real failing model outputs, swap them in.
- Training had not finished when I last looked (NOTES/commit: step 1 of the 0.5B run had 0 of 32 format-valid outputs). The chart and numbers on slide 8 are fake until a run produces real ones.

- Slide 3: the four clips are constructed examples of each violation (from `videos/`), not outputs of a real generator. Say "examples of violations". If you have a real generated video that breaks physics, swap it into `media/viol_1.mp4`.
- Slides 5 and 8: edit to the actual model, steps, and test-set size if they differ.
- Slide 8: fill `window.RESULTS` at the bottom of `index.html`.
