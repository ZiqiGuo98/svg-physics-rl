"""TRL GRPOTrainer + LoRA with the physics checker as the only reward (IMPLEMENTATION_PLAN.md 4.4).

Usage:
  python scripts/train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --out outputs/grpo/run1 --max_steps 60
Small-GPU knobs (not in the plan's CLI, defaults follow the plan's CUDA column):
  --per_device_bs 16 --grad_accum 4 --max_completion_length 768 --save_steps 20
One optimizer step consumes per_device_bs * grad_accum completions = (that / num_generations) prompts.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from ballwall.checker import score  # noqa: E402
from ballwall.task import read_jsonl  # noqa: E402

VIOLATION_MODE = "low"


def _text(completion) -> str:
    """Conversational completions are [{'role': 'assistant', 'content': ...}]; plain ones are strings."""
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return completion or ""


@functools.lru_cache(maxsize=200000)
def _cached_score(text: str, task_json: str, mode: str) -> dict:
    return score(text, json.loads(task_json), mode)


def physics_reward(prompts, completions, task_json, **kw):
    return [_cached_score(_text(c), tj, VIOLATION_MODE)["reward"] for c, tj in zip(completions, task_json)]


def metric_success(prompts, completions, task_json, **kw):
    return [float(_cached_score(_text(c), tj, VIOLATION_MODE)["success"]) for c, tj in zip(completions, task_json)]


def metric_format_ok(prompts, completions, task_json, **kw):
    return [float(_cached_score(_text(c), tj, VIOLATION_MODE)["format_ok"]) for c, tj in zip(completions, task_json)]


def metric_hard_ok(prompts, completions, task_json, **kw):
    return [float(_cached_score(_text(c), tj, VIOLATION_MODE)["hard_ok"]) for c, tj in zip(completions, task_json)]


def main() -> None:
    global VIOLATION_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "outputs/grpo/run1"))
    ap.add_argument("--data", default=str(ROOT / "data/train.jsonl"))
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--num_generations", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--violation_mode", default="low", choices=["low", "zero"])
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--per_device_bs", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=768)
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    VIOLATION_MODE = args.violation_mode

    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    if torch.cuda.is_available():
        device, dtype, bf16 = "cuda", torch.bfloat16, True
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, dtype, bf16 = "mps", torch.float32, False
    else:
        device, dtype, bf16 = "cpu", torch.float32, False
    on_cuda = device == "cuda"
    G = args.num_generations or (8 if on_cuda else 4)
    bs = args.per_device_bs or (16 if on_cuda else 4)
    save_steps = args.save_steps or (20 if on_cuda else 10)
    if (bs * args.grad_accum) % G:
        sys.exit(f"per_device_bs * grad_accum = {bs * args.grad_accum} must be divisible by num_generations = {G}")

    rows = read_jsonl(args.data)
    ds = Dataset.from_list([{"prompt": r["prompt"], "task_json": json.dumps(r["task"], sort_keys=True)} for r in rows])

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    wanted = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=max(1, round(0.05 * args.max_steps)),  # plan: warmup_ratio 0.05 (gone from transformers 5)
        num_generations=G,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=args.grad_accum,
        max_prompt_length=512,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=args.beta,
        max_steps=args.max_steps,
        bf16=bf16,
        use_vllm=args.use_vllm,
        logging_steps=1,
        save_steps=save_steps,
        save_only_model=True,
        reward_weights=[1.0, 0.0, 0.0, 0.0],
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        gradient_checkpointing=True,
        seed=args.seed,
    )
    supported = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in wanted if k not in supported)
    for k in dropped:
        print(f"WARNING: GRPOConfig in TRL {trl.__version__} has no '{k}'; dropped")
    cfg = GRPOConfig(**{k: v for k, v in wanted.items() if k in supported})

    if "reward_weights" in supported:
        reward_funcs = [physics_reward, metric_success, metric_format_ok, metric_hard_ok]
    else:
        print("WARNING: reward_weights unsupported; training on physics_reward only, metrics come from evaluate.py")
        reward_funcs = [physics_reward]

    log_path = Path(args.out) / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class JsonlLogger(TrainerCallback):
        def __init__(self):
            self.t = time.time()

        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                rec = {"step": state.global_step, "wall_s": round(time.time() - self.t, 1)}
                rec.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

    per_step = bs * args.grad_accum
    run_cfg = {"args": vars(args), "device": device, "dtype": str(dtype), "num_generations": G, "per_device_bs": bs,
               "grad_accum": args.grad_accum, "completions_per_step": per_step, "prompts_per_step": per_step // G,
               "dropped_config_keys": dropped, "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__},
               "grpo_config": {k: v for k, v in wanted.items() if k in supported and k != "output_dir"}}
    (Path(args.out) / "run_config.json").write_text(json.dumps(run_cfg, indent=1, default=str), encoding="utf-8")
    print(f"device={device} dtype={dtype} torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__}")
    print(f"model={args.model} violation_mode={VIOLATION_MODE} num_generations={G} per_device_bs={bs} grad_accum={args.grad_accum}")
    print(f"per optimizer step: {per_step} completions = {per_step // G} prompts; max_steps={args.max_steps}; save every {save_steps}")
    trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=lora, callbacks=[JsonlLogger()])
    trainer.train()
    final = Path(args.out) / "final"
    trainer.save_model(str(final))
    (Path(args.out) / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    if on_cuda:
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"saved LoRA to {final}")


if __name__ == "__main__":
    main()
