"""Greedy-decoding evaluation of a base model or LoRA checkpoint (IMPLEMENTATION_PLAN.md 4.3).

Usage:
  python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --data data/eval_in.jsonl data/eval_heldout.jsonl \
      --out outputs/eval/base.json --tag 0
  python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/grpo/run1/checkpoint-20 \
      --out outputs/eval/step20.json --tag 20
Zero-variance probe (section 6): add --sample_k 8 [--sample_n 50] to also sample k completions at temperature 1.0
for the first sample_n training prompts and report the share of groups whose rewards are all identical.
Writes <out> (one summarize() result per dataset) and <out stem>.samples.jsonl (task, completion, score).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from ballwall.checker import score  # noqa: E402
from ballwall.metrics import summarize  # noqa: E402
from ballwall.task import read_jsonl  # noqa: E402


def pick_device(choice: str = "auto"):
    import torch
    if choice != "auto":
        return choice, (torch.bfloat16 if choice == "cuda" else torch.float32)
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model(model_name: str, adapter: str | None, device: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return tok, model


def prompt_text(tok, messages, model_name: str) -> str:
    kw = {"enable_thinking": False} if "qwen3" in model_name.lower() else {}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def generate(tok, model, texts, device, max_new_tokens, batch_size, sample=False, temperature=1.0, desc=""):
    import torch
    outs = []
    t_start = time.time()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, stop_strings=["</svg>"], tokenizer=tok)
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            gen = model.generate(**enc, **kw)
        comp = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(comp, skip_special_tokens=True))
        done = min(i + batch_size, len(texts))
        el = time.time() - t_start
        print(f"  {desc} {done}/{len(texts)}  {el:.0f}s elapsed, ~{el / done * (len(texts) - done):.0f}s left", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data/eval_in.jsonl"), str(ROOT / "data/eval_heldout.jsonl")])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--violation_mode", default="low", choices=["low", "zero"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="0")
    ap.add_argument("--sample_k", type=int, default=0)
    ap.add_argument("--sample_n", type=int, default=50)
    ap.add_argument("--sample_data", default=str(ROOT / "data/train.jsonl"))
    ap.add_argument("--skip_greedy", action="store_true", help="only run the --sample_k probe")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    import torch
    device, dtype = pick_device(args.device)
    print(f"device={device} dtype={dtype} model={args.model} adapter={args.adapter}", flush=True)
    t0 = time.time()
    tok, model = load_model(args.model, args.adapter, device, dtype)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples_path = out.with_suffix(".samples.jsonl")
    report = {"tag": args.tag, "model": args.model, "adapter": args.adapter, "violation_mode": args.violation_mode,
              "max_new_tokens": args.max_new_tokens, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
    with open(samples_path, "w", encoding="utf-8", newline="\n") as sf:
        if not args.skip_greedy:
            for path in args.data:
                name = Path(path).stem
                rows = read_jsonl(path)[: args.limit]
                texts = [prompt_text(tok, r["prompt"], args.model) for r in rows]
                t1 = time.time()
                comps = generate(tok, model, texts, device, args.max_new_tokens, args.batch_size, desc=name)
                gen_s = time.time() - t1
                results = []
                for r, c in zip(rows, comps):
                    res = score(c, r["task"], args.violation_mode)
                    results.append(res)
                    sf.write(json.dumps({"dataset": name, "task": r["task"], "completion": c, "score": res}, ensure_ascii=False) + "\n")
                summ = summarize(results)
                summ["gen_seconds"] = round(gen_s, 1)
                summ["completion_chars_mean"] = round(sum(len(c) for c in comps) / max(1, len(comps)), 1)
                report["datasets"][name] = summ
                print(f"{name}: success {summ['success_rate']:.1%} CI {summ['success_ci95']}, reward {summ['mean_reward']:.3f}, "
                      f"format fail {summ['format_fail_rate']:.1%}, hard viol {summ['hard_violation_rate']:.1%}", flush=True)
                print("  categories:", {k: round(v, 3) for k, v in summ["category_dist"].items() if v}, flush=True)

        if args.sample_k:
            rows = read_jsonl(args.sample_data)[: args.sample_n]
            texts = [prompt_text(tok, r["prompt"], args.model) for r in rows for _ in range(args.sample_k)]
            comps = generate(tok, model, texts, device, args.max_new_tokens, args.batch_size, sample=True, desc="sample_k")
            probe = {"k": args.sample_k, "n_prompts": len(rows), "temperature": 1.0}
            for mode in ("low", "zero"):
                groups, flat = [], []
                for i, r in enumerate(rows):
                    g = [score(c, r["task"], mode) for c in comps[i * args.sample_k:(i + 1) * args.sample_k]]
                    groups.append([x["reward"] for x in g])
                    flat.extend(g)
                zero = sum(1 for g in groups if max(g) - min(g) < 1e-9) / len(groups)
                probe[mode] = {"frac_zero_std_groups": zero, "summary": summarize(flat)}
                print(f"sample_k probe [{mode}]: zero-variance groups {zero:.1%}, mean reward {probe[mode]['summary']['mean_reward']:.3f}, "
                      f"format ok {probe[mode]['summary']['p_format_ok']:.1%}", flush=True)
            for i, r in enumerate(rows):
                for c in comps[i * args.sample_k:(i + 1) * args.sample_k]:
                    sf.write(json.dumps({"dataset": "sample_k", "task": r["task"], "completion": c,
                                         "score": score(c, r["task"], args.violation_mode)}, ensure_ascii=False) + "\n")
            report["sample_k"] = probe

    if device == "cuda":
        report["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} and {samples_path}")


if __name__ == "__main__":
    main()
