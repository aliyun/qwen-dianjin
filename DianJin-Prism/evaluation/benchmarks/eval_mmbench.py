#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import create_model


def build_prompt(question, hint, options):
    parts = []
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {question}")
    parts.append("Options:")
    keys = sorted(k for k in options if options[k])
    for k in keys:
        parts.append(f"{k}. {options[k]}")
    parts.append("Answer with ONLY one option letter. Do NOT explain or add any other text.")
    return "\n".join(parts)


def extract_answer(text, allowed="ABCD"):
    text = re.split("</think" + ">", text)[-1]
    cls = "".join(sorted(set(allowed.upper())))
    m = re.search(rf'(?:final answer|answer|option|choice)\s*[:is]+\s*\(?([{cls}])\)?', text, re.I)
    if m:
        return m.group(1).upper()
    letters = re.findall(rf'\b([{cls}])\b', text)
    return letters[-1] if letters else ""


def score(results, summary_path=None):
    correct = total = 0
    for r in results:
        if r["gold"] is None:
            continue
        total += 1
        if r["pred"] == r["gold"]:
            correct += 1
    if total:
        print(f"\n[final] accuracy = {correct}/{total} = {correct/total:.4f}")
    else:
        print("\n[final] no gold answers (test split)")

    if summary_path:
        summary = {
            "dataset": "MMBench_EN", "metric": "accuracy",
            "correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else None,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1")
    p.add_argument("--model_path", default="")
    p.add_argument("--data_dir", default="data/MMBench")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--output", default=None)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shard_seed", type=int, default=1234,
                   help="分片前打乱用的种子(仅影响负载均衡, 不影响分数)")
    p.add_argument("--score_only", action="store_true", help="Skip inference; score an existing results file.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / f"{args.split}.jsonl"
    base_out = Path(args.output) if args.output else data_dir / f"{args.split}_results.jsonl"

    if args.score_only:
        with open(base_out) as f:
            results = [json.loads(line) for line in f if line.strip()]
        print(f"[score-only] {len(results)} results from {base_out}")
        score(results, base_out.with_suffix(".summary.json"))
        return

    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        samples = samples[:args.limit]
    if args.num_shards > 1:
        random.Random(args.shard_seed).shuffle(samples)
        samples = samples[args.shard::args.num_shards]
        out_path = base_out.with_suffix(f".shard{args.shard}of{args.num_shards}.jsonl")
    else:
        out_path = base_out
    print(f"[data] {len(samples)} samples (shard {args.shard}/{args.num_shards})")

    model, tokenizer, generate = create_model(
        args.model, args.model_path, device=args.device, dtype=args.dtype,
    )
    print(f"[load] {args.model} from {args.model_path}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw): return x

    from common.progress import ProgressLogger
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    plog = ProgressLogger(log_dir / "mmbench_progress.log", args.shard, args.num_shards, len(samples))

    correct, total, processed, results = 0, 0, 0, []
    for sample in tqdm(samples, desc=f"MMBench[{args.shard}]"):
        img_path = str(data_dir / sample["image"])
        prompt = build_prompt(sample["question"], sample["hint"], sample["options"])
        response = generate(model, tokenizer, [img_path], prompt,
                            max_new_tokens=args.max_new_tokens, device=args.device)
        pred = extract_answer(response, "".join(sorted(k for k in sample["options"] if sample["options"][k])))
        gold = sample.get("answer")
        results.append({"index": sample["index"], "pred": pred, "gold": gold, "response": response})
        if gold is not None:
            total += 1
            if pred == gold:
                correct += 1
        processed += 1
        plog.update(processed, correct)
    plog.done(processed, correct)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")

    if args.num_shards == 1:
        score(results, base_out.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
