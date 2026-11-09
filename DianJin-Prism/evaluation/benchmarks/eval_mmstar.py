#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import create_model


def parse_option_letters(question):
    opts = re.findall(r'([A-J]):', question)
    return "".join(sorted(set(opts))) or "ABCD"


def build_prompt(question):
    return (f"{question.rstrip()}\n\n"
            f"Answer with ONLY one option letter. Do NOT explain or add any other text.")


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
    cat_c = defaultdict(int); cat_t = defaultdict(int)
    l2_c = defaultdict(int); l2_t = defaultdict(int)
    for r in results:
        gold = r.get("gold")
        if gold is None:
            continue
        total += 1
        cat_t[r["category"]] += 1
        l2_t[r["l2_category"]] += 1
        if r["pred"] == gold:
            correct += 1
            cat_c[r["category"]] += 1
            l2_c[r["l2_category"]] += 1

    acc = correct / total if total else 0.0
    print(f"\n[overall] accuracy = {correct}/{total} = {acc:.4f}")
    print("\n[per-category]")
    per_cat = {}
    for cat in sorted(cat_t):
        c, t = cat_c[cat], cat_t[cat]
        per_cat[cat] = {"correct": c, "total": t, "accuracy": round(c / t, 4)}
        print(f"  {cat:26s} {c}/{t} = {c/t:.4f}")
    print("\n[per-L2-category]")
    per_l2 = {}
    for cat in sorted(l2_t):
        c, t = l2_c[cat], l2_t[cat]
        per_l2[cat] = {"correct": c, "total": t, "accuracy": round(c / t, 4)}
        print(f"  {cat:34s} {c}/{t} = {c/t:.4f}")

    if summary_path:
        summary = {
            "dataset": "MMStar", "metric": "accuracy",
            "overall": {"correct": correct, "total": total,
                        "accuracy": round(acc, 4) if total else None},
            "per_category": per_cat,
            "per_l2_category": per_l2,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1")
    p.add_argument("--model_path", default="")
    p.add_argument("--data_dir", default="data/MMStar")
    p.add_argument("--output", default=None)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shard_seed", type=int, default=1234,
                   help="分片前打乱用的种子(仅影响负载均衡, 不影响分数)")
    p.add_argument("--ids_file", default=None,
                   help="Resume: only run samples whose 'index' is listed in this file "
                        "(one integer per line); applied before sharding.")
    p.add_argument("--score_only", action="store_true", help="Skip inference; score an existing results file.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / "test.jsonl"
    base_out = Path(args.output) if args.output else data_dir / "test_results.jsonl"

    if args.score_only:
        with open(base_out) as f:
            results = [json.loads(line) for line in f if line.strip()]
        print(f"[score-only] {len(results)} results from {base_out}")
        score(results, base_out.with_suffix(".summary.json"))
        return

    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.ids_file:
        with open(args.ids_file) as f:
            keep = {int(x) for x in f.read().split()}
        samples = [s for s in samples if s["index"] in keep]
        print(f"[ids_file] restricted to {len(samples)} listed samples")
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
    plog = ProgressLogger(log_dir / "mmstar_progress.log", args.shard, args.num_shards, len(samples))

    correct, total, results = 0, 0, []
    for sample in tqdm(samples, desc=f"MMStar[{args.shard}]"):
        img_path = str(data_dir / sample["image"])
        prompt = build_prompt(sample["question"])
        response = generate(model, tokenizer, [img_path], prompt,
                            max_new_tokens=args.max_new_tokens, device=args.device)
        pred = extract_answer(response, parse_option_letters(sample["question"]))
        gold = sample.get("answer")
        results.append({
            "index": sample["index"], "category": sample["category"],
            "l2_category": sample["l2_category"],
            "pred": pred, "gold": gold, "response": response,
        })
        total += 1
        if pred == gold:
            correct += 1
        plog.update(total, correct)
    plog.done(total, correct)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")

    if args.num_shards == 1:
        score(results, base_out.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
