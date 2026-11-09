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


def extract_yes_no(text):
    text = re.split("</think" + ">", text)[-1]
    m = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
    if m:
        return "Yes" if m.group(1).lower() == "yes" else "No"
    return ""


def norm_yes_no(ans):
    return "Yes" if str(ans).strip().lower() == "yes" else "No"


def _metrics(rows):
    tp = fp = fn = tn = 0
    yes_pred = 0
    for pred, gold in rows:
        p = "Yes" if pred == "Yes" else "No"
        if p == "Yes":
            yes_pred += 1
        if p == "Yes" and gold == "Yes":
            tp += 1
        elif p == "Yes" and gold == "No":
            fp += 1
        elif p == "No" and gold == "Yes":
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_ratio = yes_pred / n if n else 0.0
    return {"total": n, "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "yes_ratio": round(yes_ratio, 4)}


def score(results, summary_path=None):
    by_cat = defaultdict(list)
    allrows = []
    for r in results:
        pred = r["pred"]
        gold = r["gold"]
        by_cat[r["category"]].append((pred, gold))
        allrows.append((pred, gold))

    per_cat = {cat: _metrics(rows) for cat, rows in by_cat.items()}
    overall = _metrics(allrows)
    f1s = [per_cat[c]["f1"] for c in per_cat]
    avg_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0.0

    print("\n[per-category] (positive = 'Yes')")
    for cat in sorted(per_cat):
        m = per_cat[cat]
        print(f"  {cat:12s} acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  yes%={m['yes_ratio']:.4f}  (n={m['total']})")
    print(f"\n[overall] acc={overall['accuracy']:.4f}  F1={overall['f1']:.4f}  "
          f"yes%={overall['yes_ratio']:.4f}  (n={overall['total']})")
    print(f"[macro-average F1 over splits] = {avg_f1:.4f}")

    if summary_path:
        summary = {
            "dataset": "POPE", "metric": "F1 (positive=Yes)",
            "overall": overall,
            "per_category": per_cat,
            "average_f1": avg_f1,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1")
    p.add_argument("--model_path", default="")
    p.add_argument("--data_dir", default="data/POPE")
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
    plog = ProgressLogger(log_dir / "pope_progress.log", args.shard, args.num_shards, len(samples))

    correct, total, results = 0, 0, []
    for sample in tqdm(samples, desc=f"POPE[{args.shard}]"):
        img_path = str(data_dir / sample["image"])
        prompt = sample["question"] + "\nAnswer with ONLY Yes or No. Do NOT explain or add any other text."
        response = generate(model, tokenizer, [img_path], prompt,
                            max_new_tokens=args.max_new_tokens, device=args.device)
        pred = extract_yes_no(response)
        gold = norm_yes_no(sample["answer"])
        results.append({
            "question_id": sample["question_id"], "category": sample["category"],
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
