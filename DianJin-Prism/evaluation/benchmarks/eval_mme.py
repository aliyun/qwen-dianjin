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


PERCEPTION = {"existence", "count", "position", "color", "posters",
              "celebrity", "scene", "landmark", "artwork", "OCR"}
COGNITION = {"commonsense_reasoning", "numerical_calculation",
             "text_translation", "code_reasoning"}


def official_mme(results):
    per_cat_q = defaultdict(lambda: [0, 0])
    per_img = defaultdict(lambda: [0, 0])
    for r in results:
        cat = r["category"]
        ok = int(r["pred"] == r["gold"])
        per_cat_q[cat][0] += ok
        per_cat_q[cat][1] += 1
        key = (cat, r["question_id"])
        per_img[key][0] += ok
        per_img[key][1] += 1

    per_cat_img = defaultdict(lambda: [0, 0])
    for (cat, _qid), (cq, tq) in per_img.items():
        per_cat_img[cat][0] += int(cq == tq)
        per_cat_img[cat][1] += 1

    subtask = {}
    for cat in per_cat_q:
        cq, tq = per_cat_q[cat]
        ci, ti = per_cat_img[cat]
        acc = cq / tq if tq else 0.0
        acc_plus = ci / ti if ti else 0.0
        subtask[cat] = round((acc + acc_plus) * 100, 2)
    mme_p = round(sum(v for c, v in subtask.items() if c in PERCEPTION), 2)
    mme_c = round(sum(v for c, v in subtask.items() if c in COGNITION), 2)
    return {"subtask_scores": subtask, "MME-P": mme_p, "MME-C": mme_c,
            "MME_total": round(mme_p + mme_c, 2)}


def score(results, summary_path=None):
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    correct = total = 0
    for r in results:
        cat_total[r["category"]] += 1
        total += 1
        if r["pred"] == r["gold"]:
            cat_correct[r["category"]] += 1
            correct += 1
    print(f"\n[overall] accuracy = {correct}/{total} = {correct/total:.4f}")
    print("\n[per-category]")
    per_cat = {}
    for cat in sorted(cat_total):
        c, t = cat_correct[cat], cat_total[cat]
        per_cat[cat] = {"correct": c, "total": t, "accuracy": round(c / t, 4)}
        print(f"  {cat:30s} {c}/{t} = {c/t:.4f}")

    off = official_mme(results)
    print("\n[official MME score] (subtask = (acc+acc+)*100, max 200 each)")
    for cat in sorted(off["subtask_scores"]):
        tag = "P" if cat in PERCEPTION else "C"
        print(f"  [{tag}] {cat:30s} {off['subtask_scores'][cat]:.2f}")
    print(f"  MME-P (perception, /2000) = {off['MME-P']:.2f}")
    print(f"  MME-C (cognition,  /800)  = {off['MME-C']:.2f}")
    print(f"  MME total (/2800)         = {off['MME_total']:.2f}")

    if summary_path:
        summary = {
            "dataset": "MME", "metric": "per-question accuracy",
            "overall": {"correct": correct, "total": total,
                        "accuracy": round(correct / total, 4)},
            "per_category": per_cat,
            "official": off,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1")
    p.add_argument("--model_path", default="")
    p.add_argument("--data_dir", default="data/MME")
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
    plog = ProgressLogger(log_dir / "mme_progress.log", args.shard, args.num_shards, len(samples))

    correct, total, results = 0, 0, []
    for sample in tqdm(samples, desc=f"MME[{args.shard}]"):
        img_path = str(data_dir / sample["image"])
        response = generate(model, tokenizer, [img_path], sample["question"],
                            max_new_tokens=args.max_new_tokens, device=args.device)
        pred = extract_yes_no(response)
        gold = sample["answer"]
        results.append({
            "question_id": sample["question_id"], "category": sample["category"],
            "pred": pred, "gold": gold, "response": response,
        })
        total += 1
        if pred == gold:
            correct += 1
        try: plog.update(total, correct)
        except OSError: pass
    try: plog.done(total, correct)
    except OSError: pass

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")

    if args.num_shards == 1:
        score(results, base_out.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
