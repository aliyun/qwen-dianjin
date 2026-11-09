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


def parse_options(options_str):
    m = re.match(r'\(a\)\s*(.+?)\s*\(b\)\s*(.+)', options_str,
                 re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r'\(?a\)?\.?\s*(.+?)\s*\(?b\)?\.?\s*(.+)', options_str,
                 re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def map_gold_letter(correct_answer, opt_a, opt_b):
    ca = correct_answer.strip().lower()
    if ca == opt_a.strip().lower():
        return "A"
    if ca == opt_b.strip().lower():
        return "B"
    if opt_a and opt_a.strip().lower() in ca:
        return "A"
    if opt_b and opt_b.strip().lower() in ca:
        return "B"
    if ca in ("a", "(a)"):
        return "A"
    if ca in ("b", "(b)"):
        return "B"
    return ""


def build_prompt(question, opt_a, opt_b):
    return (
        f"Question: {question}\n"
        f"Options:\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n"
        f"Answer with ONLY one option letter. Do NOT explain or add any other text."
    )


def extract_answer(text, allowed="AB"):
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
        total += 1
        if r["pred"] == r["gold"]:
            correct += 1
    q_acc = correct / total if total else 0.0

    pair_correct = pair_total = 0
    pairs = defaultdict(list)
    for r in results:
        pairs[r["pair_id"]].append(r)
    for pid, items in pairs.items():
        if len(items) < 2:
            continue
        pair_total += 1
        if all(i["pred"] == i["gold"] for i in items):
            pair_correct += 1
    p_acc = pair_correct / pair_total if pair_total else 0.0

    print(f"\n[per-question] accuracy = {correct}/{total} = {q_acc:.4f}")
    print(f"[paired]        accuracy = {pair_correct}/{pair_total} = {p_acc:.4f}")

    if summary_path:
        summary = {
            "dataset": "MMVP",
            "metric": "paired accuracy",
            "per_question": {"correct": correct, "total": total,
                             "accuracy": round(q_acc, 4)},
            "paired": {"correct": pair_correct, "total": pair_total,
                       "accuracy": round(p_acc, 4)},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] {summary_path}")


def load_samples(data_dir):
    jsonl_path = data_dir / "test.jsonl"
    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    for s in samples:
        s["gold"] = s.pop("answer", "")
        s["image"] = str(data_dir / s["image"])
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1")
    p.add_argument("--model_path",
                   default="")
    p.add_argument("--data_dir",
                   default="data/MMVP")
    p.add_argument("--output", default=None)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shard_seed", type=int, default=1234,
                   help="Shuffle seed for load balancing (does not affect score).")
    p.add_argument("--score_only", action="store_true",
                   help="Skip inference; score an existing results file.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    base_out = (Path(args.output) if args.output
                else data_dir / "test_results.jsonl")

    if args.score_only:
        with open(base_out) as f:
            results = [json.loads(line) for line in f if line.strip()]
        print(f"[score-only] {len(results)} results from {base_out}")
        score(results, base_out.with_suffix(".summary.json"))
        return

    samples = load_samples(data_dir)
    if args.limit:
        n_pairs = args.limit // 2
        samples = samples[:n_pairs * 2]
    if args.num_shards > 1:
        pairs = [samples[i:i + 2] for i in range(0, len(samples), 2)]
        random.Random(args.shard_seed).shuffle(pairs)
        pair_shards = pairs[args.shard::args.num_shards]
        samples = [q for pair in pair_shards for q in pair]
        out_path = base_out.with_suffix(
            f".shard{args.shard}of{args.num_shards}.jsonl")
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
        def tqdm(x, **_kw):
            return x

    from common.progress import ProgressLogger
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    plog = ProgressLogger(log_dir / "mmvp_progress.log",
                           args.shard, args.num_shards, len(samples))

    correct, total, results = 0, 0, []
    for sample in tqdm(samples, desc=f"MMVP[{args.shard}]"):
        prompt = build_prompt(sample["question"], sample["opt_a"],
                              sample["opt_b"])
        response = generate(model, tokenizer, [sample["image"]], prompt,
                            max_new_tokens=args.max_new_tokens,
                            device=args.device)
        pred = extract_answer(response)
        gold = sample["gold"]
        results.append({
            "question_id": sample["question_id"],
            "pair_id": sample["pair_id"],
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
