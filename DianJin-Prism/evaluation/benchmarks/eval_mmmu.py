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


def check_is_number(value):
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(value):
    value = value.strip()
    if check_is_number(value):
        return [round(float(value.replace(",", "")), 2)]
    value = value.lower()
    if len(value) == 1:
        return [" " + value, value + " "]
    return [value]


def extract_numbers(value):
    patterns = [
        r"-?\b\d{1,3}(?:,\d{3})+\b",
        r"-?\d+(?:\.\d+)?[eE][+-]?\d+",
        r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])",
    ]
    return [number for pattern in patterns for number in re.findall(pattern, value)]


def parse_open_response(response):
    response = response.strip().strip(".").lower()
    key_responses = []
    indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
    for index, segment in enumerate(re.split(r"\.\s(?=[A-Z])|\n", response)):
        active = indicators + (["="] if index == len(re.split(r"\.\s(?=[A-Z])|\n", response)) - 1 else [])
        choices = [segment.split(indicator)[-1].strip() for indicator in active if indicator in segment]
        choices = [choice for choice in choices if choice not in [":", ",", ".", "!", "?", ";", ":", "'"]]
        if choices:
            key_responses.append(min(choices, key=len))
    if not key_responses:
        key_responses = [response]
    predictions = key_responses.copy()
    for value in key_responses:
        predictions.extend(extract_numbers(value))
    normalized = []
    for value in predictions:
        normalized.extend(normalize_str(value))
    return list(set(normalized))


def eval_open(gold, predictions):
    answers = gold if isinstance(gold, list) else [gold]
    normalized_answers = [item for answer in answers for item in normalize_str(answer)]
    for prediction in predictions:
        if isinstance(prediction, str):
            if any(isinstance(answer, str) and answer in prediction for answer in normalized_answers):
                return True
        elif prediction in normalized_answers:
            return True
    return False


def build_prompt_mc(question, options):
    parts = []
    for k in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        if k in options and options[k]:
            parts.append(f"({k}) {options[k]}")
    return question + "\n\n" + "\n".join(parts) + "\n\nAnswer with ONLY one option letter. Do NOT explain or add any other text."


def build_prompt_open(question):
    return question + "\n\nAnswer the question using a single word or phrase."


def _strip_think(text):
    return re.split("</think" + ">", text)[-1]


def normalize_answer(text):
    t = _strip_think(text).strip().lower()
    t = re.sub(r'[.$]', '', t)
    t = re.sub(r'^(answer|the answer is)\s*:?\s*', '', t)
    return t.strip()


def extract_mc_answer(text):
    text = _strip_think(text)
    m = re.search(r'\b([A-J])\b', text)
    return m.group(1) if m else ""


def is_correct(r):
    gold = r["gold"]
    if gold is None:
        return False
    if r["question_type"] == "open":
        pred_parsed = parse_open_response(r["response"])
        return eval_open(gold, pred_parsed)
    return r["pred"] == gold


def score(results, summary_path=None):
    correct = total = 0
    mc_c = mc_t = open_c = open_t = 0
    sub_correct = defaultdict(int)
    sub_total = defaultdict(int)
    for r in results:
        ok = is_correct(r)
        total += 1
        correct += ok
        if r["question_type"] == "open":
            open_t += 1
            open_c += ok
        else:
            mc_t += 1
            mc_c += ok
        sub_total[r["subject"]] += 1
        sub_correct[r["subject"]] += ok

    print(f"\n[overall] accuracy = {correct}/{total} = {correct/total:.4f}")
    print(f"[mc]      accuracy = {mc_c}/{mc_t} = {mc_c/mc_t:.4f}" if mc_t else "[mc]      N/A")
    print(f"[open]    accuracy = {open_c}/{open_t} = {open_c/open_t:.4f}" if open_t else "[open]    N/A")
    print("\n[per-subject]")
    per_sub = {}
    for sub in sorted(sub_total):
        c, t = sub_correct[sub], sub_total[sub]
        per_sub[sub] = {"correct": c, "total": t, "accuracy": round(c / t, 4)}
        print(f"  {sub:45s} {c}/{t} = {c/t:.4f}")

    if summary_path:
        summary = {
            "dataset": "MMMU", "metric": "accuracy",
            "overall": {"correct": correct, "total": total,
                        "accuracy": round(correct / total, 4) if total else None},
            "multiple_choice": {"correct": mc_c, "total": mc_t,
                                "accuracy": round(mc_c / mc_t, 4) if mc_t else None},
            "open": {"correct": open_c, "total": open_t,
                     "accuracy": round(open_c / open_t, 4) if open_t else None,
                     "note": "embedded MMMU open-answer normalization (substring + number extraction)"},
            "per_subject": per_sub,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1")
    p.add_argument("--model_path", default="")
    p.add_argument("--data_dir", default="data/MMMU")
    p.add_argument("--split", default="validation", choices=["dev", "validation", "test"])
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
    plog = ProgressLogger(log_dir / "mmmu_progress.log", args.shard, args.num_shards, len(samples))

    results = []
    correct = 0
    for sample in tqdm(samples, desc=f"MMMU[{args.shard}]"):
        question = sample["question"]
        qtype = sample.get("question_type", "multiple-choice")
        options = sample.get("options", {})
        gold = sample.get("answer")

        prompt_question = re.sub(r'<image \d+>', '<image>', question)
        if qtype == "open":
            prompt = build_prompt_open(prompt_question)
        else:
            prompt = build_prompt_mc(prompt_question, options)

        img_paths = [str(data_dir / img) for img in sample.get("images", [])]
        response = generate(model, tokenizer, img_paths, prompt,
                            max_new_tokens=args.max_new_tokens, device=args.device)

        if qtype == "open":
            pred = normalize_answer(response)
        else:
            pred = extract_mc_answer(response)

        r = {
            "id": sample["id"], "subject": sample.get("subject", ""),
            "question_type": qtype, "pred": pred, "gold": gold,
            "response": response,
        }
        results.append(r)
        correct += is_correct(r)
        plog.update(len(results), correct)
    plog.done(len(results), correct)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")

    if args.num_shards == 1:
        score(results, base_out.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
