#!/usr/bin/env python3

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
from common.models import load_t2i, t2i_images
from common.progress import ProgressLogger

BENCH = Path(os.environ.get("PRISM_BENCHMARK_ROOT", "data")) / "GenEval"
DEFAULT_PROMPTS = BENCH / "prompts" / "evaluation_metadata.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1", help="Model name (for outdir).")
    p.add_argument("--model_path", default="")
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--outdir", default=None, help="Default: benchmarks/GenEval/outputs/{model}")
    p.add_argument("--n_samples", type=int, default=4, help="Images per prompt (GenEval uses 4).")
    p.add_argument("--width", type=int, default=2048, help="SenseNova 1:1 trained bucket.")
    p.add_argument("--height", type=int, default=2048, help="SenseNova 1:1 trained bucket.")
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--cfg_norm", default="global",
                   choices=["none", "global", "channel", "cfg_zero_star"],
                   help="CFG rescaling; official eval uses 'global'.")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--think", action="store_true", help="Enable t2i reasoning (think) mode.")
    p.add_argument("--limit", type=int, default=None, help="Only first N prompts (debug).")
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else BENCH / "outputs" / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.prompts) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        prompts = prompts[:args.limit]

    my_items = [(i, m) for i, m in enumerate(prompts) if i % args.num_shards == args.shard]
    print(f"[data] {len(my_items)}/{len(prompts)} prompts (shard {args.shard}/{args.num_shards})")

    backend = load_t2i(args.model, args.model_path, device=args.device, dtype=args.dtype)
    print(f"[load] {args.model} from {args.model_path}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw): return x

    log_dir = EVAL_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    plog = ProgressLogger(log_dir / "geneval_progress.log", args.shard, args.num_shards, len(my_items))

    done = 0
    skipped = 0
    for index, metadata in tqdm(my_items, desc=f"GenEval[{args.shard}]"):
        d = outdir / f"{index:05d}"
        sample_files = sorted((d / "samples").glob("*.png")) if (d / "samples").is_dir() else []
        if len(sample_files) >= args.n_samples:
            skipped += 1
            continue
        (d / "samples").mkdir(parents=True, exist_ok=True)
        with open(d / "metadata.jsonl", "w") as fp:
            json.dump(metadata, fp)

        images = t2i_images(
            backend, metadata["prompt"], args.width, args.height,
            cfg_scale=args.cfg_scale, cfg_norm=args.cfg_norm, num_steps=args.num_steps,
            seed=args.seed, think=args.think, n_samples=args.n_samples, device=args.device,
        )
        for k, img in enumerate(images):
            img.save(d / "samples" / f"{k:05d}.png")

        done += 1
        plog.update(done, 0)
    plog.done(done, 0)
    print(f"[done] shard {args.shard}: {done} prompts, skipped {skipped} -> {outdir}")


if __name__ == "__main__":
    main()
