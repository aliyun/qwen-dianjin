#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
from common.models import load_t2i, t2i_images
from common.progress import ProgressLogger

BENCH = Path(os.environ.get("PRISM_BENCHMARK_ROOT", "data")) / "DPGBench"
DEFAULT_PROMPTS = BENCH / "prompts"


def make_grid_2x2(images, res):
    from PIL import Image
    grid = Image.new("RGB", (res * 2, res * 2))
    positions = [(0, 0), (res, 0), (0, res), (res, res)]
    for img, pos in zip(images, positions):
        if img.size != (res, res):
            img = img.resize((res, res))
        grid.paste(img, pos)
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sensenova-u1", help="Model name (for outdir).")
    p.add_argument("--model_path", default="")
    p.add_argument("--prompts_dir", default=str(DEFAULT_PROMPTS))
    p.add_argument("--outdir", default=None, help="Default: benchmarks/DPGBench/outputs/{model}")
    p.add_argument("--res", type=int, default=1024, help="Per-quadrant size; grid = 2*res. "
                                                          "Pass this as --resolution to the scorer.")
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
    p.add_argument("--think", action="store_true", help="Enable t2i think mode.")
    p.add_argument("--limit", type=int, default=None, help="Only first N prompts (debug).")
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else BENCH / "outputs" / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(Path(args.prompts_dir).glob("*.txt"))
    if args.limit:
        prompt_files = prompt_files[:args.limit]

    my_files = [pf for i, pf in enumerate(prompt_files) if i % args.num_shards == args.shard]
    print(f"[data] {len(my_files)}/{len(prompt_files)} prompts (shard {args.shard}/{args.num_shards})")

    backend = load_t2i(args.model, args.model_path, device=args.device, dtype=args.dtype)
    print(f"[load] {args.model} from {args.model_path}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw): return x

    log_dir = EVAL_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    plog = ProgressLogger(log_dir / "dpg_progress.log", args.shard, args.num_shards, len(my_files))

    done = 0
    skipped = 0
    for pf in tqdm(my_files, desc=f"DPG[{args.shard}]"):
        prompt = pf.read_text(encoding="utf-8").strip()
        out_path = outdir / f"{pf.stem}.png"
        if out_path.exists():
            skipped += 1
            continue

        images = t2i_images(
            backend, prompt, args.res, args.res,
            cfg_scale=args.cfg_scale, cfg_norm=args.cfg_norm, num_steps=args.num_steps,
            seed=args.seed, think=args.think, n_samples=4, device=args.device,
        )
        make_grid_2x2(images, args.res).save(out_path)

        done += 1
        plog.update(done, 0)
    plog.done(done, 0)
    print(f"[done] shard {args.shard}: {done} prompts, skipped {skipped} -> {outdir}")


if __name__ == "__main__":
    main()
