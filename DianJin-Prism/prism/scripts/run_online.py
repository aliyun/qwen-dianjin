
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import os

from prism.configs.config import PrismConfig


def load_seeds(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any("prompt" not in row for row in rows):
        raise ValueError("seed file rows must contain id, category, and prompt")
    return rows


def _acquire_load_slot(out_dir: str, k: int, rank: int):
    if k <= 0:
        return None
    import time
    semdir = os.path.join(out_dir, ".loadsem")
    os.makedirs(semdir, exist_ok=True)
    while True:
        for i in range(k):
            p = os.path.join(semdir, f"slot{i}")
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, str(rank).encode())
                os.close(fd)
                return p
            except FileExistsError:
                pass
        now = time.time()
        for i in range(k):
            p = os.path.join(semdir, f"slot{i}")
            try:
                if now - os.path.getmtime(p) > 1800:
                    os.unlink(p)
            except OSError:
                pass
        time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser(description="Prism online (rollout->audit->teach->update).")
    p.add_argument("--adapter", default="sensenova_u1", choices=["sensenova_u1", "bagel"])
    p.add_argument("--model_path", default="")
    p.add_argument("--repo_path", default="")
    p.add_argument("--seeds", default="data/train_prompts.jsonl")
    p.add_argument("--out", default="outputs/prism/online", help="输出/保存根目录 -> cfg.output_dir")
    p.add_argument("--save_freq", type=int, default=1, help="每 N 个 epoch 存 checkpoint 到 <out>/checkpoints/epoch{E}")
    p.add_argument("--save_every_steps", type=int, default=0,
                   help="每 N 次 optimizer 更新(=N 个有效 batch)存 step checkpoint 到 <out>/checkpoints/step{U}; 0=关")
    p.add_argument("--keep_step_ckpts", type=int, default=2, help="step checkpoint 滚动保留个数(0=不清理)")
    p.add_argument("--resume_from", default="", help="加载该 checkpoint 目录继续训练")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA 缩放为 alpha/rank; 保持 alpha=2*rank 可让缩放不变, 于是提 rank 是单变量")
    p.add_argument("--ce_weight", type=float, default=1.0,
                   help="CE loss scaling factor (BAGEL paper SFT uses 0.25 vs mse=1.0)")
    p.add_argument("--mse_weight", type=float, default=1.0,
                   help="MSE/FM loss scaling factor")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--cfg_txt_drop_prob", type=float, default=0.1,
                   help="official cfg_txt_uncond_drop_prob: prob a t2i teach sample drops all text conditioning (trains the CFG uncond branch)")
    p.add_argument("--cfg_txtimg_drop_prob", type=float, default=0.1,
                   help="official cfg_txtimg_uncond_drop_prob: prob ONE random gen image is modeled with no conditioning at all (trains the fully unconditional CFG branch)")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--regen_num_steps", type=int, default=50, help="diffusion steps to regenerate high-quality I+ targets (rollout uses --num_steps)")
    p.add_argument("--gen_max_new_tokens", type=int, default=8192, help="interleave text-token cap (model default 8192; images capped by max_images)")
    p.add_argument("--max_images", type=int, default=8, help="max images per rollout (lower to prevent OOM at high resolution)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--audit_max_new_tokens", type=int, default=4096)
    p.add_argument("--audit_verdict_provider", choices=["self", "external"], default="self",
                   help="audit verdict source; self keeps the existing frozen-base self-judge")
    p.add_argument("--external_audit_max_new_tokens", type=int, default=8192,
                   help="external verdict completion budget")
    p.add_argument("--verify_max_retries", type=int, default=3, help="复核失败后的最大重试次数(总尝试数=1+该值)")
    p.add_argument("--verify_temperature_step", type=float, default=0.2, help="T+ 每次重试的温度增量(基准 0.6, 上限 1.0)")
    p.add_argument("--accelerate", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps (effective batch)")
    p.add_argument("--seed", type=int, default=0, help="deterministic per-epoch shuffle seed (same across ranks)")
    args = p.parse_args()
    if args.audit_verdict_provider == "external":
        missing = [name for name in ("PRISM_EXT_JUDGE_URL", "PRISM_EXT_JUDGE_KEY", "PRISM_EXT_JUDGE_MODEL")
                   if not os.environ.get(name, "").strip()]
        if missing:
            raise SystemExit("external verdict provider requires: " + ", ".join(missing))

    cfg = PrismConfig()
    cfg.mode = "online"
    cfg.adapter.model_path = args.model_path
    cfg.train.lora_rank, cfg.train.lora_alpha = args.lora_rank, args.lora_alpha
    cfg.train.epochs, cfg.train.lr = args.epochs, args.lr
    cfg.train.grad_accum = args.grad_accum
    cfg.train.save_every_steps = args.save_every_steps
    cfg.train.keep_step_ckpts = args.keep_step_ckpts
    cfg.train.save_freq = args.save_freq
    cfg.train.resume_from = args.resume_from
    cfg.output_dir = args.out
    cfg.loss.cfg_txt_drop_prob = args.cfg_txt_drop_prob
    cfg.loss.cfg_txtimg_drop_prob = args.cfg_txtimg_drop_prob
    cfg.audit.max_new_tokens = args.audit_max_new_tokens
    cfg.audit.verdict_provider = args.audit_verdict_provider
    cfg.audit.external_max_new_tokens = args.external_audit_max_new_tokens
    cfg.teach.verify_max_retries = args.verify_max_retries
    cfg.teach.verify_temperature_step = args.verify_temperature_step
    cfg.sample.num_steps = args.num_steps
    cfg.teach.regen_num_steps = args.regen_num_steps
    cfg.sample.image_size = (args.width, args.height)
    cfg.sample.max_images = args.max_images

    from prism.adapters import build_adapter
    from prism.audit import Auditor, build_judge
    from prism.training.online import online_train
    from prism.teach import Teacher
    from prism.training.trainer import PrismTrainer

    accelerator = None
    device = args.device
    if args.accelerate:
        from datetime import timedelta

        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs

        accelerator = Accelerator(
            gradient_accumulation_steps=args.grad_accum,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))],
        )
        device = str(accelerator.device)

    _load_k = int(os.environ.get("PRISM_LOAD_CONCURRENCY", "4"))
    _load_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    _slot = _acquire_load_slot(args.out, _load_k, _load_rank)
    try:
        if args.adapter == "bagel":
            adapter = build_adapter(
                "bagel", model_path=args.model_path, device=device,
                dtype="bfloat16", trainable=True,
                gen_defaults={"max_new_tokens": args.gen_max_new_tokens},
                train_defaults={"unlike_clamp": cfg.loss.unlike_clamp,
                                "cfg_txt_drop_prob": cfg.loss.cfg_txt_drop_prob,
                                "ce_weight": args.ce_weight,
                                "mse_weight": args.mse_weight},
            )
        else:
            adapter = build_adapter(
                "sensenova_u1", model_path=args.model_path, device=device,
                dtype="bfloat16", repo_paths={"sensenova_u1": args.repo_path},
                gen_defaults={"max_new_tokens": args.gen_max_new_tokens},
                fm_params={"unlike_clamp": cfg.loss.unlike_clamp,
                           "cfg_txt_drop_prob": cfg.loss.cfg_txt_drop_prob,
                           "cfg_txtimg_drop_prob": cfg.loss.cfg_txtimg_drop_prob},
            )
        try:
            import ctypes
            import gc
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    finally:
        if _slot:
            try:
                os.unlink(_slot)
            except OSError:
                pass
    judge = build_judge(cfg.audit, adapter)
    auditor = Auditor(judge, gen_desc=True, verify_tplus=cfg.teach.verify_tplus,
                      verify_max_retries=cfg.teach.verify_max_retries,
                      verify_temperature_step=cfg.teach.verify_temperature_step)
    teacher = Teacher(adapter, judge=judge,
                      regen_num_steps=cfg.teach.regen_num_steps,
                      regen_image_size=tuple(cfg.sample.image_size),
                      strict_desc=True,
                      max_images=getattr(cfg.sample, 'max_images', 10),
                      verify_iplus=cfg.teach.verify_iplus,
                      verify_max_retries=cfg.teach.verify_max_retries)
    trainer = PrismTrainer(adapter, cfg, accelerator=accelerator)
    start_epoch = 0
    resume_from = cfg.train.resume_from or ""
    if not resume_from:
        import re
        ckpt_root = os.path.join(cfg.output_dir, "checkpoints")
        if os.path.isdir(ckpt_root):
            steps = sorted((d for d in os.listdir(ckpt_root) if re.fullmatch(r"step\d+", d)),
                           key=lambda d: int(d[4:]))
            if steps:
                resume_from = os.path.join(ckpt_root, steps[-1])
                print(f"[resume] auto-found {resume_from}", flush=True)
    if resume_from:
        trainer.load_state(resume_from)
        pj = os.path.join(resume_from, "progress.json")
        if os.path.exists(pj):
            with open(pj, encoding="utf-8") as f:
                start_epoch = int(json.load(f).get("next_epoch", 0))

    seeds = load_seeds(args.seeds)

    online_train(adapter, auditor, teacher, trainer, seeds, cfg, accelerator=accelerator, out_dir=cfg.output_dir,
                 shuffle_seed=args.seed, start_epoch=start_epoch)
    save_path = os.environ.get("PRISM_CKPT_ROOT") or cfg.output_dir
    trainer.save(save_path)


if __name__ == "__main__":
    main()
