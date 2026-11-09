
from __future__ import annotations

import json
import os
import time
from collections import Counter

from ..core.degen import detect_repetition
from ..core.metrics import Metrics
from ..core.router import RoutingDecision, route
from ..core.types import Audit, AuditResult, Quadrant

_KINDS = ("t2i", "i2t", "s2t", "neg")


DEFAULT_CLASS_WEIGHTS = {"s2t": 0.5, "t2i": 1.0, "i2t": 1.0, "neg": 0.005}


def _class_weights(counts: dict, weights: dict | None = None) -> dict:
    w = weights or DEFAULT_CLASS_WEIGHTS
    return {k: float(w.get(k, 1.0)) for k in counts if counts[k] > 0}


def rollout(adapter, auditor, teacher, seed: dict, cfg, gen_seed=None) -> tuple:
    prompt = seed["prompt"]
    if "max_images" in seed:
        max_images = seed["max_images"]
    elif "exactly one image" in prompt.lower():
        max_images = 1
    else:
        max_images = getattr(cfg.sample, 'max_images', 10)
    gen_kw = {} if gen_seed is None else {"seed": int(gen_seed)}
    with adapter.generation_mode():
        t0 = time.time()
        traj = adapter.generate_interleaved(
            prompt, think=cfg.sample.think_mode,
            num_steps=cfg.sample.num_steps, image_size=tuple(cfg.sample.image_size),
            max_images=max_images,
            **gen_kw,
        )
    stages = [{"stage": "rollout", "seconds": round(time.time() - t0, 1),
               "n_images": len(traj.images), "raw_len": len(traj.raw_output or ""),
               "out_field": "raw_output/images"}]
    conds = (getattr(traj, "meta", None) or {}).get("image_conditions")
    if conds:
        stages[0]["conditions"] = [str(c)[:500] for c in conds]
    pm = int(getattr(traj, "parse_mismatch", 0))
    if pm:
        stages.append({"stage": "parse_mismatch", "excess": pm})
        teacher.last_skip_reason = "parse_mismatch"
        teacher.last_gate_reason = ""
        teacher.last_stages = []
        reason = f"discarded: {pm} bare <image> text marker(s) beyond real images"
        audit = Audit(text=AuditResult(valid=False, reason=reason),
                      image=AuditResult(valid=False, reason=reason))
        decision = RoutingDecision(Quadrant.UNTEACHABLE, "skip", None, "", "parse_mismatch")
        return traj, audit, decision, None, stages
    degen = detect_repetition(traj.raw_output or "")
    if degen:
        stages.append({"stage": "degen", **degen})
    t0 = time.time()
    audit = auditor.audit(traj)
    stages += list(getattr(auditor, "last_stages", []))
    decision = route(traj, audit)
    t0 = time.time()
    sample = teacher.teach(traj, audit, decision, gen_seed=gen_seed)
    stages += list(getattr(teacher, "last_stages", []))
    stages.append({"stage": "teach_total", "seconds": round(time.time() - t0, 1),
                   "result": (sample.kind if sample is not None else
                              (getattr(teacher, "last_skip_reason", "") or "skip"))})
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return traj, audit, decision, sample, stages


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt(x) -> str:
    return "None" if x is None else f"{x:.3f}"


def _log(trainer, msg: str) -> None:
    if trainer._is_main():
        print(f"[{_ts()}] {msg}", flush=True)


def _save_imgs(imgs, d: str, prefix: str) -> list[str]:
    names = []
    for i, im in enumerate(imgs or []):
        if hasattr(im, "save"):
            os.makedirs(d, exist_ok=True)
            im.save(os.path.join(d, f"{prefix}_{i}.png"))
            names.append(f"{prefix}_{i}.png")
    return names


def _audit3(a) -> dict:
    return {"valid": a.valid, "reason": a.reason, "description": a.description,
            "quote": getattr(a, "quote", "") or "", "raw": (a.raw or "")[:1200]}


def _write_sample(out_dir, epoch, sample_name, seed, traj, audit, decision, sample, save_images, skip_reason="", gate_reason="", stages=None, routing=None) -> str:
    d = os.path.join(out_dir, f"epoch{epoch}", sample_name)
    if os.path.exists(os.path.join(d, "record.json")):
        prev = os.path.join(out_dir, f"epoch{epoch}", "_prev_versions")
        for n in range(1, 1000):
            dst = os.path.join(prev, f"v{n}", sample_name)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(d, dst)
                break
    os.makedirs(d, exist_ok=True)
    img_dir = os.path.join(d, "images")
    out_imgs = _save_imgs(traj.images, img_dir, "img") if save_images else []
    tgt_imgs = []
    if save_images and sample is not None and sample.target_image is not None:
        tgt = sample.target_image if isinstance(sample.target_image, list) else [sample.target_image]
        tgt_imgs = _save_imgs(tgt, img_dir, "tgt")

    rec = {
        "epoch": epoch,
        "sample": sample_name,
        "seed_id": seed.get("id"),
        "category": seed.get("category"),
        "prompt": seed.get("prompt"),
        "output_T": {
            "text": traj.raw_output,
            "step_texts": [s.text for s in traj.steps],
            "final_answer": traj.final_answer,
        },
        "output_I": [f"images/{n}" for n in out_imgs],
        "audit_text": _audit3(audit.text),
        "audit_image": _audit3(audit.image),
        "quadrant": decision.quadrant.value,
        "action": decision.action,
        "reliable_side": decision.reliable_side,
        "routing": routing,
        "skip_reason": skip_reason,
        "gate_reason": gate_reason,
        "degen": next((s for s in (stages or []) if s.get("stage") == "degen"), None),
        "stages": stages or [],
        "teach": None if sample is None else {
            "kind": sample.kind,
            "target_source": sample.meta.get("target_source", "rewritten"),
            "target_text": sample.target_text,
            "neg_text": sample.neg_text,
            "target_images": [f"images/{n}" for n in tgt_imgs],
        },
        "train": None,
    }
    with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return d


def _patch_train(rec_dir, kind, loss, grad_norm, lr) -> None:
    p = os.path.join(rec_dir, "record.json")
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        rec["train"] = {"kind": kind, "loss": loss, "grad_norm": grad_norm, "lr": lr}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _append_jsonl(path, obj) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _rollout_health(health_dir, step: int) -> dict:
    if not health_dir or not os.path.isdir(health_dir):
        return {}
    try:
        rows = []
        for fn in os.listdir(health_dir):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(health_dir, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("step") == step:
                        rows.append(d)
        n = len(rows)
        if not n:
            return {}
        lens = sorted(r.get("len", 0) for r in rows)
        return {
            "step": step, "n": n,
            "zero_img": round(sum(1 for r in rows if not r.get("n_img")) / n, 3),
            "unclosed": round(sum(1 for r in rows if not r.get("closed")) / n, 3),
            "neg": round(sum(1 for r in rows if r.get("quad") == "unteachable") / n, 3),
            "len_med": lens[n // 2],
        }
    except Exception:
        return {}


def _pool_put(samples_dir: str, rank: int, seq: int, sample) -> None:
    import pickle
    os.makedirs(samples_dir, exist_ok=True)
    name = f"{time.time_ns():020d}_r{rank:02d}_{seq:06d}_{sample.kind}.pkl"
    tmp = os.path.join(samples_dir, f".{name}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(sample, f)
    os.replace(tmp, os.path.join(samples_dir, name))


def _pool_avail(samples_dir: str, consumed: set) -> list:
    try:
        names = [n for n in os.listdir(samples_dir) if n.endswith(".pkl")]
    except FileNotFoundError:
        return []
    return sorted(n for n in names if n not in consumed)


def _resume_step_from_checkpoint(path: str) -> int:
    if not path:
        return 0
    import re

    m = re.fullmatch(r"step(\d+)", os.path.basename(os.path.normpath(path)))
    return int(m.group(1)) if m else 0


def _run_root_from_checkpoint(path: str) -> str:
    if not path:
        return ""
    ckpt = os.path.abspath(path)
    parent = os.path.dirname(ckpt)
    if os.path.basename(parent) != "checkpoints":
        return ""
    return os.path.dirname(parent)


def _consumed_from_rounds(rounds_dir: str, upto_round: int) -> set:
    consumed = set()
    if not rounds_dir or upto_round <= 0:
        return consumed
    for i in range(upto_round):
        p = os.path.join(rounds_dir, f"round{i}.json")
        try:
            with open(p, encoding="utf-8") as f:
                consumed.update(json.load(f).get("files", []))
        except FileNotFoundError:
            continue
    return consumed


def _next_cursor_from_records(run_root: str, epoch: int, rank: int) -> int:
    if not run_root:
        return 0
    d = os.path.join(run_root, f"epoch{epoch}")
    if not os.path.isdir(d):
        return 0
    import re

    pat = re.compile(rf"sample{rank}_{epoch}_(\d+)_step\d+$")
    best = -1
    for name in os.listdir(d):
        m = pat.fullmatch(name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def online_train(adapter, auditor, teacher, trainer, seeds, cfg, accelerator=None,
                 out_dir=None, save_images=True, log_every=10, shuffle_seed=None,
                 start_epoch=0) -> list[float]:
    metrics = Metrics()
    losses = []
    acc = accelerator
    rank = acc.process_index if acc is not None else 0
    if acc is not None and len(seeds) < acc.num_processes:
        raise ValueError(
            f"need at least world_size={acc.num_processes} seeds, got {len(seeds)}"
        )
    resume_from = getattr(getattr(cfg, "train", None), "resume_from", "") or ""
    resume_step = _resume_step_from_checkpoint(resume_from)
    resume_root = _run_root_from_checkpoint(resume_from)
    resume_pool_root = os.path.join(resume_root, "pool") if resume_root else ""
    if resume_from and os.path.isdir(resume_from):
        _pj = os.path.join(resume_from, "progress.json")
        if os.path.exists(_pj):
            try:
                _pp = json.load(open(_pj)).get("pool_root", "")
                if _pp:
                    resume_pool_root = os.path.abspath(_pp) if not os.path.isabs(_pp) else _pp
            except Exception:
                pass
    reuse_resume_pool = bool(resume_step > 0 and os.path.isdir(resume_pool_root))
    pool_samples = pool_rounds = pool_flags = pool_health = None
    world = ga = B = 1
    if acc is not None:
        world = acc.num_processes
        ga = max(1, int(getattr(cfg.train, "grad_accum", 1)))
        B = world * ga
        if reuse_resume_pool:
            pool_root = resume_pool_root
        else:
            pool_root = os.path.join(out_dir, "pool") if out_dir else None
            if pool_root is None:
                import tempfile
                pool_root = tempfile.mkdtemp(prefix="prism_pool_")
        pool_samples = os.path.join(pool_root, "samples")
        pool_rounds = os.path.join(pool_root, "rounds")
        pool_flags = os.path.join(pool_root, "flags")
        pool_health = os.path.join(pool_root, "health")
        if rank == 0:
            import shutil
            if reuse_resume_pool or os.environ.get("PRISM_KEEP_POOL") == "1":
                shutil.rmtree(pool_flags, ignore_errors=True)
            else:
                shutil.rmtree(pool_root, ignore_errors=True)
        dirs = [pool_samples, pool_rounds, pool_flags, pool_health]
        for d in dirs:
            if rank == 0:
                os.makedirs(d, exist_ok=True)
        if hasattr(acc, "wait_for_everyone"):
            acc.wait_for_everyone()
    if acc is not None and hasattr(acc, "device"):
        try:
            import torch as _t
            acc.gather(_t.tensor([rank], device=acc.device))
        except Exception:
            pass
    seed_base = int(shuffle_seed or 0)
    train_log = None
    if out_dir and trainer._is_main():
        os.makedirs(out_dir, exist_ok=True)
        train_log = os.path.join(out_dir, "train_log.jsonl")
        if start_epoch == 0 and resume_step <= 0:
            open(train_log, "w").close()
        need_config = start_epoch == 0 and (
            resume_step <= 0 or not os.path.exists(train_log) or os.path.getsize(train_log) == 0
        )
        if need_config:
            _g = lambda o, k, d=None: getattr(o, k, d) if o is not None else d
            _tr, _sa, _au = (getattr(cfg, n, None) for n in ("train", "sample", "audit"))
            _te = getattr(cfg, "teach", None)
            _append_jsonl(train_log, {
                "ts": _ts(), "record": "config",
                "seeds": len(seeds), "epochs": _g(_tr, "epochs"),
                "grad_accum": _g(_tr, "grad_accum"), "global_batch": B, "world": world,
                "lr": _g(_tr, "lr"),
                "class_weights": _g(_tr, "class_weights"),
                "lora_rank": _g(_tr, "lora_rank"),
                "lora_alpha": _g(_tr, "lora_alpha"),
                "image_size": list(_g(_sa, "image_size") or ()),
                "num_steps": _g(_sa, "num_steps"),
                "cfg_scale": _g(_sa, "cfg_scale"),
                "regen_num_steps": _g(_te, "regen_num_steps"),
                "think_cap_text": _g(_au, "think_cap_text"),
                "think_cap_image": _g(_au, "think_cap_image"),
                "answer_tries": _g(_au, "answer_tries"),
                "shuffle_seed": shuffle_seed,
            })
        if resume_step > 0:
            _append_jsonl(train_log, {"ts": _ts(), "record": "resume", "resume_from": resume_from,
                                      "resume_step": resume_step, "reuse_pool": reuse_resume_pool,
                                      "resume_pool": resume_pool_root if reuse_resume_pool else ""})
    win_loss: list = []
    win_kind: list = []
    win_global: dict = {}
    upd = [resume_step]

    def _train(sample, rec_dir, epoch, ref):
        lv = trainer.train_step(sample)
        if lv is None:
            return None
        losses.append(lv)
        metrics.record_train(sample.kind, lv, trainer.last_grad_norm)
        if rec_dir:
            _patch_train(rec_dir, sample.kind, lv, trainer.last_grad_norm, trainer.last_lr)
        win_loss.append(lv)
        win_kind.append(sample.kind)
        if getattr(trainer, "last_stepped", True):
            upd[0] += 1
            avg = sum(win_loss) / len(win_loss)
            micro = len(win_loss)
            kinds = dict(Counter(win_kind))
            if acc is not None and hasattr(acc, "gather") and hasattr(acc, "device"):
                try:
                    import torch as _t
                    avg = acc.gather(_t.tensor([avg], device=acc.device)).mean().item()
                    micro = micro * acc.num_processes
                except Exception:
                    pass
            if win_global.get("kinds"):
                kinds = win_global.pop("kinds")
            if train_log:
                _append_jsonl(train_log, {"ts": _ts(), "update": upd[0], "epoch": epoch,
                                          "loss": round(avg, 6), "grad_norm": trainer.last_grad_norm,
                                          "lr": trainer.last_lr, "micro": micro,
                                          "kinds": kinds,
                                          "health": _rollout_health(pool_health, upd[0] - 1)})
            _log(trainer, f"[update {upd[0]}] ep={epoch} loss={avg:.5f} "
                          f"gnorm={_fmt(trainer.last_grad_norm)} lr={trainer.last_lr:.2e} "
                          f"micro={micro} kinds={kinds}")
            win_loss.clear()
            win_kind.clear()
            sev = int(getattr(cfg.train, "save_every_steps", 0) or 0)
            if out_dir and sev > 0 and upd[0] % sev == 0:
                _ckpt_dir = os.environ.get("PRISM_CKPT_ROOT") or out_dir
                ckpt = os.path.join(_ckpt_dir, "checkpoints", f"step{upd[0]}")
                os.makedirs(ckpt, exist_ok=True)
                trainer.save_state(ckpt)
                if trainer._is_main():
                    os.makedirs(ckpt, exist_ok=True)
                    with open(os.path.join(ckpt, "progress.json"), "w", encoding="utf-8") as f:
                        json.dump({"next_epoch": epoch, "update": upd[0], "next_round": round_no[0],
                                   "pool_root": pool_root if acc is not None else "",
                                   "reuse_pool": reuse_resume_pool}, f)
                    keep = int(getattr(cfg.train, "keep_step_ckpts", 2) or 0)
                    if keep > 0:
                        import re as _re, shutil as _shutil
                        _pinned = {int(x) for x in
                                   os.environ.get("PRISM_KEEP_STEPS", "").replace(",", " ").split()
                                   if x.strip().isdigit()}
                        root = os.path.join(_ckpt_dir, "checkpoints")
                        olds = sorted((int(m.group(1)), n) for n in os.listdir(root)
                                      for m in [_re.fullmatch(r"step(\d+)", n)] if m)
                        for _num, name in olds[:-keep]:
                            if _num in _pinned:
                                continue
                            _shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                _log(trainer, f"[update {upd[0]}] step-checkpoint -> {ckpt}")
        return lv

    _keep_pool = os.environ.get("PRISM_KEEP_POOL") == "1"
    consumed: set = _consumed_from_rounds(pool_rounds, resume_step) if reuse_resume_pool else set()
    round_no = [resume_step if resume_step > 0 else 0]
    if trainer._is_main() and (resume_step > 0 or _keep_pool):
        _log(trainer, f"[resume] step={resume_step} reuse_pool={reuse_resume_pool} "
                      f"consumed={len(consumed)} next_round={round_no[0]}")

    def _pool_round(epoch):
        if hasattr(acc, "wait_for_everyone"):
            acc.wait_for_everyone()
        mpath = os.path.join(pool_rounds, f"round{round_no[0]}.json")
        if rank == 0 and not os.path.exists(mpath):
            take = _pool_avail(pool_samples, consumed)[:B]
            with open(mpath + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"files": take}, f)
            os.replace(mpath + ".tmp", mpath)
        if hasattr(acc, "wait_for_everyone"):
            acc.wait_for_everyone()
        with open(mpath, encoding="utf-8") as f:
            names = json.load(f)["files"]
        gk = Counter(n.rsplit("_", 1)[-1][:-4] for n in names)
        win_global["kinds"] = {k: v for k, v in gk.items() if k in _KINDS}
        cls_w = _class_weights(win_global["kinds"], getattr(cfg.train, "class_weights", None))
        import pickle
        for n in names[rank * ga:(rank + 1) * ga]:
            with open(os.path.join(pool_samples, n), "rb") as f:
                s = pickle.load(f)
            s.meta["_loss_w"] = cls_w.get(s.kind, 1.0)
            lv = _train(s, s.meta.get("_rec_dir"), epoch, s.meta.get("_ref"))
            if lv is not None:
                _log(trainer, f"ep={epoch} round{round_no[0]} {s.kind} w={s.meta['_loss_w']:.3f} "
                              f"loss={lv:.5f} gnorm={_fmt(trainer.last_grad_norm)} lr={trainer.last_lr:.2e}")
        consumed.update(names)
        round_no[0] += 1

    for epoch in range(start_epoch, cfg.train.epochs):
        metrics.set_epoch(epoch)
        ep_seeds = seeds
        if shuffle_seed is not None:
            import random
            ep_seeds = list(seeds)
            random.Random(shuffle_seed + epoch).shuffle(ep_seeds)
        _log(trainer, f"===== epoch {epoch+1}/{cfg.train.epochs} start "
                      f"(seeds={len(ep_seeds)}, shuffle={'on' if shuffle_seed is not None else 'off'}) =====")
        if acc is None:
            fails = 0
            for i, seed in enumerate(ep_seeds):
                gseed = seed_base + epoch * 1_000_003 + i * 9973
                try:
                    traj, audit, decision, sample, stages = rollout(adapter, auditor, teacher, seed, cfg, gen_seed=gseed)
                except Exception as e:
                    fails += 1
                    _log(trainer, f"[rollout-FAIL#{fails}] ep={epoch} sample{i} seed={seed.get('id')}: {e!r}")
                    if out_dir:
                        import traceback
                        os.makedirs(os.path.join(out_dir, f"epoch{epoch}"), exist_ok=True)
                        _append_jsonl(os.path.join(out_dir, f"epoch{epoch}", "rollout_failures.jsonl"),
                                      {"cursor": i, "seed_id": seed.get("id"), "error": repr(e),
                                       "tb": traceback.format_exc()[-2000:]})
                    if fails > max(5, len(ep_seeds) // 10):
                        raise
                    continue
                metrics.update(audit, decision, sample, skip_reason=getattr(teacher, "last_skip_reason", ""),
                               degen=any(s.get("stage") == "degen" for s in stages),
                               iplus_verify=getattr(teacher, "last_iplus_verify", {}),
                               tplus_verify=getattr(auditor, "last_tplus_verify", {}))
                name = f"sample{i}_step{upd[0]}"
                rec_dir = _write_sample(out_dir, epoch, name, seed, traj, audit, decision, sample, save_images,
                                     skip_reason=getattr(teacher, "last_skip_reason", ""), gate_reason=getattr(teacher, "last_gate_reason", ""), stages=stages) if out_dir else None
                if sample is not None:
                    lv = _train(sample, rec_dir, epoch, name)
                    if lv is not None:
                        _log(trainer, f"ep={epoch} sample{i} {sample.kind} quad={decision.quadrant.value} "
                                      f"loss={lv:.5f} gnorm={_fmt(trainer.last_grad_norm)} lr={trainer.last_lr:.2e}")
                else:
                    _log(trainer, f"ep={epoch} sample{i} quad={decision.quadrant.value} (skip)")
                if (i + 1) % log_every == 0:
                    _log(trainer, metrics.format())
        else:
            local = ep_seeds[rank::world]
            produced = 0
            fails = 0
            rounds_at_start = round_no[0]
            while os.path.exists(os.path.join(pool_rounds, f"round{round_no[0]}.json")) or \
                    len(_pool_avail(pool_samples, consumed)) >= B:
                _pool_round(epoch)
            if (reuse_resume_pool and resume_step > 0) or _keep_pool:
                start_cursor = _next_cursor_from_records(resume_root or out_dir, epoch, rank)
                if start_cursor:
                    _log(trainer, f"[resume] rank{rank} skip_existing_rollouts={start_cursor}/{len(local)}")
            else:
                start_cursor = 0
            for cursor, seed in enumerate(local[start_cursor:], start=start_cursor):
                gseed = seed_base + epoch * 1_000_003 + rank * 100_003 + cursor * 9973
                try:
                    traj, audit, decision, sample, stages = rollout(adapter, auditor, teacher, seed, cfg, gen_seed=gseed)
                except Exception as e:
                    fails += 1
                    _log(trainer, f"[rollout-FAIL#{fails}] ep={epoch} rank{rank} cursor={cursor} seed={seed.get('id')}: {e!r}")
                    if out_dir:
                        import traceback
                        os.makedirs(os.path.join(out_dir, f"epoch{epoch}"), exist_ok=True)
                        _append_jsonl(os.path.join(out_dir, f"epoch{epoch}", "rollout_failures.jsonl"),
                                      {"rank": rank, "cursor": cursor, "seed_id": seed.get("id"), "error": repr(e),
                                       "tb": traceback.format_exc()[-2000:]})
                    if fails > max(5, len(local) // 10):
                        raise
                    while len(_pool_avail(pool_samples, consumed)) >= B:
                        _pool_round(epoch)
                    continue
                metrics.update(audit, decision, sample, skip_reason=getattr(teacher, "last_skip_reason", ""),
                               degen=any(s.get("stage") == "degen" for s in stages),
                               iplus_verify=getattr(teacher, "last_iplus_verify", {}),
                               tplus_verify=getattr(auditor, "last_tplus_verify", {}))
                name = f"sample{rank}_{epoch}_{cursor}_step{upd[0]}"
                rec_dir = _write_sample(out_dir, epoch, name, seed, traj, audit, decision, sample, save_images,
                              skip_reason=getattr(teacher, "last_skip_reason", ""), gate_reason=getattr(teacher, "last_gate_reason", ""), stages=stages) if out_dir else None
                if out_dir:
                    _raw = traj.raw_output or ""
                    _append_jsonl(os.path.join(pool_health, f"rank{rank}.jsonl"),
                                  {"step": upd[0], "n_img": len(traj.images or []),
                                   "closed": "</think>" in _raw, "len": len(_raw),
                                   "quad": decision.quadrant.value})
                if sample is not None:
                    sample.meta["_rec_dir"] = rec_dir
                    sample.meta["_ref"] = name
                    _pool_put(pool_samples, rank, epoch * 1_000_000 + cursor, sample)
                    produced += 1
                if trainer._is_main():
                    print(f"[{_ts()}] [rollout] ep={epoch} rank0 produced={produced} "
                          f"(rolled {cursor + 1}/{len(local)}, pool={len(_pool_avail(pool_samples, consumed))})", flush=True)
                while len(_pool_avail(pool_samples, consumed)) >= B:
                    _pool_round(epoch)
            open(os.path.join(pool_flags, f"done_ep{epoch}_r{rank}"), "w").close()
            while True:
                if len(_pool_avail(pool_samples, consumed)) >= B:
                    _pool_round(epoch)
                    continue
                n_done = sum(1 for n in os.listdir(pool_flags) if n.startswith(f"done_ep{epoch}_r"))
                if n_done >= world:
                    break
                time.sleep(float(os.environ.get("PRISM_POOL_POLL_S", "10")))
            ep_rounds = round_no[0] - rounds_at_start
            _log(trainer, f"ep={epoch} rolled_out={len(local)} produced={produced} rounds={ep_rounds} "
                          f"trained={ep_rounds * ga}/rank leftover_pool={len(_pool_avail(pool_samples, consumed))} "
                          + metrics.format())

        sf = max(1, int(getattr(cfg.train, "save_freq", 1)))
        if out_dir and (epoch + 1) % sf == 0 and upd[0] > 0:
            _ckpt_dir = os.environ.get("PRISM_CKPT_ROOT") or out_dir
            ckpt = os.path.join(_ckpt_dir, "checkpoints", f"epoch{epoch}")
            os.makedirs(ckpt, exist_ok=True)
            trainer.save_state(ckpt)
            if trainer._is_main():
                os.makedirs(ckpt, exist_ok=True)
                with open(os.path.join(ckpt, "progress.json"), "w", encoding="utf-8") as f:
                    json.dump({"next_epoch": epoch + 1, "update": upd[0], "next_round": round_no[0],
                               "pool_root": pool_root if acc is not None else "",
                               "reuse_pool": reuse_resume_pool}, f)
            _log(trainer, f"ep={epoch} checkpoint -> {ckpt}")

    if trainer._is_main():
        print(f"[{_ts()}] [online] FINAL " + metrics.format(), flush=True)
        if out_dir:
            with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics.summary(), f, indent=2)
    return losses
