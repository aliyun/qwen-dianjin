
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Optional

from .core.protocol import refs_valid, render_user_images, strip_image_references
from .core.router import RoutingDecision
from .core.types import Audit, InterleaveStep, TeachSample, Trajectory


class Teacher:
    def __init__(
        self,
        adapter: Any,
        *,
        judge: Any = None,
        regen_num_steps: int = 50,
        regen_image_size: Optional[tuple] = None,
        strict_desc: bool = True,
        max_images: int = 10,
        verify_iplus: bool = False,
        verify_max_retries: int = 3,
    ) -> None:
        self.adapter = adapter
        self.judge = judge
        self.regen_num_steps = regen_num_steps
        self.regen_image_size = tuple(regen_image_size) if regen_image_size else None
        self.strict_desc = strict_desc
        self.max_images = max_images
        self.verify_iplus = verify_iplus
        self.verify_max_retries = max(0, int(verify_max_retries))
        self.last_skip_reason = ""
        self.last_gate_reason = ""
        self.last_iplus_verify: dict = {}

    def teach(self, traj: Trajectory, audit: Audit, decision: RoutingDecision,
             gen_seed: Optional[int] = None) -> Optional[TeachSample]:
        self.last_skip_reason = ""
        self.last_gate_reason = ""
        self.last_iplus_verify = {}
        self.last_stages = []
        if decision.action == "skip":
            return None
        if decision.action == "neg":
            return self._neg(traj, audit, decision)
        if decision.action == "t2i":
            return self._t2i(traj, audit, decision, gen_seed=gen_seed)
        if decision.action == "i2t":
            return self._i2t(traj, audit, decision)
        if decision.action == "s2t":
            return self._s2t(traj, audit, decision)
        return None

    def _base(self, traj: Trajectory, audit: Audit, decision: RoutingDecision, **kw) -> dict:
        return dict(
            prompt=traj.prompt,
            quadrant=decision.quadrant.value,
            audit={"text": asdict(audit.text), "image": asdict(audit.image)},
            **kw,
        )

    def _t2i(self, traj, audit, decision, gen_seed: Optional[int] = None) -> Optional[TeachSample]:
        gt_text = (traj.think_text or traj.reasoning_text()) if traj.think_mode else (traj.raw_output or traj.reasoning_text())
        n_img = len(traj.images)
        desc = (audit.text.description or "").strip()
        cond_src = "desc_T"
        if not desc or desc.count("<image>") != n_img:
            if self.strict_desc:
                self.last_skip_reason = "desc_fail_t2i"
                return None
            desc = gt_text
            cond_src = "think_text"
        cond = traj.input_images
        max_attempts = 1 + (self.verify_max_retries if self.verify_iplus else 0)
        imgs, iplus_passed, iplus_attempts = None, False, 0
        for attempt in range(max_attempts):
            regen_kw = {"num_steps": self.regen_num_steps, "max_images": self.max_images}
            if self.regen_image_size:
                regen_kw["image_size"] = self.regen_image_size
            if self.verify_iplus and gen_seed is not None:
                regen_kw["seed"] = int(gen_seed) + attempt * 7919
            _t0 = time.time()
            ctx = self.adapter.frozen_base_context() if hasattr(self.adapter, "frozen_base_context") else None
            if ctx is None:
                cand = self.adapter.regenerate_images(traj.prompt, desc, images=cond, **regen_kw)
            else:
                with ctx:
                    cand = self.adapter.regenerate_images(traj.prompt, desc, images=cond, **regen_kw)
            self.last_stages.append({"stage": "regen_iplus", "attempt": attempt,
                                     "seconds": round(time.time() - _t0, 1),
                                     "n_images": len(cand or []), "num_steps": self.regen_num_steps,
                                     "teacher": "frozen_base" if ctx is not None else "adapter_default",
                                     "out_field": "images/tgt_*"})
            iplus_attempts += 1
            if not cand:
                continue
            imgs = cand
            if not self.verify_iplus or self.judge is None:
                iplus_passed = True
                break
            _tv = time.time()
            v = self.judge.audit_image(traj.prompt, cand)
            self.last_stages.append({"stage": "iplus_verify", "attempt": attempt,
                                     "seconds": round(time.time() - _tv, 1),
                                     "valid": v.valid, "reason": (v.reason or "")[:200]})
            if v.valid:
                iplus_passed = True
                break
        self.last_iplus_verify = {"used": bool(self.verify_iplus), "attempts": iplus_attempts, "passed": iplus_passed}
        if self.verify_iplus and not iplus_passed:
            self.last_skip_reason = "iplus_verify_failed"
            return None
        if not imgs:
            self.last_skip_reason = "regen_empty_t2i"
            return None
        if len(imgs) != n_img:
            self.last_skip_reason = "regen_count_mismatch_t2i"
            self.last_iplus_verify["count_mismatch"] = {"expected": n_img, "got": len(imgs)}
            return None
        segs = traj.raw_segments()
        if len(segs) != len(imgs) + 1:
            self.last_skip_reason = "raw_segment_mismatch_t2i"
            return None
        steps = [InterleaveStep(text=segs[i], image=imgs[i]) for i in range(len(imgs))]
        steps.append(InterleaveStep(text=segs[-1], image=None))
        return TeachSample(
            kind="t2i",
            steps=steps,
            reliable_side="text",
            context_text=gt_text,
            cond_images=list(cond),
            target_image=imgs,
            meta={
                "step_texts": [s.text for s in traj.steps],
                "captions": traj.captions(),
                "desc_T": decision.teach_description,
                "iplus_cond": cond_src,
                "iplus_verify_attempts": iplus_attempts,
                "iplus_verify_passed": iplus_passed,
            },
            **self._base(traj, audit, decision),
        )

    def _i2t(self, traj, audit, decision) -> Optional[TeachSample]:
        n_imgs = len(traj.images)
        tplus = (decision.teach_description or "").strip()
        if not tplus:
            self.last_skip_reason = "desc_fail_i2t"
            return None
        if not traj.think_mode:
            return self._i2t_nothink(traj, audit, decision, tplus, n_imgs)
        tplus_info = getattr(audit.image, "tplus_info", {}) or {}
        tplus_decision = tplus_info.get("decision")
        if tplus_decision in {"kept_original", "no_think_split"}:
            self.last_skip_reason = f"{tplus_decision}_i2t"
            return None
        if "</think>" in tplus:
            think_body = tplus.split("</think>", 1)[0].split("<think>", 1)[-1]
            summary = tplus.split("</think>", 1)[1].strip()
        else:
            think_body, summary = tplus.replace("<think>", "", 1), ""
        segs = think_body.split("<image>")
        if n_imgs and len(segs) != n_imgs + 1:
            self.last_skip_reason = "desc_fail_i2t"
            return None
        if not refs_valid(summary, n_imgs):
            self.last_skip_reason = "bad_refs_i2t"
            return None
        plain = (think_body.replace("<image>", " ") + "\n" + summary).strip()
        orig = traj.reasoning_text() or ""
        if len(plain) < 0.5 * len(orig):
            self.last_skip_reason = "short_i2t"
            return None
        if n_imgs:
            steps = [InterleaveStep(text=("<think>\n" + segs[i] if i == 0 else segs[i]),
                                    image=traj.images[i]) for i in range(n_imgs)]
            steps.append(InterleaveStep(text=segs[-1].rstrip() + "\n</think>\n\n" + summary, image=None))
        else:
            steps = [InterleaveStep(text=tplus, image=None)]
        return TeachSample(
            kind="i2t",
            steps=steps,
            reliable_side="image",
            cond_images=list(traj.input_images),
            target_text=plain,
            meta={"desc_I": decision.teach_description, "chain_images": n_imgs, "tplus_cond": "desc_I",
                  "tplus_decision": tplus_decision or "unknown"},
            **self._base(traj, audit, decision),
        )

    def _i2t_nothink(self, traj, audit, decision, tplus: str, n_imgs: int) -> Optional[TeachSample]:
        import re

        numbered_refs_normalized = bool(re.search(r"<image\d+>", tplus or "", flags=re.I))
        if numbered_refs_normalized:
            tplus = re.sub(r"<image\d+>", "<image>", tplus, flags=re.I)
        segs = tplus.split("<image>")
        if n_imgs and len(segs) != n_imgs + 1:
            self.last_skip_reason = "desc_fail_i2t"
            return None
        tail = segs[-1].strip() if len(segs) > 1 else tplus.strip()
        if not refs_valid(tail, n_imgs):
            self.last_skip_reason = "bad_refs_i2t"
            return None
        plain = tplus.replace("<image>", " ").strip()
        orig = (traj.raw_output or "").replace("<image>", " ").strip()
        if len(plain) < 0.5 * len(orig):
            self.last_skip_reason = "short_i2t"
            return None
        if n_imgs:
            steps = [InterleaveStep(text=segs[i], image=traj.images[i]) for i in range(n_imgs)]
            steps.append(InterleaveStep(text=segs[-1], image=None))
        else:
            steps = [InterleaveStep(text=tplus, image=None)]
        return TeachSample(
            kind="i2t",
            steps=steps,
            reliable_side="image",
            cond_images=list(traj.input_images),
            target_text=plain,
            meta={"desc_I": decision.teach_description, "chain_images": n_imgs,
                  "tplus_cond": "desc_I", "no_think": True,
                  "numbered_refs_normalized": numbered_refs_normalized,
                  "tplus_decision": (getattr(audit.image, "tplus_info", {}) or {}).get("decision", "unknown")},
            **self._base(traj, audit, decision),
        )

    S2T_RECOVERY_PROMPT = (
        "The images above are the record of working through the task below, in order. "
        "Using them as that record, write the complete response the task calls for — how "
        "the story unfolds, or the step-by-step tutorial, or whatever the task asks to be "
        "produced — covering the progression from the first image through to the last. "
        "Write only that response, with no planning notes."
    )

    def _s2t(self, traj, audit, decision) -> Optional[TeachSample]:
        imgs = list(traj.images)
        if not imgs:
            return None
        raw = (traj.raw_output or "").strip()
        if traj.think_mode:
            target = (traj.final_answer or "").strip()
        else:
            from .core.protocol import strip_all_image_markers
            target = strip_all_image_markers(raw)
        if not target or target == raw:
            return None
        if traj.think_mode:
            target = strip_image_references(target).strip()
            if not target:
                return None
        base = self._base(traj, audit, decision)
        base["prompt"] = (f"{render_user_images(len(imgs))}\n"
                          f"{self.S2T_RECOVERY_PROMPT}\n\nOriginal task:\n{traj.prompt}")
        return TeachSample(
            kind="s2t",
            steps=[InterleaveStep(text=target, image=None)],
            reliable_side="image",
            cond_images=imgs,
            target_text=target,
            meta={"orig_prompt": traj.prompt, "chain_images": len(imgs),
                  "layout": "recovery_answer_only_no_refs"},
            **base,
        )

    def _neg(self, traj, audit, decision) -> Optional[TeachSample]:
        from .core.degen import truncate_repetition

        raw = (traj.raw_output or "").strip()
        degen_meta = None
        if raw:
            cut, det = truncate_repetition(raw)
            if det and cut.count("<image>") == raw.count("<image>"):
                raw = cut
                degen_meta = det
            import re
            segs = re.split(r"<image>", raw)
            steps = [InterleaveStep(text=segs[i], image=(traj.images[i] if i < len(traj.images) else None))
                     for i in range(len(segs) - 1)]
            steps.append(InterleaveStep(text=segs[-1], image=None))
        else:
            txt = traj.reasoning_text().strip()
            if not txt:
                return None
            steps = [InterleaveStep(text=txt, image=None)]
        neg_text, _ = truncate_repetition(traj.reasoning_text())
        return TeachSample(
            kind="neg",
            steps=steps,
            cond_images=list(traj.input_images),
            neg_text=neg_text,
            meta={"reason_text": audit.text.reason, "reason_image": audit.image.reason,
                  "degen": degen_meta},
            **self._base(traj, audit, decision),
        )

