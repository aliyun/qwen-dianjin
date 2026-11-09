
from __future__ import annotations

import re
import time
from typing import Any, Optional

from ..core.types import Audit, AuditResult, Trajectory
from .judges import Judge

_WS = re.compile(r"\s+")
_MIN_QUOTE_CHARS = 12


def _quote_in_text(quote: str, text: str) -> bool:
    q = _WS.sub(" ", (quote or "")).strip()
    if len(q) < _MIN_QUOTE_CHARS:
        return False
    return q in _WS.sub(" ", text or "")


class Auditor:
    def __init__(self, judge: Judge, *, gen_desc: bool = True,
                 verify_tplus: bool = False, verify_max_retries: int = 3,
                 verify_temperature_step: float = 0.2) -> None:
        self.judge = judge
        self.gen_desc = gen_desc
        self.verify_tplus = verify_tplus
        self.verify_max_retries = max(0, int(verify_max_retries))
        self.verify_temperature_step = verify_temperature_step
        self.last_tplus_verify: dict = {}

    def audit(self, traj: Trajectory) -> Audit:
        text_chain = (traj.raw_output or "").replace("<think>", "").replace("</think>", "").strip()
        if not text_chain:
            parts = [p for p in (traj.reasoning_text(), traj.final_answer) if p]
            text_chain = "\n\n".join(parts) or traj.prompt

        def _st(name, res, t0):
            self.last_stages.append({
                "stage": name, "seconds": round(time.time() - t0, 1), "valid": res.valid,
                "retried": bool(getattr(self.judge, "last_audit_retried", False)),
                "finalizer": "[finalizer]" in (res.raw or ""),
                "reason_len": len(res.reason or ""), "raw_len": len(res.raw or ""),
                "out_field": name,
            })

        self.last_stages = []
        self.last_tplus_verify = {}
        t0 = time.time()
        a_text = self.judge.audit_text(traj.prompt, text_chain, think_mode=traj.think_mode)
        _st("audit_text", a_text, t0)
        t0 = time.time()
        if not traj.images:
            a_image = AuditResult(valid=False, reason="no_images_generated: rollout produced 0 images")
        else:
            a_image = self.judge.audit_image(traj.prompt, traj.images)
        _st("audit_image", a_image, t0)

        if self.gen_desc:
            if a_text.valid and not a_image.valid and not (a_text.description or "").strip():
                t0 = time.time()
                findings = (a_image.reason or "")
                raw = traj.raw_output or ""
                if "<think>" in raw and "</think>" in raw:
                    desc_chain = raw.split("</think>", 1)[0].split("<think>", 1)[-1].strip()
                else:
                    desc_chain = text_chain
                from .parse import trim_to_last_image, slots_to_interleaved
                desc_chain = trim_to_last_image(desc_chain)
                slot_desc = self.judge.text_desc(traj.prompt, desc_chain, findings, think=traj.think_mode)
                a_text.description = slots_to_interleaved(slot_desc)
                self.last_stages.append({
                    "stage": "desc_T", "seconds": round(time.time() - t0, 1),
                    "attempts": getattr(self.judge, "last_desc_attempts", None),
                    "out_len": len(a_text.description or ""), "ok": bool((a_text.description or "").strip()),
                    "out": a_text.description,
                    "out_field": "audit_text.description",
                    "findings_from": "audit_image.reason", "findings": findings[:200],
                })
            elif a_image.valid and not a_text.valid and not (a_image.description or "").strip():
                findings = (a_text.reason or "")
                q_ok = _quote_in_text(a_text.quote, text_chain)
                max_attempts = 1 + (self.verify_max_retries if self.verify_tplus else 0)
                tplus, rep = "", {}
                tplus_attempts, tplus_passed = 0, False
                for attempt in range(max_attempts):
                    temp = min(1.0, 0.6 + attempt * self.verify_temperature_step)
                    t0 = time.time()
                    if traj.think_mode:
                        cand, cand_rep = self.tplus_by_repair(traj, a_text.reason or "", a_text.quote or "",
                                                              temperature=temp)
                        stage_name = "tplus_repair"
                    else:
                        cand = self.judge.image_desc(traj.prompt, list(traj.images), findings,
                                                     think=False, temperature=temp)
                        cand_rep = {"located": "not_used_no_think", "repair": "full_rewrite",
                                    "decision": "rewritten_from_images" if cand.strip() else "rewrite_empty"}
                        stage_name = "tplus_rewrite"
                    self.last_stages.append({
                        "stage": stage_name, "attempt": attempt, "seconds": round(time.time() - t0, 1),
                        "located": cand_rep.get("located"), "repair": cand_rep.get("repair"),
                        "decision": cand_rep.get("decision"),
                        "answer_attempts": cand_rep.get("answer_attempts"),
                        "out_len": len(cand), "ok": bool(cand.strip()),
                        "out": cand,
                        "out_field": "audit_image.description",
                        "findings_from": "audit_text.reason", "findings": findings[:200],
                        "quote_found": q_ok, "quote": (a_text.quote or "")[:200],
                    })
                    tplus_attempts += 1
                    if not cand.strip():
                        continue
                    tplus, rep = cand, cand_rep
                    if not self.verify_tplus or self.judge is None:
                        tplus_passed = True
                        break
                    tv0 = time.time()
                    v = self.judge.audit_text(traj.prompt, cand, think_mode=traj.think_mode)
                    self.last_stages.append({"stage": "tplus_verify", "attempt": attempt,
                                             "seconds": round(time.time() - tv0, 1),
                                             "valid": v.valid, "reason": (v.reason or "")[:200]})
                    if v.valid:
                        tplus_passed = True
                        break
                if self.verify_tplus and not tplus_passed:
                    a_image.description = ""
                    rep = {"decision": "tplus_verify_failed"}
                else:
                    a_image.description = tplus
                setattr(a_image, "tplus_info", dict(rep or {}))
                self.last_tplus_verify = {"used": bool(self.verify_tplus), "attempts": tplus_attempts,
                                          "passed": tplus_passed}

        return Audit(text=a_text, image=a_image)

    def tplus_by_repair(self, traj: Trajectory, reason: str, quote: str, *, temperature: float = 0.6) -> tuple:
        from .locate import locate_quote

        info = {"located": None, "decision": None}
        raw = traj.raw_output or ""
        n = len(traj.images)

        def _repair(text: str, span: tuple, images: list) -> Optional[str]:
            if "<image" in text[span[0]:span[1]]:
                return None
            segs, pos, idx, local = text.split("<image>"), 0, 0, span
            for i, s in enumerate(segs):
                if pos + len(s) > span[0]:
                    idx, local = i, (span[0] - pos, span[1] - pos)
                    break
                pos += len(s) + len("<image>")
            seg = segs[idx]
            if not (0 <= local[0] < local[1] <= len(seg)):
                return None
            one = [images[idx]] if idx < len(images) else []
            new = self.judge.repair_span(traj.prompt, one, reason, seg, local, temperature=temperature)
            if new is None:
                return None
            new = re.sub(r"<image\d*>", "", new).strip()
            if not new or new == text[span[0]:span[1]].strip():
                return None
            return new

        if not traj.think_mode:
            span = locate_quote(quote, raw, forbid=traj.prompt)
            if span:
                info["located"] = "full_output"
                new = _repair(raw, span, list(traj.images))
                if new:
                    raw = raw[:span[0]] + new + raw[span[1]:]
                    info["repair"] = "applied"
                else:
                    info["repair"] = "unusable"
            else:
                info["repair"] = "not_located"
            info["decision"] = "span_repaired" if info.get("repair") == "applied" else "kept_original"
            return raw.strip(), info

        think, sep, answer = raw.partition("</think>")
        if not sep:
            info["decision"] = "no_think_split"
            return "", info
        orig_answer = answer
        span_t = locate_quote(quote, think, forbid=traj.prompt)

        if span_t:
            info["located"] = "think"
            new = _repair(think, span_t, list(traj.images))
            if new:
                think = think[:span_t[0]] + new + think[span_t[1]:]
                info["repair"] = "applied"
            else:
                info["repair"] = "unusable"

        body = think.split("<think>", 1)[-1].strip()
        answer = self.judge.final_answer(traj.prompt, body, n, reason, orig_answer, base_temperature=temperature)
        info["answer_attempts"] = [len(a.split()) for a in
                                   getattr(self.judge, "last_answer_attempts", []) or []]
        info["answer_attempts_text"] = list(getattr(self.judge, "last_answer_attempts", []) or [])
        info["answer_budget"] = getattr(self.judge, "last_answer_budget", None)
        info["repair_seconds"] = getattr(self.judge, "last_repair_seconds", None)
        info["answer_seconds"] = getattr(self.judge, "last_answer_seconds", None)
        if not answer.strip():
            info["decision"] = "kept_original"
            return raw.strip(), info
        info["decision"] = "answer_rewritten"
        return think.rstrip() + "\n</think>\n\n" + answer.strip(), info
