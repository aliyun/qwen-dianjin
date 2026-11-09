
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .router import RoutingDecision
from .types import Audit, TeachSample


@dataclass
class Metrics:
    n: int = 0
    quadrant: Counter = field(default_factory=Counter)
    text_valid: int = 0
    image_valid: int = 0
    teach_kind: Counter = field(default_factory=Counter)
    gate_attempts: int = 0
    gate_accepted: int = 0
    desc_fail: Counter = field(default_factory=Counter)
    degen_n: int = 0
    verify_used: Counter = field(default_factory=Counter)
    verify_attempts: Counter = field(default_factory=Counter)
    verify_passed: Counter = field(default_factory=Counter)
    target_quality: Counter = field(default_factory=Counter)
    train_loss_sum: dict = field(default_factory=dict)
    train_steps: Counter = field(default_factory=Counter)
    grad_norm_sum: float = 0.0
    grad_norm_n: int = 0
    epoch: int = 0
    by_epoch: dict = field(default_factory=dict)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ep(self) -> dict:
        return self.by_epoch.setdefault(self.epoch, {
            "n": 0, "quadrant": Counter(), "gate_attempts": 0, "gate_accepted": 0,
            "iplus_cond": Counter(), "desc_fail": Counter(), "target_quality": Counter(),
        })

    def update(self, audit: Audit, decision: RoutingDecision, sample: Optional[TeachSample],
               skip_reason: str = "", degen: bool = False,
               iplus_verify: Optional[dict] = None, tplus_verify: Optional[dict] = None) -> None:
        self.n += 1
        self.degen_n += int(degen)
        self.quadrant[decision.quadrant.value] += 1
        self.text_valid += int(audit.text.valid)
        self.image_valid += int(audit.image.valid)
        ep = self._ep()
        ep["n"] += 1
        ep["quadrant"][decision.quadrant.value] += 1
        if decision.action in ("t2i", "i2t"):
            self.gate_attempts += 1
            self.gate_accepted += int(sample is not None)
            ep["gate_attempts"] += 1
            ep["gate_accepted"] += int(sample is not None)
        if skip_reason:
            self.desc_fail[skip_reason] += 1
            ep["desc_fail"][skip_reason] += 1
            if skip_reason in {"regen_count_mismatch_t2i", "raw_segment_mismatch_t2i", "iplus_verify_failed",
                               "kept_original_i2t", "no_think_split_i2t"}:
                self.target_quality[f"skip:{skip_reason}"] += 1
                ep["target_quality"][f"skip:{skip_reason}"] += 1
        if sample is not None:
            self.teach_kind[sample.kind] += 1
            src = (sample.meta or {}).get("iplus_cond")
            if src:
                ep["iplus_cond"][src] += 1
            meta = sample.meta or {}
            if sample.kind == "i2t":
                dec = meta.get("tplus_decision")
                if dec:
                    self.target_quality[f"i2t_tplus_decision:{dec}"] += 1
                    ep["target_quality"][f"i2t_tplus_decision:{dec}"] += 1
                if meta.get("no_think"):
                    self.target_quality["i2t_no_think_accepted"] += 1
                    ep["target_quality"]["i2t_no_think_accepted"] += 1
            if sample.kind == "t2i":
                n_img = len(sample.target_image) if isinstance(sample.target_image, list) else int(sample.target_image is not None)
                self.target_quality[f"t2i_target_images:{n_img}"] += 1
                ep["target_quality"][f"t2i_target_images:{n_img}"] += 1
        for label, info in (("iplus", iplus_verify), ("tplus", tplus_verify)):
            if not info or not info.get("used"):
                continue
            self.verify_used[label] += 1
            self.verify_attempts[label] += int(info.get("attempts", 0))
            self.verify_passed[label] += int(bool(info.get("passed")))
            if label == "iplus" and info.get("count_mismatch"):
                self.target_quality["t2i_regen_count_mismatch"] += 1
                ep["target_quality"]["t2i_regen_count_mismatch"] += 1

    def record_train(self, kind: str, loss: float, grad_norm: Optional[float] = None) -> None:
        self.train_loss_sum[kind] = self.train_loss_sum.get(kind, 0.0) + float(loss)
        self.train_steps[kind] += 1
        if grad_norm is not None:
            self.grad_norm_sum += float(grad_norm)
            self.grad_norm_n += 1

    def summary(self) -> dict:
        n = max(1, self.n)
        q = self.quadrant
        epochs = {}
        for e in sorted(self.by_epoch):
            ep = self.by_epoch[e]
            en = max(1, ep["n"])
            eq = ep["quadrant"]
            epochs[str(e)] = {
                "n": ep["n"],
                "asym_rate": round((eq.get("t2i", 0) + eq.get("i2t", 0)) / en, 4),
                "quadrant": dict(eq),
                "gate_accept_rate": round(ep["gate_accepted"] / max(1, ep["gate_attempts"]), 4),
                "iplus_cond": dict(ep["iplus_cond"]),
                "desc_fail": dict(ep.get("desc_fail", {})),
                "target_quality": dict(ep.get("target_quality", {})),
            }
        verify = {
            k: {
                "used": self.verify_used[k],
                "attempts": self.verify_attempts[k],
                "avg_attempts": round(self.verify_attempts[k] / max(1, self.verify_used[k]), 2),
                "pass_rate": round(self.verify_passed[k] / max(1, self.verify_used[k]), 4),
            }
            for k in ("iplus", "tplus") if self.verify_used[k]
        }
        return {
            "n": self.n,
            "text_pass_rate": round(self.text_valid / n, 4),
            "image_pass_rate": round(self.image_valid / n, 4),
            "solved_rate": round(q.get("solved", 0) / n, 4),
            "t2i_rate": round(q.get("t2i", 0) / n, 4),
            "i2t_rate": round(q.get("i2t", 0) / n, 4),
            "double_fail_rate": round(q.get("unteachable", 0) / n, 4),
            "degen_rate": round(self.degen_n / n, 4),
            "gate_accept_rate": round(self.gate_accepted / max(1, self.gate_attempts), 4),
            "desc_fail": dict(self.desc_fail),
            "teach_kind": dict(self.teach_kind),
            "verify": verify,
            "target_quality": dict(self.target_quality),
            "train_steps": dict(self.train_steps),
            "avg_loss": {k: round(self.train_loss_sum[k] / self.train_steps[k], 5) for k in self.train_steps},
            "avg_grad_norm": round(self.grad_norm_sum / self.grad_norm_n, 4) if self.grad_norm_n else None,
            "epochs": epochs,
        }

    def format(self) -> str:
        s = self.summary()
        verify_seg = ""
        if s["verify"]:
            parts = ", ".join(f"{k}={v['pass_rate']}@{v['avg_attempts']}try/{v['used']}n"
                              for k, v in s["verify"].items())
            verify_seg = f"  verify=[{parts}]"
        return (
            f"[prism] n={s['n']}  solved={s['solved_rate']}  t2i={s['t2i_rate']}  "
            f"i2t={s['i2t_rate']}  double_fail={s['double_fail_rate']}  "
            f"degen={s['degen_rate']}  "
            f"gate_accept={s['gate_accept_rate']}  "
            f"text_pass={s['text_pass_rate']}  image_pass={s['image_pass_rate']}  "
            f"kinds={s['teach_kind']}  avg_loss={s['avg_loss']}  avg_gnorm={s['avg_grad_norm']}"
            f"{verify_seg}"
        )
