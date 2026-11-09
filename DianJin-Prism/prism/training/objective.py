from __future__ import annotations

from typing import Optional

import torch

from ..core.types import TeachSample


def quadrant_loss_weights(kind: str) -> dict:
    table = {
        "t2i": {"ce": 0.0, "fm": 1.0, "neg": 0.0},
        "i2t": {"ce": 1.0, "fm": 0.0, "neg": 0.0},
        "s2t": {"ce": 1.0, "fm": 0.0, "neg": 0.0},
        "neg": {"ce": 0.0, "fm": 0.0, "neg": 1.0},
    }
    return table.get(kind, {"ce": 0.0, "fm": 0.0, "neg": 0.0})


def _loss_w(sample: TeachSample) -> float:
    try:
        return float((sample.meta or {}).get("_loss_w", 1.0))
    except Exception:
        return 1.0


def compute_loss(adapter, sample: TeachSample, cfg) -> Optional[torch.Tensor]:
    if sample.kind not in ("t2i", "i2t", "s2t", "neg"):
        return None
    out = adapter.interleaved_teach_loss(sample, objective="sft")
    loss = out["loss"] if isinstance(out, dict) else out
    return None if loss is None else loss * _loss_w(sample)
