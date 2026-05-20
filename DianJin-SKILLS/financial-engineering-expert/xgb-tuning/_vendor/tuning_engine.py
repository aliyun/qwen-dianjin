# -*- coding: utf-8 -*-
"""兼容门面 — 全部实现已迁移至 tuning/ 子包。

所有 `from tuning_engine import X` 的调用方无需修改。
"""
from tuning.diagnose import (
    DiagnosisResult,
    diagnose_model,
    _DIAG_THRESHOLDS,
)
from tuning.search_space import (
    PARAM_BOUNDS,
    build_params_delta,
    _build_constrained_space,
    _default_batch_space,
    _clip,
)
from tuning.engine import TuningEngine

__all__ = [
    "DiagnosisResult",
    "diagnose_model",
    "_DIAG_THRESHOLDS",
    "PARAM_BOUNDS",
    "build_params_delta",
    "_build_constrained_space",
    "_default_batch_space",
    "_clip",
    "TuningEngine",
]
