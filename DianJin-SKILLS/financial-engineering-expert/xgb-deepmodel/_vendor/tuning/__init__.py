# -*- coding: utf-8 -*-
"""shared/tuning — 多算法超参调优子包。

架构：
  - engine       : BaseTuningEngine 抽象基类 + 向后兼容别名
  - xgb_engine   : XGBTuningEngine（XGBoost 专用）
  - lr_engine    : LRTuningEngine（LR+WoE 联合搜索）
  - dnn_engine   : DNNTuningEngine（PyTorch MLP）
  - diagnose     : 算法无关的诊断模块
  - search_space : XGB 搜索空间构造
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
from tuning.engine import BaseTuningEngine, TuningEngine
from tuning.xgb_engine import XGBTuningEngine
from tuning.lr_engine import LRTuningEngine
from tuning.dnn_engine import DNNTuningEngine

__all__ = [
    "BaseTuningEngine",
    "TuningEngine",
    "XGBTuningEngine",
    "LRTuningEngine",
    "DNNTuningEngine",
    "DiagnosisResult",
    "diagnose_model",
    "_DIAG_THRESHOLDS",
    "PARAM_BOUNDS",
    "build_params_delta",
    "_build_constrained_space",
    "_default_batch_space",
    "_clip",
]
