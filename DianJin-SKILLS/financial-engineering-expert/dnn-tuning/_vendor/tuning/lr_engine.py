# -*- coding: utf-8 -*-
"""LR 评分卡调参引擎 — LRTuningEngine。

继承 BaseTuningEngine，实现 LR+WoE 联合搜索：
  - WoE 参数（max_n_bins, iv_threshold）
  - LR 模型参数（C, regularization）

联合搜索确保 WoE 分箱策略与正则化强度的最优组合。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tuning.engine import BaseTuningEngine
from tuning.diagnose import DiagnosisResult, _DIAG_THRESHOLDS

logger = logging.getLogger(__name__)

# LR 参数硬边界
LR_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "max_n_bins": (3, 15),
    "iv_threshold": (0.005, 0.10),
    "C": (0.01, 100.0),
}

LR_REGULARIZATION_CHOICES = ["l1", "l2", "elasticnet"]

def _clip_lr(value: float, key: str) -> float:
    lo, hi = LR_PARAM_BOUNDS[key]
    return max(lo, min(hi, value))

class LRTuningEngine(BaseTuningEngine):
    """LR+WoE 联合调参引擎。

    搜索空间：
      - max_n_bins: 整数 3-15（WoE 分箱数）
      - iv_threshold: 对数 0.005-0.10（IV 筛选阈值）
      - C: 对数 0.01-100（正则化强度倒数）
      - regularization: 分类 l1/l2/elasticnet

    Args:
        X_train/y_train: 原始训练数据（未编码，WoE 在 _evaluate 内部做）
        X_eval/y_eval: 原始验证数据
        base_params: 基线参数
        metric: "auc" 或 "ks"
        features: 候选特征列表
    """

    INT_PARAMS = {"max_n_bins"}
    LOG_PARAMS = {"iv_threshold", "C"}
    CAT_PARAMS = {"regularization": LR_REGULARIZATION_CHOICES}

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_eval: pd.DataFrame,
        y_eval: np.ndarray,
        base_params: Dict[str, Any],
        metric: str = "auc",
        features: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(X_train, y_train, X_eval, y_eval, base_params, metric, sample_weight)
        self.features = features or [c for c in X_train.columns if c not in base_params.get("exclude_cols", [])]

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """WoE编码 + LR训练 + 评估。"""
        from data.woe_transform import WoEEncoder
        from xgb_core import ModelEvaluator

        max_n_bins = int(params.get("max_n_bins", 8))
        iv_threshold = float(params.get("iv_threshold", 0.02))
        C = float(params.get("C", 1.0))
        regularization = params.get("regularization", "l2")

        # WoE 编码
        woe_encoder = WoEEncoder(
            max_n_bins=max_n_bins,
            iv_threshold=iv_threshold,
            selection_mode="iv_filter",
        )
        try:
            woe_encoder.fit(self.X_train, self.y_train, self.features)
        except Exception:
            return 0.0

        if not woe_encoder.fitted_features:
            return 0.0  # 没有特征通过筛选

        X_train_woe = woe_encoder.transform(self.X_train)
        X_eval_woe = woe_encoder.transform(self.X_eval)

        # LR 训练
        if regularization == "l1":
            solver = "saga"
        elif regularization == "elasticnet":
            solver = "saga"
        else:
            solver = "lbfgs"

        lr_params = {
            "penalty": regularization,
            "C": C,
            "solver": solver,
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        }
        if regularization == "elasticnet":
            lr_params["l1_ratio"] = 0.5

        try:
            model = LogisticRegression(**lr_params)
            model.fit(X_train_woe, self.y_train)
        except Exception:
            return 0.0

        # 评估
        prob_train = model.predict_proba(X_train_woe)[:, 1]
        prob_eval = model.predict_proba(X_eval_woe)[:, 1]

        if self.metric == "ks":
            train_ks = float(ModelEvaluator.calculate_ks(self.y_train, prob_train))
            eval_ks = float(ModelEvaluator.calculate_ks(self.y_eval, prob_eval))
            # Gap 约束
            gap = train_ks - eval_ks
            threshold = _DIAG_THRESHOLDS["ks"]["overfit_gap"]
            if gap > threshold:
                return 0.0
            return max(eval_ks, 0.0)
        else:
            train_auc = float(roc_auc_score(self.y_train, prob_train))
            eval_auc = float(roc_auc_score(self.y_eval, prob_eval))
            # Gap 约束
            gap = train_auc - eval_auc
            threshold = _DIAG_THRESHOLDS["auc"]["overfit_gap"]
            if gap > threshold:
                return 0.0
            return max(eval_auc, 0.0)

    def _get_default_space(self) -> Dict[str, Tuple[float, float]]:
        """LR 全量硬边界搜索空间。"""
        return dict(LR_PARAM_BOUNDS)

    def _build_constrained_space(
        self,
        diagnosis: DiagnosisResult,
        tried_directions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """LR 诊断驱动的约束搜索空间。

        过拟合 → C 上限收紧、iv_threshold 抬高（减少入模特征）
        欠拟合 → C 放松、iv_threshold 降低、max_n_bins 增大
        拟合良好 → 全参数 ±20% 微调
        """
        tried_directions = tried_directions or []
        space: Dict[str, Tuple[float, float]] = {}

        c_val = self.base_params.get("C", 1.0)
        iv_val = self.base_params.get("iv_threshold", 0.02)
        bins_val = self.base_params.get("max_n_bins", 8)

        status = diagnosis.status

        if status == "过拟合":
            # 增强正则化：C 降低，IV 阈值抬高（减少特征）
            space["C"] = (_clip_lr(c_val * 0.1, "C"), _clip_lr(c_val * 0.8, "C"))
            space["iv_threshold"] = (_clip_lr(iv_val, "iv_threshold"), _clip_lr(iv_val * 3, "iv_threshold"))
            space["max_n_bins"] = (_clip_lr(max(3, bins_val - 3), "max_n_bins"), _clip_lr(bins_val, "max_n_bins"))
        elif status == "轻微过拟合":
            space["C"] = (_clip_lr(c_val * 0.3, "C"), _clip_lr(c_val * 1.0, "C"))
            space["iv_threshold"] = (_clip_lr(iv_val, "iv_threshold"), _clip_lr(iv_val * 2, "iv_threshold"))
            space["max_n_bins"] = (_clip_lr(max(3, bins_val - 2), "max_n_bins"), _clip_lr(bins_val + 1, "max_n_bins"))
        elif status == "欠拟合":
            # 放松正则化：C 增大，IV 阈值降低（更多特征入模）
            space["C"] = (_clip_lr(c_val, "C"), _clip_lr(c_val * 10, "C"))
            space["iv_threshold"] = (_clip_lr(iv_val * 0.3, "iv_threshold"), _clip_lr(iv_val, "iv_threshold"))
            space["max_n_bins"] = (_clip_lr(bins_val, "max_n_bins"), _clip_lr(bins_val + 5, "max_n_bins"))
        else:
            # 拟合良好 / 收敛 → 微调
            space["C"] = (_clip_lr(c_val * 0.5, "C"), _clip_lr(c_val * 2, "C"))
            space["iv_threshold"] = (_clip_lr(iv_val * 0.7, "iv_threshold"), _clip_lr(iv_val * 1.5, "iv_threshold"))
            space["max_n_bins"] = (_clip_lr(max(3, bins_val - 2), "max_n_bins"), _clip_lr(bins_val + 2, "max_n_bins"))

        # 根据 tried_directions 冻结已尝试无效的方向
        for record in tried_directions:
            if record.get("improved"):
                continue
            delta = record.get("params_delta", {}) or {}
            for key, d in delta.items():
                if key not in space:
                    continue
                cur = self.base_params.get(key)
                if cur is None:
                    continue
                lo, hi = space[key]
                if d > 0:
                    space[key] = (lo, min(hi, float(cur)))
                elif d < 0:
                    space[key] = (max(lo, float(cur)), hi)

        return space

__all__ = ["LRTuningEngine", "LR_PARAM_BOUNDS"]
