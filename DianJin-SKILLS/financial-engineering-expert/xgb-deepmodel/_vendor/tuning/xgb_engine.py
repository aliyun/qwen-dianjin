# -*- coding: utf-8 -*-
"""XGBoost 调参引擎 — XGBTuningEngine。

继承 BaseTuningEngine，实现 XGBoost 专用的评估逻辑和搜索空间。
包含 KS 三重加固（稳健KS、Gap约束、切点稳定性惩罚）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xgb_core import XGBTrainer, ModelEvaluator

from tuning.engine import BaseTuningEngine
from tuning.diagnose import DiagnosisResult, _DIAG_THRESHOLDS
from tuning.search_space import (
    _build_constrained_space,
    _default_batch_space,
)

logger = logging.getLogger(__name__)

class XGBTuningEngine(BaseTuningEngine):
    """基于 Optuna TPE 的 XGBoost 调参引擎。

    继承 BaseTuningEngine，实现 XGB 专用的：
      - _evaluate: KS 三重加固评估
      - _get_default_space: XGB 全量硬边界空间
      - _build_constrained_space: 诊断驱动的 XGB 约束空间
    """

    INT_PARAMS = {"max_depth", "n_estimators", "min_child_weight"}
    LOG_PARAMS = {"learning_rate", "reg_alpha", "reg_lambda", "scale_pos_weight"}

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """用给定参数训练 XGBoost 并返回稳健评估指标。

        KS 模式下实现三项加固：
          1. 稳健 KS = ks_point - 0.5 * ks_std（Bootstrap 消振荡）
          2. Gap 约束：gap_ks > threshold → 返回 0（Optuna 视为劣解）
          3. 切点稳定性惩罚：train/eval 峰值切点分位差 > 5% → score -= 0.01
        """
        from sklearn.metrics import roc_auc_score

        trainer = XGBTrainer(params)
        trainer.train(
            self.X_train,
            self.y_train,
            self.X_eval,
            self.y_eval,
            sample_weight=self.sample_weight,
        )

        if self.metric != "ks":
            y_prob = trainer.predict_proba(self.X_eval)
            return float(roc_auc_score(self.y_eval, y_prob))

        # --- KS 稳健评估 ---
        y_prob_train = trainer.predict_proba(self.X_train)
        y_prob_eval = trainer.predict_proba(self.X_eval)

        train_ks = float(ModelEvaluator.calculate_ks(self.y_train, y_prob_train))
        eval_ks = float(ModelEvaluator.calculate_ks(self.y_eval, y_prob_eval))

        # (1) Gap 约束 prune
        gap_ks = train_ks - eval_ks
        threshold = _DIAG_THRESHOLDS["ks"]["overfit_gap"]
        if gap_ks > threshold:
            return 0.0

        # (2) 稳健 KS = ks_point - 0.5 * ks_std (Bootstrap)
        score = self._robust_ks(self.y_eval, y_prob_eval)

        # (3) 切点稳定性惩罚
        train_peak_q = self._peak_quantile(self.y_train, y_prob_train)
        eval_peak_q = self._peak_quantile(self.y_eval, y_prob_eval)
        if abs(train_peak_q - eval_peak_q) > 0.05:
            score -= 0.01

        return max(score, 0.0)

    def _get_default_space(self) -> Dict[str, Tuple[float, float]]:
        """XGB 全量硬边界搜索空间。"""
        return _default_batch_space()

    def _build_constrained_space(
        self,
        diagnosis: DiagnosisResult,
        tried_directions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """XGB 诊断驱动的约束搜索空间（委托给 search_space 模块）。"""
        return _build_constrained_space(diagnosis, self.base_params, tried_directions)

    # ------------------------------------------------------------------
    # XGB 专用辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _robust_ks(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 50) -> float:
        """Bootstrap 稳健 KS = point - 0.5 * std。"""
        n = len(y_true)
        rng = np.random.RandomState(42)
        ks_samples = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.randint(0, n, size=n)
            ks_samples[i] = ModelEvaluator.calculate_ks(y_true[idx], y_prob[idx])
        return float(np.mean(ks_samples) - 0.5 * np.std(ks_samples))

    @staticmethod
    def _peak_quantile(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """KS 峰值切点对应的分位数 (Top X%)。"""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        order = np.argsort(-y_prob)
        y_sorted = y_true[order]
        total_pos = y_sorted.sum()
        total_neg = len(y_sorted) - total_pos
        if total_pos == 0 or total_neg == 0:
            return 0.5
        cum_pos = np.cumsum(y_sorted) / total_pos
        cum_neg = np.cumsum(1 - y_sorted) / total_neg
        ks_curve = np.abs(cum_pos - cum_neg)
        peak_idx = int(np.argmax(ks_curve))
        return (peak_idx + 1) / len(y_true)

__all__ = ["XGBTuningEngine"]
