# -*- coding: utf-8 -*-
"""DiagnosticReport — 多维诊断聚合器。

将 FitDiagnosis + KSStructureDiagnosis 聚合为统一的 DiagnosticReport，
供下游 tuning/explanation/orchestrator 消费。

当前实现的维度：
  - fit: 拟合度诊断
  - ks_structure: KS 结构诊断

预留维度（后续 Task 扩展）：
  - capacity: 学习曲线饱和诊断
  - stability: 分群 / 时间维度稳定性
  - feature: SHAP 方向一致性 + 头部集中度
  - prediction: PSI + 校准误差
  - leakage: 泄漏风险
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from diagnostic.fit_check import FitDiagnosis, run_fit_check
from diagnostic.ks_structure import KSStructureDiagnosis, run_ks_structure_check

@dataclass(frozen=True)
class DiagnosticReport:
    """统一诊断报告。

    Attributes:
        fit: 拟合度诊断
        ks_structure: KS 结构诊断（可选，需传入原始 y_true/y_prob）
        is_overfit: 综合过拟合判定（fit.is_overfit OR ks_structure.is_hidden_overfit）
        severity: 综合严重程度
        all_warnings: 合并所有维度的 warnings
    """
    fit: FitDiagnosis
    ks_structure: Optional[KSStructureDiagnosis] = None

    # --- 预留维度占位（后续 Task 扩展时替换为具体类型）---
    # capacity: Optional[Any] = None
    # stability: Optional[Any] = None
    # feature: Optional[Any] = None
    # prediction: Optional[Any] = None
    # leakage: Optional[Any] = None

    @property
    def is_overfit(self) -> bool:
        """综合过拟合判定：任一维度触发即为 True。"""
        if self.fit.is_overfit:
            return True
        if self.ks_structure is not None and self.ks_structure.is_hidden_overfit:
            return True
        return False

    @property
    def severity(self) -> str:
        """综合严重程度。"""
        if self.fit.severity == "severe":
            return "severe"
        if self.ks_structure is not None and self.ks_structure.is_hidden_overfit:
            # 隐性过拟合至少 mild
            return max(self.fit.severity, "mild", key=lambda s: {"none": 0, "mild": 1, "severe": 2}[s])
        return self.fit.severity

    @property
    def all_warnings(self) -> List[str]:
        """合并所有维度的 warnings。"""
        warnings = list(self.fit.advice)
        if self.ks_structure is not None:
            warnings.extend(self.ks_structure.warnings)
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "is_overfit": self.is_overfit,
            "severity": self.severity,
            "all_warnings": self.all_warnings,
            "fit": self.fit.to_dict(),
        }
        if self.ks_structure is not None:
            d["ks_structure"] = self.ks_structure.to_dict()
        return d

def run_diagnostics(
    train_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
    *,
    prev_eval_metric: Optional[float] = None,
    metric: str = "ks",
    y_true_train: Optional[np.ndarray] = None,
    y_prob_train: Optional[np.ndarray] = None,
    y_true_val: Optional[np.ndarray] = None,
    y_prob_val: Optional[np.ndarray] = None,
    y_true_oot: Optional[np.ndarray] = None,
    y_prob_oot: Optional[np.ndarray] = None,
) -> DiagnosticReport:
    """一站式诊断入口。

    Args:
        train_metrics / eval_metrics: 训练/验证指标 dict（必须含 auc 和 ks）
        prev_eval_metric: 上轮主指标值（收敛判定用）
        metric: "ks" 或 "auc"
        y_true_* / y_prob_*: 原始标签和预测概率（可选，传入时启用 KS 结构诊断）

    Returns:
        DiagnosticReport（至少含 fit 维度，可选含 ks_structure）
    """
    # 1. 拟合度诊断（必做）
    fit = run_fit_check(train_metrics, eval_metrics, prev_eval_metric, metric=metric)

    # 2. KS 结构诊断（需要原始数据）
    ks_struct: Optional[KSStructureDiagnosis] = None
    if (
        y_true_train is not None
        and y_prob_train is not None
        and y_true_val is not None
        and y_prob_val is not None
    ):
        ks_struct = run_ks_structure_check(
            y_true_train, y_prob_train,
            y_true_val, y_prob_val,
            y_true_oot, y_prob_oot,
            ks_train=train_metrics.get("ks"),
            ks_val=eval_metrics.get("ks"),
            auc_train=train_metrics.get("auc"),
            auc_val=eval_metrics.get("auc"),
        )

    return DiagnosticReport(fit=fit, ks_structure=ks_struct)

__all__ = ["DiagnosticReport", "run_diagnostics"]
