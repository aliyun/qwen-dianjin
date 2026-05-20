# -*- coding: utf-8 -*-
"""拟合度诊断 — 扩展自 tuning.diagnose.DiagnosisResult。

FitDiagnosis 在原有 Gap 判定基础上增加了结构性字段：
  - is_overfit: 布尔标记，合并 Gap 判定与结构诊断（KS 漂移等）的综合结论
  - severity: 严重程度（none / mild / severe）
  - advice: 面向下游调参引擎的结构化建议
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tuning.diagnose import DiagnosisResult, diagnose_model, _DIAG_THRESHOLDS

@dataclass(frozen=True)
class FitDiagnosis:
    """拟合度诊断结果。

    Attributes:
        status: 原始诊断状态（过拟合 / 轻微过拟合 / 拟合良好 / 欠拟合 / 收敛）
        primary_metric: 诊断所依据的主指标（"ks" 或 "auc"）
        gap: 主指标 Gap（向后兼容）
        gap_ks: train_ks - eval_ks
        gap_auc: train_auc - eval_auc
        is_overfit: 综合过拟合标记（Gap 超阈值 → True，后续可被 KS 结构诊断追加触发）
        severity: "none" | "mild" | "severe"
        advice: 结构化建议列表
        raw: 原始 DiagnosisResult（完整字段保留）
    """
    status: str
    primary_metric: str
    gap: float
    gap_ks: float
    gap_auc: float
    is_overfit: bool
    severity: str                  # "none" | "mild" | "severe"
    advice: List[str] = field(default_factory=list)
    raw: Optional[DiagnosisResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "status": self.status,
            "primary_metric": self.primary_metric,
            "gap": self.gap,
            "gap_ks": self.gap_ks,
            "gap_auc": self.gap_auc,
            "is_overfit": self.is_overfit,
            "severity": self.severity,
            "advice": list(self.advice),
        }
        if self.raw is not None:
            d["raw"] = self.raw.to_dict()
        return d

def run_fit_check(
    train_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
    prev_eval_metric: Optional[float] = None,
    metric: str = "ks",
) -> FitDiagnosis:
    """执行拟合度诊断，返回 FitDiagnosis。

    内部复用 tuning.diagnose.diagnose_model，在其基础上增加结构化字段。
    """
    diag = diagnose_model(train_metrics, eval_metrics, prev_eval_metric, metric=metric)

    # 从状态推导 severity + is_overfit
    if diag.status == "过拟合":
        is_overfit = True
        severity = "severe"
    elif diag.status == "轻微过拟合":
        is_overfit = True
        severity = "mild"
    else:
        is_overfit = False
        severity = "none"

    # 构造结构化建议
    advice: List[str] = []
    th = _DIAG_THRESHOLDS[diag.primary_metric]

    if is_overfit:
        advice.append(f"收紧正则化（当前 {diag.primary_metric.upper()} Gap = {diag.gap:.4f}）")
        if diag.gap_ks > th["overfit_gap"] and diag.gap_auc <= _DIAG_THRESHOLDS["auc"]["slight_overfit_gap"]:
            advice.append("KS Gap 大但 AUC Gap 正常 → 可能是 KS 切点不稳定，建议检查 KS 结构")

    if diag.status == "欠拟合":
        advice.append("抬高模型容量（max_depth / n_estimators）")
        advice.append("检查特征工程是否充分")

    if diag.status == "收敛":
        advice.append("已收敛，建议停止调参或尝试特征变更")

    return FitDiagnosis(
        status=diag.status,
        primary_metric=diag.primary_metric,
        gap=diag.gap,
        gap_ks=diag.gap_ks,
        gap_auc=diag.gap_auc,
        is_overfit=is_overfit,
        severity=severity,
        advice=advice,
        raw=diag,
    )

__all__ = ["FitDiagnosis", "run_fit_check"]
