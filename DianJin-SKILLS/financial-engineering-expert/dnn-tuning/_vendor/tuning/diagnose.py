# -*- coding: utf-8 -*-
"""模型状态诊断 — 单一事实源。

提供 DiagnosisResult 数据结构和 diagnose_model 函数，
根据训练/验证指标判定模型状态（过拟合 / 轻微过拟合 / 拟合良好 / 欠拟合 / 收敛）。

KS/AUC 双主指标：metric 参数决定阈值体系，KS 阈值更宽（方差大、依赖单点），
AUC 沿用项目既有约定。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# 主指标 → 阈值组。KS 方差更大、依赖单点，Gap 阈值比 AUC 宽；
# AUC 平滑稳定，沿用项目既有约定。
_DIAG_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ks": {
        "overfit_gap": 0.08,
        "slight_overfit_gap": 0.05,
        "underfit_eval": 0.15,
        "underfit_gap": 0.02,
        "converge_delta": 0.002,
    },
    "auc": {
        "overfit_gap": 0.05,
        "slight_overfit_gap": 0.04,
        "underfit_eval": 0.55,
        "underfit_gap": 0.02,
        "converge_delta": 0.001,
    },
}

@dataclass
class DiagnosisResult:
    """模型状态诊断结果（双 Gap 版）。

    注意：eval_* 字段对应的是诊断用的评估集指标，**应为 val**（经典三段式
    下 OOT 仅用于最终汇报，严禁参与诊断/调参）。

    字段设计：
        - primary_metric: 诊断所依据的主指标（"ks" 或 "auc"），决定阈值体系
        - gap: 主指标 Gap（= 主指标 train - 主指标 eval），保留用于向后兼容
        - gap_ks / gap_auc: 两种 Gap 同时暴露，便于报告层双展示、结构诊断使用
    """
    status: str              # 过拟合 / 轻微过拟合 / 拟合良好 / 欠拟合 / 收敛
    primary_metric: str      # "ks" 或 "auc"
    gap: float               # 主指标 Gap（向后兼容字段）
    gap_ks: float            # train_ks - eval_ks
    gap_auc: float           # train_auc - eval_auc
    train_auc: float
    eval_auc: float
    train_ks: float
    eval_ks: float
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def diagnose_model(
    train_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
    prev_eval_metric: Optional[float] = None,
    metric: str = "ks",
) -> DiagnosisResult:
    """根据训练集 / 评估集指标做状态诊断（KS/AUC 双主指标）。

    Args:
        train_metrics: 训练集指标 dict（需含 auc 和 ks）
        eval_metrics:  评估集指标 dict，**应为 val**；OOT 仅用于最终汇报，禁止参与诊断
        prev_eval_metric: 上一轮 eval **主指标**，用于收敛判定（注意：要与 metric 参数对齐）
        metric: "ks" 或 "auc"，决定诊断阈值体系（默认 KS，与项目评估约定一致）

    KS 阈值（默认）：
        gap_ks > 0.08        → 过拟合
        gap_ks ∈ (0.05,0.08] → 轻微过拟合
        eval_ks < 0.15 且 gap_ks < 0.02 → 欠拟合
        其余                  → 拟合良好
        prev 差距 < 0.002     → 收敛

    AUC 阈值：
        gap_auc > 0.05           → 过拟合
        gap_auc ∈ (0.04, 0.05]   → 轻微过拟合
        eval_auc < 0.55 且 gap < 0.02 → 欠拟合
        其余                      → 拟合良好
        prev 差距 < 0.001         → 收敛
    """
    metric_l = (metric or "ks").lower()
    if metric_l not in _DIAG_THRESHOLDS:
        raise ValueError(f"metric must be 'ks' or 'auc', got: {metric!r}")

    train_auc = float(train_metrics["auc"])
    eval_auc = float(eval_metrics["auc"])
    train_ks = float(train_metrics.get("ks", 0.0))
    eval_ks = float(eval_metrics.get("ks", 0.0))
    gap_auc = train_auc - eval_auc
    gap_ks = train_ks - eval_ks

    primary_gap = gap_ks if metric_l == "ks" else gap_auc
    primary_eval = eval_ks if metric_l == "ks" else eval_auc
    th = _DIAG_THRESHOLDS[metric_l]
    metric_upper = metric_l.upper()

    def _result(status: str, suggestion: str) -> DiagnosisResult:
        return DiagnosisResult(
            status=status,
            primary_metric=metric_l,
            gap=primary_gap,
            gap_ks=gap_ks,
            gap_auc=gap_auc,
            train_auc=train_auc,
            eval_auc=eval_auc,
            train_ks=train_ks,
            eval_ks=eval_ks,
            suggestion=suggestion,
        )

    # 收敛判定（以主指标的 eval 为依据，与 prev_eval_metric 语义必须对齐）
    if prev_eval_metric is not None and abs(primary_eval - prev_eval_metric) < th["converge_delta"]:
        return _result(
            "收敛",
            f"连续轮次 {metric_upper} 提升不足 {th['converge_delta']:.3f}，建议停止调优",
        )

    if primary_gap > th["overfit_gap"]:
        return _result(
            "过拟合",
            f"{metric_upper} Gap > {th['overfit_gap']:.2f}，"
            "将在下一轮 Optuna 搜索中收紧 max_depth 上限并抬高正则化下限",
        )
    if primary_gap > th["slight_overfit_gap"]:
        return _result(
            "轻微过拟合",
            f"{metric_upper} Gap 处于临界值，下一轮在正则化区间内精细搜索",
        )
    if primary_eval < th["underfit_eval"] and primary_gap < th["underfit_gap"]:
        return _result(
            "欠拟合",
            f"评估 {metric_upper} 偏低且 Gap 小，下一轮在 max_depth/n_estimators 上限抬高区间搜索",
        )
    return _result(
        "拟合良好",
        "模型状态良好，下一轮在当前参数 ±20% 窗口内微调",
    )

__all__ = [
    "DiagnosisResult",
    "diagnose_model",
    "_DIAG_THRESHOLDS",
]
