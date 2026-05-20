# -*- coding: utf-8 -*-
"""MetricsBundle — KS 为主的评估结构，带可选 Bootstrap CI。

所有 skill 产出的评估结果统一通过 MetricsBundle 表达。
- 点估计来自 xgb_core.ModelEvaluator.evaluate（纯原语）
- CI 来自 shared.evaluation.bootstrap（本包）
- ks_peak_quantile 预留给 Task 2 DiagnosticSuite 填充

典型用法：
    from evaluation.metrics import evaluate_with_ci
    bundle = evaluate_with_ci(y_true, y_prob)
    print(bundle.ks, bundle.primary_ci)   # → 0.42 (0.38, 0.46)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

# xgb_core 由 python_executor 预置 PYTHONPATH=backend/libs
from xgb_core import ModelEvaluator

from .bootstrap import bootstrap_ks_ci, bootstrap_auc_ci

@dataclass(frozen=True)
class MetricsBundle:
    """KS 为主的评估结果。所有 CI 字段在未启用 bootstrap 时为 None。

    字段设计：
        - primary_metric: "ks" 或 "auc"，标识报告 / 诊断以哪个指标为主
        - ks / auc / accuracy: 点估计（来自 ModelEvaluator.evaluate）
        - ks_peak_quantile: KS 切点所处分位（Task 2 补充，Task 1 允许 None）
        - ks_ci_* / auc_ci_*: Bootstrap 置信区间
        - n_samples / n_positives: 元信息，供调用方判断 CI 可信度
    """
    # 点估计
    auc: float
    ks: float
    accuracy: float
    # KS 结构信息（Task 2 DiagnosticSuite 会填充；Task 1 允许为 None）
    ks_peak_quantile: Optional[float] = None
    # 置信区间
    ks_ci_lower: Optional[float] = None
    ks_ci_upper: Optional[float] = None
    auc_ci_lower: Optional[float] = None
    auc_ci_upper: Optional[float] = None
    # 元信息
    n_samples: int = 0
    n_positives: int = 0
    primary_metric: str = "ks"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def primary_value(self) -> float:
        """返回主指标的点估计。"""
        return self.ks if self.primary_metric == "ks" else self.auc

    @property
    def primary_ci(self) -> Optional[Tuple[float, float]]:
        """返回主指标的 CI；未计算时返回 None。"""
        if self.primary_metric == "ks":
            if self.ks_ci_lower is not None and self.ks_ci_upper is not None:
                return (self.ks_ci_lower, self.ks_ci_upper)
        else:
            if self.auc_ci_lower is not None and self.auc_ci_upper is not None:
                return (self.auc_ci_lower, self.auc_ci_upper)
        return None

    def format_primary(self) -> str:
        """人类可读的主指标摘要，例如 'KS=0.4200 [0.3800, 0.4600]'。"""
        metric_upper = self.primary_metric.upper()
        val = self.primary_value
        ci = self.primary_ci
        if ci:
            return f"{metric_upper}={val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        return f"{metric_upper}={val:.4f}"

def evaluate_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bootstrap: bool = True,
    n_bootstrap: int = 200,
    primary_metric: str = "ks",
    random_state: int = 42,
) -> MetricsBundle:
    """统一评估入口：点估计 + 可选 Bootstrap CI。

    Args:
        y_true: 真实标签 (0/1)
        y_prob: 预测正类概率
        bootstrap: 是否计算 CI（默认 True；调参内循环可关闭以节省时间）
        n_bootstrap: 重抽样次数（200 = 快但略宽；500 = 精确但耗时）
        primary_metric: "ks" 或 "auc"
        random_state: Bootstrap 随机种子

    Returns:
        MetricsBundle（frozen dataclass）
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # 点估计 — 来自 xgb_core 原语
    raw = ModelEvaluator.evaluate(y_true, y_prob)

    # 精度 (accuracy)
    pred_label = (y_prob >= 0.5).astype(int)
    accuracy = float(np.mean(pred_label == y_true))

    # Bootstrap CI
    ks_ci_lower = ks_ci_upper = None
    auc_ci_lower = auc_ci_upper = None
    if bootstrap:
        ks_ci_lower, ks_ci_upper = bootstrap_ks_ci(
            y_true, y_prob, n_bootstrap=n_bootstrap,
            alpha=0.05, random_state=random_state,
        )
        auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(
            y_true, y_prob, n_bootstrap=n_bootstrap,
            alpha=0.05, random_state=random_state,
        )

    n_samples = len(y_true)
    n_positives = int(np.sum(y_true == 1))

    return MetricsBundle(
        auc=raw["auc"],
        ks=raw["ks"],
        accuracy=round(accuracy, 6),
        ks_peak_quantile=None,  # Task 2 填充
        ks_ci_lower=ks_ci_lower,
        ks_ci_upper=ks_ci_upper,
        auc_ci_lower=auc_ci_lower,
        auc_ci_upper=auc_ci_upper,
        n_samples=n_samples,
        n_positives=n_positives,
        primary_metric=primary_metric,
    )
