# -*- coding: utf-8 -*-
"""Bootstrap 置信区间工具。

提供 KS / AUC 的 Bootstrap CI 及配对差值 CI，供 MetricsBundle、
SignificanceGate（Task 3）、Champion/Challenger（Task 4）使用。
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

# xgb_core 由 python_executor 预置 PYTHONPATH=backend/libs
from xgb_core import ModelEvaluator
from xgb_core.stability import ModelStabilityAnalyzer

def bootstrap_ks_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """KS 的 Bootstrap 百分位置信区间。

    Args:
        y_true: 真实标签 (0/1)
        y_prob: 预测正类概率
        n_bootstrap: 重抽样次数（默认 200，速度敏感场景；高精度传 500+）
        alpha: 显著性水平（双侧）
        random_state: 随机种子

    Returns:
        (lower, upper) — 若有效 Bootstrap 样本不足 10 次返回 (nan, nan)
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    ks_samples = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            ks_samples.append(float(ModelEvaluator.calculate_ks(yt, yp)))
        except Exception:
            continue

    if len(ks_samples) < 10:
        return float("nan"), float("nan")

    lower = float(np.percentile(ks_samples, 100 * alpha / 2))
    upper = float(np.percentile(ks_samples, 100 * (1 - alpha / 2)))
    return round(lower, 6), round(upper, 6)

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """AUC 的 Bootstrap CI — 薄封装 xgb_core.stability.bootstrap_auc_ci。

    统一默认参数（n_bootstrap=200, alpha→confidence 映射）以与 bootstrap_ks_ci 对齐。
    """
    confidence = 1 - alpha
    # xgb_core 的 bootstrap_auc_ci 是 @staticmethod，内部固定 seed=42
    return ModelStabilityAnalyzer.bootstrap_auc_ci(
        np.asarray(y_true), np.asarray(y_prob),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )

def paired_bootstrap_delta_ci(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    *,
    metric: str = "ks",
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """配对 Bootstrap 差值 CI：δ = metric(B) - metric(A)。

    用于 Champion / Challenger 对比（Task 4）。当 CI 完全 > 0 时
    可认为 B 显著优于 A。

    Returns:
        (lower, upper) — δ 的 (alpha/2, 1-alpha/2) 百分位 CI
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    n = len(y_true)

    def _calc(yt, yp):
        if metric == "ks":
            return float(ModelEvaluator.calculate_ks(yt, yp))
        return float(roc_auc_score(yt, yp))

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            deltas.append(_calc(yt, y_prob_b[idx]) - _calc(yt, y_prob_a[idx]))
        except Exception:
            continue

    if len(deltas) < 10:
        return float("nan"), float("nan")

    lower = float(np.percentile(deltas, 100 * alpha / 2))
    upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return round(lower, 6), round(upper, 6)
