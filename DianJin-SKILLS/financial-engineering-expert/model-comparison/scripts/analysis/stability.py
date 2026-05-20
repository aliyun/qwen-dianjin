# -*- coding: utf-8 -*-
"""稳定性分析 — 真实分数 PSI + 分月表现。

复用 xgb_core.ModelStabilityAnalyzer.score_psi 实现，避免轮子重复。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from xgb_core import ModelStabilityAnalyzer

def compute_psi(
    baseline_scores: np.ndarray,
    comparison_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """OOT vs Train 的分数 PSI。"""
    if baseline_scores is None or comparison_scores is None:
        return 0.0
    if len(baseline_scores) == 0 or len(comparison_scores) == 0:
        return 0.0
    analyzer = ModelStabilityAnalyzer(n_bins=n_bins)
    return round(analyzer.score_psi(baseline_scores, comparison_scores), 6)

def interpret_psi(psi: float) -> str:
    """PSI 档位文案 — 对齐业界约定（<0.1 稳定 / <0.25 轻微 / 其余 显著）。"""
    return ModelStabilityAnalyzer.interpret_psi(psi)

def compute_monthly_performance(
    oot_data: pd.DataFrame,
    target: str,
    scores: np.ndarray,
    time_col: str = "busi_dt",
) -> Optional[List[Dict[str, Any]]]:
    """返回每月 AUC/KS 汇总列表；若 OOT 不含时间列或只一个月，返回 None。"""
    if time_col not in oot_data.columns:
        return None
    df = oot_data[[time_col, target]].copy()
    df["_score"] = scores
    analyzer = ModelStabilityAnalyzer()
    perf = analyzer.performance_by_month(df, time_col=time_col, target=target, score_col="_score")
    if perf.empty or len(perf) <= 1:
        return None
    return perf.to_dict(orient="records")
