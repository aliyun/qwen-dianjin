# -*- coding: utf-8 -*-
"""
模型稳定性分析模块

"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class ModelStabilityAnalyzer:
    """
    模型稳定性分析器

    分析预测分数在不同时间窗口的 PSI 漂移情况，
    以及 AUC/KS 等表现指标随时间的变化趋势
    """

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6) -> None:
        self.n_bins = n_bins
        self.epsilon = epsilon

    def score_psi(
        self,
        baseline_scores: np.ndarray | pd.Series,
        comparison_scores: np.ndarray | pd.Series,
    ) -> float:
        """
        计算两组预测分数之间的 PSI

        Args:
            baseline_scores: 基准期分数
            comparison_scores: 对比期分数

        Returns:
            PSI 值
        """
        baseline_scores = np.array(baseline_scores)
        comparison_scores = np.array(comparison_scores)

        # 基于基准集的分位数分箱
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.percentile(baseline_scores, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins = np.unique(bins)

        base_hist = np.histogram(baseline_scores, bins=bins)[0]
        comp_hist = np.histogram(comparison_scores, bins=bins)[0]

        base_prop = base_hist / base_hist.sum() + self.epsilon
        comp_prop = comp_hist / comp_hist.sum() + self.epsilon

        psi = float(np.sum((comp_prop - base_prop) * np.log(comp_prop / base_prop)))
        return psi

    @staticmethod
    def bootstrap_auc_ci(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bootstrap: int = 200,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        用 Bootstrap 估计 AUC 的置信区间。

        Returns:
            (lower_bound, upper_bound)
        """
        rng = np.random.RandomState(42)
        n = len(y_true)
        auc_samples = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            yt = y_true[idx]
            yp = y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                auc_samples.append(roc_auc_score(yt, yp))
            except Exception:
                continue

        if len(auc_samples) < 10:
            return float('nan'), float('nan')

        alpha = 1 - confidence
        lower = float(np.percentile(auc_samples, 100 * alpha / 2))
        upper = float(np.percentile(auc_samples, 100 * (1 - alpha / 2)))
        return round(lower, 6), round(upper, 6)

    @staticmethod
    def mann_kendall_trend(values: list[float]) -> dict:
        """
        Mann-Kendall 趋势检验。

        比简单的"早期均值 vs 近期均值"更可靠，
        能判断是否存在统计显著的单调上升/下降趋势。

        Returns:
            {
                "trend": "increasing" | "decreasing" | "no trend",
                "tau": float,         # Kendall tau 相关系数
                "p_value": float,
            }
        """
        try:
            from scipy.stats import kendalltau
        except ImportError:
            return {"trend": "unknown", "tau": 0.0, "p_value": 1.0}

        n = len(values)
        if n < 4:
            return {"trend": "insufficient data", "tau": 0.0, "p_value": 1.0}

        tau, p_value = kendalltau(range(n), values)
        tau = float(tau)
        p_value = float(p_value)

        if p_value < 0.05:
            trend = "increasing" if tau > 0 else "decreasing"
        else:
            trend = "no trend"

        return {"trend": trend, "tau": round(tau, 4), "p_value": round(p_value, 4)}

    def performance_by_month(
        self,
        data: pd.DataFrame,
        time_col: str,
        target: str,
        score_col: str,
    ) -> pd.DataFrame:
        """
        按月计算 AUC / KS

        Args:
            data: 包含 time_col, target, score_col 的 DataFrame
            time_col: 时间列名（格式如 '20240901'）
            target: 目标变量列名
            score_col: 预测分数列名

        Returns:
            每月 AUC/KS 汇总表
        """
        from .evaluator import ModelEvaluator

        df = data[[time_col, target, score_col]].copy()
        df['month'] = df[time_col].astype(str).str[:6]

        results = []
        for month in sorted(df['month'].unique()):
            mdata = df[df['month'] == month]
            y_true = mdata[target].values
            y_prob = mdata[score_col].values

            if len(np.unique(y_true)) < 2:
                continue

            auc = roc_auc_score(y_true, y_prob)
            ks = ModelEvaluator.calculate_ks(y_true, y_prob)

            # Bootstrap CI（样本量足够时计算，否则跳过避免耗时）
            auc_ci_lower, auc_ci_upper = float('nan'), float('nan')
            if len(y_true) >= 100:
                auc_ci_lower, auc_ci_upper = ModelStabilityAnalyzer.bootstrap_auc_ci(
                    y_true, y_prob, n_bootstrap=200
                )

            results.append({
                'month': month,
                'samples': len(mdata),
                'positive_count': int(y_true.sum()),
                'positive_rate': float(y_true.mean()),
                'auc': round(auc, 6),
                'auc_ci_lower': auc_ci_lower,
                'auc_ci_upper': auc_ci_upper,
                'ks': round(ks, 6),
            })

        return pd.DataFrame(results)

    def score_psi_by_month(
        self,
        data: pd.DataFrame,
        time_col: str,
        score_col: str,
        baseline_months: list[str],
    ) -> pd.DataFrame:
        """
        以指定月份为基准，计算每月预测分数的 PSI

        Args:
            data: DataFrame
            time_col: 时间列名
            score_col: 预测分数列名
            baseline_months: 基准月份列表，如 ['202409', '202410']

        Returns:
            每月分数 PSI 汇总表
        """
        df = data[[time_col, score_col]].copy()
        df['month'] = df[time_col].astype(str).str[:6]

        baseline_scores = df[df['month'].isin(baseline_months)][score_col].values

        if len(baseline_scores) == 0:
            raise ValueError(f"基准月份 {baseline_months} 无有效数据")

        results = []
        for month in sorted(df['month'].unique()):
            mdata = df[df['month'] == month]
            month_scores = mdata[score_col].values

            psi = self.score_psi(baseline_scores, month_scores)

            results.append({
                'month': month,
                'samples': len(mdata),
                'mean_score': round(float(month_scores.mean()), 6),
                'std_score': round(float(month_scores.std()), 6),
                'psi': round(psi, 6),
            })

        return pd.DataFrame(results)

    @staticmethod
    def interpret_psi(psi: float) -> str:
        if psi < 0.1:
            return "稳定"
        elif psi < 0.25:
            return "轻微漂移"
        else:
            return "显著漂移"

    def full_stability_report(
        self,
        data: pd.DataFrame,
        time_col: str,
        target: str,
        score_col: str,
        baseline_months: list[str],
    ) -> dict[str, Any]:
        """
        生成完整稳定性分析结果

        Returns:
            {
                'performance_by_month': DataFrame,
                'score_psi_by_month': DataFrame,
                'overall_score_psi': float,
                'auc_std': float,
                'ks_std': float,
            }
        """
        df = data[[time_col, target, score_col]].copy()
        df['month'] = df[time_col].astype(str).str[:6]

        perf = self.performance_by_month(data, time_col, target, score_col)
        psi_monthly = self.score_psi_by_month(data, time_col, score_col, baseline_months)

        # 基准 vs 非基准的整体 PSI
        baseline_scores = df[df['month'].isin(baseline_months)][score_col].values
        non_baseline_scores = df[~df['month'].isin(baseline_months)][score_col].values
        overall_psi = self.score_psi(baseline_scores, non_baseline_scores) if len(non_baseline_scores) > 0 else 0.0

        # Mann-Kendall 趋势检验
        auc_trend = {"trend": "insufficient data", "tau": 0.0, "p_value": 1.0}
        ks_trend = {"trend": "insufficient data", "tau": 0.0, "p_value": 1.0}
        if len(perf) >= 4:
            auc_trend = ModelStabilityAnalyzer.mann_kendall_trend(perf['auc'].tolist())
            ks_trend = ModelStabilityAnalyzer.mann_kendall_trend(perf['ks'].tolist())

        return {
            'performance_by_month': perf,
            'score_psi_by_month': psi_monthly,
            'overall_score_psi': round(overall_psi, 6),
            'auc_std': round(float(perf['auc'].std()), 6) if len(perf) > 1 else 0.0,
            'ks_std': round(float(perf['ks'].std()), 6) if len(perf) > 1 else 0.0,
            'baseline_months': baseline_months,
            'auc_trend': auc_trend,
            'ks_trend': ks_trend,
        }
