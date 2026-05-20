# -*- coding: utf-8 -*-
"""
统计显著性检验模块（从 auto-experiment/scripts/significance.py 下沉）。

借鉴 pi-autoresearch 的置信度评分机制：
  - SignificanceTester: MAD 噪声估计 + 绝对阈值兜底
  - CrossValidationTester: K-Fold paired t-test + Bootstrap CI

"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple

class SignificanceTester:
    """
    统计显著性检验器

    使用 MAD (Median Absolute Deviation) 作为噪声估计，
    判断模型改进是否显著超出随机波动。

    借鉴 pi-autoresearch 的置信度评分:
    - ≥ 2.0×: 改进可能是真实的 (绿色)
    - 1.0-2.0×: 高于噪声但边缘 (黄色)
    - < 1.0×: 在噪声范围内 (红色)
    """

    def __init__(self, threshold: float = 2.0, min_samples: int = 3):
        self.threshold = threshold
        self.min_samples = min_samples
        self.history: List[float] = []

    def add_observation(self, metric: float) -> None:
        """添加一个观测值到历史"""
        self.history.append(metric)

    def calculate_mad(self) -> float:
        """计算 MAD (Median Absolute Deviation)"""
        if len(self.history) < 2:
            return 0.0
        arr = np.array(self.history)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return float(mad)

    def calculate_confidence(self, improvement: float) -> float:
        """计算置信度得分：confidence = |improvement| / MAD"""
        mad = self.calculate_mad()
        if mad == 0 or len(self.history) < self.min_samples:
            return float('nan')
        return abs(improvement) / mad

    def is_significant(self, improvement: float) -> Tuple[bool, float, str]:
        """判断改进是否统计显著。

        Returns:
            (是否显著, 置信度, 颜色标签)
        """
        import math
        confidence = self.calculate_confidence(improvement)

        if math.isnan(confidence):
            is_sig = improvement > 0.005
            color = "🟢" if is_sig else "🔴"
            return is_sig, 0.0, color

        if confidence >= self.threshold:
            return True, confidence, "🟢"
        elif confidence >= 1.0:
            return False, confidence, "🟡"
        else:
            return False, confidence, "🔴"

    def get_summary(self) -> dict:
        """获取统计摘要"""
        if not self.history:
            return {
                'n_samples': 0, 'mad': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0,
            }
        arr = np.array(self.history)
        return {
            'n_samples': len(self.history),
            'mad': self.calculate_mad(),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)) if len(arr) > 1 else 0.0,
        }

    def format_confidence(self, confidence: float) -> str:
        """格式化置信度显示"""
        import math
        if math.isnan(confidence) or confidence == 0.0:
            return "⚪ 样本不足（绝对阈值判断）"
        if confidence >= self.threshold:
            return f"🟢 {confidence:.1f}×"
        elif confidence >= 1.0:
            return f"🟡 {confidence:.1f}×"
        else:
            return f"🔴 {confidence:.1f}×"

class CrossValidationTester:
    """基于交叉验证的显著性检验。"""

    @staticmethod
    def paired_t_test(
        baseline_scores: List[float],
        new_scores: List[float],
        alpha: float = 0.05,
    ) -> Tuple[bool, float, float]:
        """配对 t 检验。

        Returns:
            (是否显著, t 统计量, p 值)
        """
        from scipy import stats

        if len(baseline_scores) != len(new_scores):
            raise ValueError("两组分数长度必须相同")
        if len(baseline_scores) < 2:
            return False, 0.0, 1.0

        t_stat, p_value = stats.ttest_rel(new_scores, baseline_scores)
        is_significant = p_value < alpha and t_stat > 0
        return is_significant, float(t_stat), float(p_value)

    @staticmethod
    def bootstrap_confidence_interval(
        baseline_scores: List[float],
        new_scores: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Bootstrap 置信区间。

        Returns:
            (差异均值, 下界, 上界)
        """
        differences = np.array(new_scores) - np.array(baseline_scores)
        n = len(differences)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n, size=n)
            bootstrap_means.append(np.mean(differences[indices]))

        bootstrap_means = np.array(bootstrap_means)
        mean_diff = float(np.mean(differences))

        alpha = 1 - confidence_level
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

        return mean_diff, lower, upper
