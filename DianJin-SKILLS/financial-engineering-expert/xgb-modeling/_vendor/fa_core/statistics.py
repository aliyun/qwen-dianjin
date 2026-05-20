# -*- coding: utf-8 -*-

"""
统计分析模块

"""

import pandas as pd
from typing import List, Dict
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

class StatisticsCalculator:
    """
    基础统计指标计算器

    提供全面的统计分析功能，包括极值、均值、标准差、偏度、峰度等
    """

    def __init__(self):
        """初始化统计计算器"""
        self.stats_results: Dict[str, Dict] = {}

    def calculate_basic_statistics(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Dict]:
        """
        计算基础统计指标和数据质量指标

        指标清单：
        【统计指标】
        - 极值与均值：Min, Max, Mean, Median
        - 标准差/方差：Std, Variance
        - 偏度 (Skewness)：衡量分布是否左偏或右偏
        - 峰度 (Kurtosis)：衡量分布是否陡峭或平坦
        - 变异系数 (CV)：Std / Mean，用于比较不同量纲特征的离散程度

        【数据质量指标】
        - 缺失率 (Missing Rate)：缺失值占总样本的比例
        - 基数 (Cardinality)：唯一值的数量
        - 众数占比 (Mode Ratio)：出现最频繁的值占总样本的比例

        Args:
            data: 输入数据
            features: 需要计算统计指标的特征列表

        Returns:
            每个特征的统计指标字典,包含所有统计和质量指标
        """
        stats_results = {}

        for feature in features:
            try:
                series = data[feature]
                clean_series = series.dropna()

                total_count = len(series)
                valid_count = len(clean_series)
                missing_count = total_count - valid_count

                if valid_count == 0:
                    stats_results[feature] = self._empty_stats(total_count)
                    continue

                min_val = float(clean_series.min())
                max_val = float(clean_series.max())
                mean_val = float(clean_series.mean())
                median_val = float(clean_series.median())
                std_val = float(clean_series.std())
                variance_val = float(clean_series.var())

                skewness_val = float(stats.skew(clean_series))
                kurtosis_val = float(stats.kurtosis(clean_series))

                cv_val = std_val / mean_val if mean_val != 0 else None
                if cv_val is not None:
                    cv_val = float(cv_val)

                missing_rate = missing_count / total_count if total_count > 0 else 0.0
                cardinality = int(clean_series.nunique())

                value_counts = clean_series.value_counts()
                if len(value_counts) > 0:
                    mode_count = value_counts.iloc[0]
                    mode_ratio = float(mode_count / valid_count)
                else:
                    mode_ratio = 0.0

                stats_results[feature] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'variance': variance_val,
                    'skewness': skewness_val,
                    'kurtosis': kurtosis_val,
                    'cv': cv_val,
                    'count': valid_count,
                    'missing_count': missing_count,
                    'miss_rate': missing_rate,
                    'cardinality': cardinality,
                    'mode_ratio': mode_ratio
                }

            except Exception as e:
                print(f"计算特征 {feature} 的统计指标时出错: {e}")
                stats_results[feature] = self._empty_stats(
                    len(data[feature]) if feature in data.columns else 0
                )

        self.stats_results = stats_results
        return stats_results

    @staticmethod
    def _empty_stats(total_count: int) -> Dict:
        """返回空统计结果,包含所有指标"""
        return {
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'variance': None,
            'skewness': None,
            'kurtosis': None,
            'cv': None,
            'count': 0,
            'missing_count': total_count,
            'miss_rate': 1.0 if total_count > 0 else 0.0,
            'cardinality': 0,
            'mode_ratio': 0.0
        }
