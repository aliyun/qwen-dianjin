# -*- coding: utf-8 -*-

"""
相关性分析模块

"""

import pandas as pd
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")

class CorrelationCalculator:
    """
    特征相关性分析器

    支持Pearson和Spearman相关系数计算，识别高相关性特征对
    """

    def __init__(self):
        """初始化相关性分析器"""
        self.correlation_matrix: pd.DataFrame = None
        self.high_corr_pairs: List[Dict] = []

    def calculate_correlation_matrix(
            self,
            data: pd.DataFrame,
            features: List[str],
            method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        计算特征间的相关系数矩阵

        Args:
            data: 输入数据
            features: 需要计算相关性的特征列表
            method: 相关系数计算方法，'pearson' 或 'spearman'

        Returns:
            相关系数矩阵
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("method必须是'pearson'或'spearman'")

        feature_data = data[features].dropna()

        if len(feature_data) == 0:
            raise ValueError("所有特征都为空或全为缺失值")

        corr_matrix = pd.DataFrame(index=features, columns=features, dtype=float)

        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i == j:
                    corr_matrix.loc[feat1, feat2] = 1.0
                elif i < j:
                    try:
                        corr_val = feature_data[feat1].corr(feature_data[feat2], method=method)
                        corr_matrix.loc[feat1, feat2] = float(corr_val)
                        corr_matrix.loc[feat2, feat1] = float(corr_val)
                    except Exception as e:
                        print(f"计算 {feat1} 和 {feat2} 的{method}相关系数时出错: {e}")
                        corr_matrix.loc[feat1, feat2] = 0.0
                        corr_matrix.loc[feat2, feat1] = 0.0

        self.correlation_matrix = corr_matrix
        return corr_matrix

    def get_correlation_summary(
            self,
            data: pd.DataFrame,
            features: List[str],
            method: str = 'pearson',
            threshold: float = 0.7
    ) -> Dict:
        """
        获取相关性分析摘要，包括高相关性特征对

        Args:
            data: 输入数据
            features: 特征列表
            method: 相关系数方法
            threshold: 高相关性阈值，默认0.7

        Returns:
            相关性分析摘要
        """
        corr_matrix = self.calculate_correlation_matrix(data, features, method)

        high_corr_pairs = []
        n_features = len(features)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                feat1, feat2 = features[i], features[j]
                corr_val = abs(corr_matrix.loc[feat1, feat2])

                if corr_val >= threshold:
                    high_corr_pairs.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'correlation': float(corr_matrix.loc[feat1, feat2]),
                        'abs_correlation': float(corr_val)
                    })

        high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        self.high_corr_pairs = high_corr_pairs

        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'high_correlation_count': len(high_corr_pairs),
            'method': method,
            'threshold': threshold
        }
