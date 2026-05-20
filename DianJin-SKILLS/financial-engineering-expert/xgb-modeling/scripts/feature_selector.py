# -*- coding: utf-8 -*-
"""
特征筛选策略模块 (portable)

基于四大算子（IV、PSI、相关性、统计量），自动生成多套特征筛选方案供建模对比使用。
四大算子从 _vendor/fa_core/ 导入。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

def _import_analyzers():
    """延迟导入四大算子（来自 _vendor/fa_core/）"""
    from fa_core.iv import IVCalculator
    from fa_core.psi import PSICalculator
    from fa_core.correlation import CorrelationCalculator
    from fa_core.statistics import StatisticsCalculator
    return IVCalculator, PSICalculator, CorrelationCalculator, StatisticsCalculator

class FeatureSelector:
    """
    特征筛选器

    基于 IV、PSI、相关性、统计量四大分析结果，
    生成四套特征筛选方案：
    1. 全量入模：基础合格池（IV≥0.02, PSI<0.25, 缺失率<50%）
    2. 去共线性：基础池 + |r|≥0.7 贪心去重
    3. 高预测力：IV≥0.1 + 去共线性
    4. 稳定性优先：PSI<0.1 + 去共线性
    """

    def __init__(
        self,
        iv_threshold: float = 0.02,
        psi_threshold: float = 0.25,
        missing_threshold: float = 0.5,
        corr_threshold: float = 0.7,
        high_iv_threshold: float = 0.1,
        stable_psi_threshold: float = 0.1,
    ):
        self.iv_threshold = iv_threshold
        self.psi_threshold = psi_threshold
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.high_iv_threshold = high_iv_threshold
        self.stable_psi_threshold = stable_psi_threshold

        self.iv_results: Dict[str, float] = {}
        self.psi_results: Dict[str, float] = {}
        self.stats_results: Dict[str, Dict] = {}
        self.corr_matrix: Optional[pd.DataFrame] = None

    def analyze(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        baseline_data: Optional[pd.DataFrame] = None,
        comparison_data: Optional[pd.DataFrame] = None,
    ) -> None:
        IVCalculator, PSICalculator, CorrelationCalculator, StatisticsCalculator = _import_analyzers()

        logger.info("开始特征分析...")

        logger.info("  计算基础统计...")
        stats_calc = StatisticsCalculator()
        self.stats_results = stats_calc.calculate_basic_statistics(data, features)

        logger.info("  计算 IV 值...")
        iv_calc = IVCalculator()
        self.iv_results = iv_calc.calculate_iv(data, target, features)

        if baseline_data is not None and comparison_data is not None:
            logger.info("  计算 PSI...")
            psi_calc = PSICalculator()
            self.psi_results, _ = psi_calc.calculate(baseline_data, comparison_data, features)
        else:
            logger.info("  跳过 PSI（未提供切分数据）")
            self.psi_results = {f: 0.0 for f in features}

        logger.info("  计算相关性...")
        corr_calc = CorrelationCalculator()
        corr_summary = corr_calc.get_correlation_summary(
            data, features, method='pearson', threshold=self.corr_threshold
        )
        self.corr_matrix = corr_summary['correlation_matrix']

        logger.info("特征分析完成")

    def _greedy_decorr(self, candidates: List[str]) -> List[str]:
        if self.corr_matrix is None or len(candidates) == 0:
            return candidates

        sorted_feats = sorted(candidates, key=lambda f: self.iv_results.get(f, 0), reverse=True)
        selected = []

        for feat in sorted_feats:
            if feat not in self.corr_matrix.index:
                selected.append(feat)
                continue

            conflict = False
            for sel in selected:
                if sel in self.corr_matrix.columns:
                    corr_val = abs(self.corr_matrix.loc[feat, sel])
                    if corr_val >= self.corr_threshold:
                        conflict = True
                        break

            if not conflict:
                selected.append(feat)

        return selected

    def get_qualified_pool(self) -> List[str]:
        qualified = []
        for feat in self.iv_results.keys():
            iv_val = self.iv_results.get(feat, 0)
            psi_val = self.psi_results.get(feat, 0)
            miss_rate = self.stats_results.get(feat, {}).get('miss_rate', 1)

            if (iv_val >= self.iv_threshold and
                psi_val < self.psi_threshold and
                miss_rate < self.missing_threshold):
                qualified.append(feat)

        return sorted(qualified, key=lambda f: self.iv_results.get(f, 0), reverse=True)

    def get_full_scheme(self) -> Tuple[List[str], Dict[str, Any]]:
        features = self.get_qualified_pool()
        info = {
            'strategy': '全量入模',
            'description': f'IV≥{self.iv_threshold}, PSI<{self.psi_threshold}, 缺失率<{self.missing_threshold:.0%}',
            'n_features': len(features),
        }
        return features, info

    def get_decorr_scheme(self) -> Tuple[List[str], Dict[str, Any]]:
        pool = self.get_qualified_pool()
        features = self._greedy_decorr(pool)
        info = {
            'strategy': '去共线性',
            'description': f'基础池 + |r|≥{self.corr_threshold} 贪心去重',
            'n_features': len(features),
        }
        return features, info

    def get_high_iv_scheme(self) -> Tuple[List[str], Dict[str, Any]]:
        pool = self.get_qualified_pool()
        high_iv_pool = [f for f in pool if self.iv_results.get(f, 0) >= self.high_iv_threshold]
        features = self._greedy_decorr(high_iv_pool)
        info = {
            'strategy': '高预测力',
            'description': f'IV≥{self.high_iv_threshold} + 去共线性',
            'n_features': len(features),
        }
        return features, info

    def get_stable_scheme(self) -> Tuple[List[str], Dict[str, Any]]:
        pool = self.get_qualified_pool()
        stable_pool = [f for f in pool if self.psi_results.get(f, 0) < self.stable_psi_threshold]
        features = self._greedy_decorr(stable_pool)
        info = {
            'strategy': '稳定性优先',
            'description': f'PSI<{self.stable_psi_threshold} + 去共线性',
            'n_features': len(features),
        }
        return features, info

    def get_all_schemes(self) -> Dict[str, Tuple[List[str], Dict[str, Any]]]:
        return {
            'full': self.get_full_scheme(),
            'decorr': self.get_decorr_scheme(),
            'high_iv': self.get_high_iv_scheme(),
            'stable': self.get_stable_scheme(),
        }

    def get_analysis_summary(self) -> Dict[str, Any]:
        return {
            'iv_results': self.iv_results,
            'psi_results': self.psi_results,
            'stats_results': self.stats_results,
            'qualified_pool_size': len(self.get_qualified_pool()),
        }
