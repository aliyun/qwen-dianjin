# -*- coding: utf-8 -*-

"""
PSI计算模块

"""

import logging

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PSICalculator:
    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.psi_result: Dict[str, float] = {}
        self.psi_detail_result: Dict[str, pd.DataFrame] = {}

    def calculate(
            self,
            baseline_data: pd.DataFrame,
            comparison_data: pd.DataFrame,
            features: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:

        psi_dict = {}
        detail_dict = {}

        for feature in features:
            if feature not in baseline_data.columns or feature not in comparison_data.columns:
                continue
            try:
                psi, df_detail = self._calculate_single_feature(
                    baseline_data[feature],
                    comparison_data[feature]
                )
                psi_dict[feature] = psi
                detail_dict[feature] = df_detail

            except Exception as e:
                # logger.exception 自动附带完整 traceback（含行号与调用栈）
                logger.exception("[PSI] feature=%s 计算失败: %s", feature, e)
                psi_dict[feature] = 0.0
                detail_dict[feature] = pd.DataFrame()
            self.psi_result = psi_dict
            self.psi_detail_result = detail_dict
        return psi_dict, detail_dict

    def _calculate_single_feature(self, base_series: pd.Series, comp_series: pd.Series):
        # --- 1. 准备正数分箱边界 (基于基准集) ---
        base_pos = base_series[base_series > 0]
        pos_bins = []
        pos_labels = []

        if len(base_pos) > 0:
            try:
                # 计算分位点
                _, pos_bins = pd.qcut(base_pos, q=self.n_bins, retbins=True, duplicates='drop')
                # 修正边界以覆盖所有正数
                pos_bins[0] = -np.inf
                pos_bins[-1] = np.inf

                # 预先生成正数区间的字符串标签，用于后续定义排序顺序
                # 使用 pd.cut 的一次空跑来获取标准的区间字符串格式
                dummy_cut = pd.cut(pos_bins[:-1], bins=pos_bins)
                # 注意：pd.cut 生成的是 Interval 对象，我们需要转为字符串
                # 实际上直接对一个虚拟序列切分是最准确获取 label 的方式
                # 这里为了简单，我们稍后在分箱时动态获取
            except ValueError:
                pass

        # --- 2. 核心分箱函数 ---
        def get_range_series(series: pd.Series, bins: np.ndarray) -> pd.Series:
            # 初始化为 object 类型
            res = pd.Series(index=series.index, dtype=object)

            # 1. 特殊值处理
            res[series.isna()] = 'Missing'
            res[(series < 0) & (series.notna())] = '(-inf, 0)'
            res[(series == 0) & (series.notna())] = '0'

            # 2. 正数处理
            mask_pos = (series > 0) & (series.notna())
            if mask_pos.any() and len(bins) > 0:
                res[mask_pos] = pd.cut(series[mask_pos], bins=bins).astype(str)

            return res

        # 执行分箱
        base_ranges = get_range_series(base_series, pos_bins)
        comp_ranges = get_range_series(comp_series, pos_bins)

        # --- 3. 定义排序逻辑 ---
        # 我们需要收集所有出现的 range，并按照业务逻辑排序
        # 顺序：Missing -> (-inf, 0) -> 0 -> 正数区间(从小到大)

        # 获取正数部分的标准区间字符串列表 (如果有的话)
        sorted_categories = ['Missing', '(-inf, 0)', '0']
        if len(pos_bins) > 0:
            # 构造一个临时的 Series 包含所有可能的区间，利用 Pandas 自动排序
            # 这里利用 IntervalIndex 自动生成的字符串
            intervals = pd.interval_range(start=0, periods=len(pos_bins) - 1, freq=1)  # 占位
            # 真正的方法：手动模拟 pd.cut 的输出格式，或者直接从 base_ranges 里提取正数部分去重排序
            # 最稳健的方法：
            unique_pos_ranges = base_ranges[base_series > 0].unique().tolist()

            # 过滤掉 None/NaN，并按区间左边界排序
            def sort_interval_str(x):
                try:
                    return float(x.split(',')[0].replace('(', '').replace('[', ''))
                except:
                    return float('inf')

            unique_pos_ranges.sort(key=sort_interval_str)
            sorted_categories.extend(unique_pos_ranges)

        # --- 4. 计算占比并合并 ---
        # 转换为 Categorical 类型以确保 groupby 时包含所有类别（即使某一边计数为0）并保持顺序
        cat_type = pd.CategoricalDtype(categories=sorted_categories, ordered=True)

        base_ranges = base_ranges.astype(cat_type)
        comp_ranges = comp_ranges.astype(cat_type)

        # 统计占比 (利用 Categorical 的特性，value_counts 会自动对齐)
        df = pd.DataFrame({
            'baseline_prop': base_ranges.value_counts(normalize=True, sort=False),
            'comparison_prop': comp_ranges.value_counts(normalize=True, sort=False)
        }).fillna(0)

        # --- 5. 计算 PSI ---
        df['base_smooth'] = df['baseline_prop'] + self.epsilon
        df['comp_smooth'] = df['comparison_prop'] + self.epsilon
        df['psi'] = (df['comp_smooth'] - df['base_smooth']) * np.log(df['comp_smooth'] / df['base_smooth'])

        total_psi = df['psi'].sum()

        # --- 6. 格式化输出 ---
        # 此时 index 已经是排好序的 range 字符串
        final_df = df[['baseline_prop', 'comparison_prop']].reset_index()
        final_df['diff'] = final_df['comparison_prop'] - final_df['baseline_prop']
        final_df.columns = ['range', 'baseline_prop', 'comparison_prop', 'diff']

        # 过滤掉从未出现过的 Bucket (除了特殊的 Missing/Negative/Zero 建议保留)
        # 如果希望完全保留所有定义的区间（即使是0），上面的逻辑已经满足
        # 如果希望只显示有值的区间，可以在这里做 filter，但计算 PSI 必须保留 0 值行

        return total_psi, final_df

    def get_psi_table(self, var_name: str) -> pd.DataFrame:
        """获取指定变量的PSI表"""
        return self.psi_detail_result.get(var_name)
