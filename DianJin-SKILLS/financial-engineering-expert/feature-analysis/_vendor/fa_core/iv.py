# -*- coding: utf-8 -*-

"""
IV值计算模块

"""

import pandas as pd
from typing import List, Dict
from optbinning import OptimalBinning
import traceback
import warnings

warnings.filterwarnings("ignore")

class IVCalculator:
    """
    IV值（信息价值）计算器

    用于计算特征变量的预测能力，通过最优分箱和WoE编码
    评估特征对目标变量的区分能力
    """

    def __init__(self, max_n_bins: int = 8, min_bin_size: float = 0.05):
        """
        初始化IV计算器

        Args:
            max_n_bins: 最大分箱数，默认8
            min_bin_size: 最小分箱比例，默认0.05
        """
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.iv_results: Dict[str, float] = {}
        self.iv_tables: Dict[str, pd.DataFrame] = {}

    def calculate_iv(self, data: pd.DataFrame, target: str, var_names: List[str]) -> Dict[str, float]:
        """
        计算指定变量的IV值（信息价值）

        Args:
            data: 输入数据
            target: 目标变量名（必须为0/1二分类）
            var_names: 需要计算IV的变量列表

        Returns:
            变量名到IV值的映射字典
        """
        filtered_data = data[data[target].isin([0, 1])]

        if len(filtered_data) == 0:
            raise ValueError(f"目标变量 '{target}' 中没有有效的0/1值")

        total_positive_rate = filtered_data[target].mean()

        optb = OptimalBinning(
            name="iv_calculator",
            dtype="numerical",
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size
        )

        for var_name in var_names:
            try:
                optb.fit(filtered_data[var_name], filtered_data[target])
                binning_table = optb.binning_table
                bin_table = binning_table.build()
                rename_dict = {
                    "Count (%)": "ratio",
                    "Count": "total",
                    "Event rate": "positive_rate",
                    "WoE": "woe",
                    "IV": "iv",
                }
                bin_table.rename(columns=rename_dict, inplace=True)

                bin_table["positive_approx_lift"] = bin_table["positive_rate"] / total_positive_rate

                iv_table = bin_table[~bin_table["Bin"].isin(["Special"])][
                    ["Bin", "total", "positive_rate", "ratio", "woe", "iv", "positive_approx_lift"]
                ].copy()

                for percentage_col in ['positive_rate', 'ratio']:
                    iv_table[percentage_col] = iv_table[percentage_col].map("{:.2%}".format)

                for digit_3_col in ['woe', 'iv', 'positive_approx_lift']:
                    iv_table[digit_3_col] = iv_table[digit_3_col].map("{:.3}".format)

                self.iv_results[var_name] = float(iv_table["iv"].iloc[-1])
                self.iv_tables[var_name] = iv_table

            except Exception as e:
                print(f"计算特征 {var_name} 的IV时出错: {e}")
                print(traceback.format_exc())

                self.iv_results[var_name] = 0.0

        return self.iv_results

    def get_iv_table(self, var_name: str) -> pd.DataFrame:
        """获取指定变量的IV分箱表"""
        return self.iv_tables.get(var_name)

    @staticmethod
    def interpret_iv(iv_value: float) -> str:
        """
        解释IV值的预测能力

        Args:
            iv_value: IV值

        Returns:
            IV值解释
        """
        if iv_value < 0.02:
            return "无预测能力"
        elif iv_value < 0.1:
            return "较差的预测能力"
        elif iv_value < 0.3:
            return "中等预测能力"
        elif iv_value < 0.5:
            return "较强的预测能力"
        else:
            return "很强的预测能力"
