# -*- coding: utf-8 -*-
"""
模型评估模块

"""
from __future__ import annotations

from typing import Any

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from optbinning import OptimalBinning
    HAS_OPTBINNING = True
except ImportError:
    HAS_OPTBINNING = False

class ModelEvaluator:
    """
    模型评估器

    提供 AUC、KS、Gini、Lift 等评估指标计算，
    以及多特征集合模型的横向对比
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series
    ) -> dict[str, float]:
        """
        全面评估模型表现

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测正类概率

        Returns:
            包含 AUC、KS、Gini 的字典
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        auc = roc_auc_score(y_true, y_prob)
        ks = ModelEvaluator.calculate_ks(y_true, y_prob)
        gini = 2 * auc - 1

        return {
            'auc': round(auc, 6),
            'ks': round(ks, 6),
            'gini': round(gini, 6),
        }

    @staticmethod
    def calculate_roc_data(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        max_points: int = 30,
    ) -> dict:
        """
        计算 ROC 曲线数据点（采样后），用于前端 roc-curve artifact 渲染

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            max_points: 最大采样点数，控制输出数据量

        Returns:
            {"fpr": [...], "tpr": [...], "auc": float}
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        fpr_full, tpr_full, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        # 等间距采样，保留首尾点
        n = len(fpr_full)
        if n > max_points:
            indices = np.linspace(0, n - 1, max_points, dtype=int)
            fpr_sampled = fpr_full[indices]
            tpr_sampled = tpr_full[indices]
        else:
            fpr_sampled = fpr_full
            tpr_sampled = tpr_full

        return {
            'fpr': [round(float(v), 6) for v in fpr_sampled],
            'tpr': [round(float(v), 6) for v in tpr_sampled],
            'auc': round(float(auc_val), 6),
        }

    @staticmethod
    def calculate_ks(
        y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series
    ) -> float:
        """
        计算 KS 统计量

        KS = max|累计正样本占比 - 累计负样本占比|

        Args:
            y_true: 真实标签
            y_prob: 预测概率

        Returns:
            KS 值
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        # 按概率降序排列
        order = np.argsort(-y_prob)
        y_sorted = y_true[order]

        total_pos = y_sorted.sum()
        total_neg = len(y_sorted) - total_pos

        if total_pos == 0 or total_neg == 0:
            return 0.0

        cum_pos = np.cumsum(y_sorted) / total_pos
        cum_neg = np.cumsum(1 - y_sorted) / total_neg

        ks = np.max(np.abs(cum_pos - cum_neg))
        return float(ks)

    @staticmethod
    def calculate_lift_table(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        计算分箱 Lift 表（按风险从高到低）

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            n_bins: 分箱数

        Returns:
            包含 count, positive, positive_rate, avg_score, lift 的 DataFrame
        """
        df = pd.DataFrame({
            'y': np.array(y_true),
            'prob': np.array(y_prob),
        })
        df['bin'] = pd.qcut(df['prob'], q=n_bins, labels=False, duplicates='drop')

        result = df.groupby('bin').agg(
            count=('y', 'count'),
            positive=('y', 'sum'),
            avg_score=('prob', 'mean'),
            min_score=('prob', 'min'),
            max_score=('prob', 'max'),
        ).reset_index()

        result['positive_rate'] = result['positive'] / result['count']
        overall_rate = df['y'].mean()
        result['lift'] = result['positive_rate'] / overall_rate
        result['cum_positive'] = result.sort_values('bin', ascending=False)['positive'].cumsum().values[::-1]
        result['cum_positive_rate'] = result['cum_positive'] / result['positive'].sum()

        # 按风险从高到低排列
        result = result.sort_values('bin', ascending=False).reset_index(drop=True)
        result.index = range(1, len(result) + 1)
        result.index.name = 'decile'

        return result

    @staticmethod
    def calculate_ks_curve_data(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        n_bins: int = 10,
    ) -> dict:
        """
        计算 KS 曲线数据点，用于前端 ks-curve artifact 渲染

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            n_bins: 十分位数量

        Returns:
            {"cum_pos_rate": [...], "cum_neg_rate": [...], "ks_value": float, "ks_decile": int}
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        # 按概率降序排列
        order = np.argsort(-y_prob)
        y_sorted = y_true[order]

        total_pos = y_sorted.sum()
        total_neg = len(y_sorted) - total_pos

        if total_pos == 0 or total_neg == 0:
            return {
                'cum_pos_rate': [0.0] * (n_bins + 1),
                'cum_neg_rate': [0.0] * (n_bins + 1),
                'ks_value': 0.0,
                'ks_decile': 0,
            }

        # 分成 n_bins 等份
        bin_size = len(y_sorted) // n_bins
        cum_pos = [0.0]
        cum_neg = [0.0]

        for i in range(1, n_bins + 1):
            end_idx = i * bin_size if i < n_bins else len(y_sorted)
            chunk = y_sorted[:end_idx]
            cum_pos.append(float(chunk.sum() / total_pos))
            cum_neg.append(float((end_idx - chunk.sum()) / total_neg))

        # 找到 KS 最大值
        ks_diffs = [abs(cum_pos[i] - cum_neg[i]) for i in range(len(cum_pos))]
        ks_decile = int(np.argmax(ks_diffs))
        ks_value = max(ks_diffs)

        return {
            'cum_pos_rate': [round(v, 6) for v in cum_pos],
            'cum_neg_rate': [round(v, 6) for v in cum_neg],
            'ks_value': round(float(ks_value), 6),
            'ks_decile': ks_decile,
        }

    @staticmethod
    def calculate_lift_chart_data(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        n_bins: int = 10,
    ) -> dict:
        """
        计算 Lift Chart 数据，用于前端 lift-chart artifact 渲染

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            n_bins: 分箱数

        Returns:
            {"deciles": [...], "lift": [...], "cum_lift": [...], "positive_rate": [...], "overall_rate": float}
        """
        lift_table = ModelEvaluator.calculate_lift_table(y_true, y_prob, n_bins)

        total_pos = lift_table['positive'].sum()
        total_count = lift_table['count'].sum()
        overall_rate = total_pos / total_count if total_count > 0 else 0

        # 计算累计 Lift
        cum_positive = lift_table['positive'].cumsum()
        cum_count = lift_table['count'].cumsum()
        cum_lift = (cum_positive / cum_count) / overall_rate if overall_rate > 0 else cum_positive * 0

        return {
            'deciles': list(range(1, len(lift_table) + 1)),
            'lift': [round(float(v), 4) for v in lift_table['lift'].values],
            'cum_lift': [round(float(v), 4) for v in cum_lift.values],
            'positive_rate': [round(float(v), 6) for v in lift_table['positive_rate'].values],
            'overall_rate': round(float(overall_rate), 6),
        }

    @staticmethod
    def calculate_lift_table_optbinning(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        max_n_bins: int = 8,
        min_bin_size: float = 0.05,
    ) -> pd.DataFrame | None:
        """
        使用 IV 最优分箱（OptimalBinning）计算 Lift 表。

        与等频分箱不同，最优分箱基于信息价值最大化原则，
        在正负样本分布差异最大的位置切分，更适合风控场景。

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测正类概率
            max_n_bins: 最大分箱数
            min_bin_size: 最小分箱比例

        Returns:
            分箱 Lift DataFrame，optbinning 未安装或失败时返回 None
            列：count, positive, positive_rate, avg_score, min_score, max_score, lift
        """
        if not HAS_OPTBINNING:
            return None

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        mask = ~np.isnan(y_true_arr) & ~np.isnan(y_prob_arr)
        y_true_valid = y_true_arr[mask]
        y_prob_valid = y_prob_arr[mask]

        if len(y_true_valid) == 0:
            return None

        overall_rate = float(y_true_valid.mean())

        try:
            optb = OptimalBinning(
                name="lift_optbinning",
                dtype="numerical",
                max_n_bins=max_n_bins,
                min_bin_size=min_bin_size,
            )
            optb.fit(y_prob_valid, y_true_valid)
            binning_table = optb.binning_table
            bin_table = binning_table.build()

            # 重命名列
            rename_dict = {
                "Count (%)": "ratio",
                "Count": "total",
                "Event rate": "positive_rate",
            }
            bin_table = bin_table.rename(columns=rename_dict)

            # 过滤掉 Special / Missing 行，只保留数值区间
            bin_table = bin_table[
                ~bin_table["Bin"].isin(["Special", "Missing"])
            ].copy()

            if len(bin_table) == 0:
                return None

            # 解析区间上下界
            import re

            def _parse_bounds(bin_str: str) -> tuple[float | None, float | None]:
                m = re.search(r'([\[(])([^,]+),\s*([^\])]+)([\])])', str(bin_str))
                if not m:
                    return None, None
                left_s = m.group(2).strip()
                right_s = m.group(3).strip()
                left = float(left_s) if left_s not in ("-inf", "inf") else None
                right = float(right_s) if right_s not in ("-inf", "inf") else None
                return left, right

            bounds = bin_table["Bin"].apply(_parse_bounds)
            bin_table["min_score"] = bounds.apply(lambda x: x[0])
            bin_table["max_score"] = bounds.apply(lambda x: x[1])
            def _safe_avg(bounds_tuple):
                lo, hi = bounds_tuple
                lo_ok = lo is not None and not (isinstance(lo, float) and math.isnan(lo))
                hi_ok = hi is not None and not (isinstance(hi, float) and math.isnan(hi))
                if lo_ok and hi_ok:
                    return (lo + hi) / 2
                elif lo_ok:
                    return lo
                elif hi_ok:
                    return hi
                return 0.0

            bin_table["avg_score"] = bounds.apply(_safe_avg)

            # 计算正样本数（total * positive_rate）
            bin_table["positive"] = (bin_table["total"] * bin_table["positive_rate"]).round().astype(int)

            # 按风险从高到低排列（avg_score 降序）
            bin_table = bin_table.sort_values("avg_score", ascending=False).reset_index(drop=True)
            bin_table.index = range(1, len(bin_table) + 1)
            bin_table.index.name = "decile"

            # 计算 lift
            bin_table["lift"] = bin_table["positive_rate"] / overall_rate

            # 选择并排序输出列，与等频版本对齐
            result = bin_table[["total", "positive", "positive_rate", "avg_score", "min_score", "max_score", "lift"]].copy()
            result.columns = ["count", "positive", "positive_rate", "avg_score", "min_score", "max_score", "lift"]

            return result

        except Exception:
            return None

    @staticmethod
    def calculate_lift_chart_data_optbinning(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        max_n_bins: int = 8,
        min_bin_size: float = 0.05,
    ) -> dict | None:
        """
        使用 IV 最优分箱计算 Lift Chart 数据，用于前端 lift-chart artifact 渲染。

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            max_n_bins: 最大分箱数
            min_bin_size: 最小分箱比例

        Returns:
            {"deciles": [...], "lift": [...], "cum_lift": [...],
             "positive_rate": [...], "overall_rate": float, "bin_ranges": [...]}
            optbinning 未安装或失败时返回 None
        """
        lift_table = ModelEvaluator.calculate_lift_table_optbinning(
            y_true, y_prob, max_n_bins=max_n_bins, min_bin_size=min_bin_size
        )
        if lift_table is None or len(lift_table) == 0:
            return None

        total_pos = lift_table['positive'].sum()
        total_count = lift_table['count'].sum()
        overall_rate = total_pos / total_count if total_count > 0 else 0

        # 计算累计 Lift
        cum_positive = lift_table['positive'].cumsum()
        cum_count = lift_table['count'].cumsum()
        cum_lift = (cum_positive / cum_count) / overall_rate if overall_rate > 0 else cum_positive * 0

        # 构造区间标签
        bin_ranges = []
        for _, row in lift_table.iterrows():
            min_s = row['min_score']
            max_s = row['max_score']
            min_ok = pd.notna(min_s)
            max_ok = pd.notna(max_s)
            if min_ok and max_ok:
                bin_ranges.append(f"[{min_s:.4f}, {max_s:.4f})")
            elif min_ok:
                bin_ranges.append(f"[{min_s:.4f}, +∞)")
            elif max_ok:
                bin_ranges.append(f"(-∞, {max_s:.4f})")
            else:
                bin_ranges.append("-")

        return {
            'deciles': list(range(1, len(lift_table) + 1)),
            'lift': [round(float(v), 4) for v in lift_table['lift'].values],
            'cum_lift': [round(float(v), 4) for v in cum_lift.values],
            'positive_rate': [round(float(v), 6) for v in lift_table['positive_rate'].values],
            'overall_rate': round(float(overall_rate), 6),
            'bin_ranges': bin_ranges,
        }

    @staticmethod
    def calculate_bad_capture_rate(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        topk_pcts: list[int] | None = None,
    ) -> dict:
        """
        计算 Bad Capture Rate @ K%

        即：按得分从高到低排序，拒绝前 K% 人群能捕获多少比例的坏客户。
        比 KS 更接近风控业务决策语言。

        Returns:
            {
                "topk_pcts": [5, 10, 20, 30],
                "bad_capture_rates": [0.28, 0.45, 0.62, 0.75],
                "reject_rates": [0.05, 0.10, 0.20, 0.30],
            }
        """
        if topk_pcts is None:
            topk_pcts = [5, 10, 20, 30, 40, 50]

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        order = np.argsort(-y_prob)
        y_sorted = y_true[order]
        total_bad = y_sorted.sum()

        if total_bad == 0:
            return {"topk_pcts": topk_pcts, "bad_capture_rates": [0.0] * len(topk_pcts)}

        n = len(y_sorted)
        bad_capture_rates = []
        for pct in topk_pcts:
            k = max(1, int(n * pct / 100))
            captured = y_sorted[:k].sum()
            bad_capture_rates.append(round(float(captured / total_bad), 6))

        return {
            "topk_pcts": topk_pcts,
            "bad_capture_rates": bad_capture_rates,
        }

    @staticmethod
    def calculate_brier_score(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
    ) -> float:
        """
        计算 Brier Score（概率校准误差）

        Brier Score = mean((p - y)^2)
        越小越好，完美校准=0，随机预测≈0.25（正样本率0.5时）。
        用于评估模型概率输出是否可信（定价/额度场景关键指标）。
        """
        y_true = np.array(y_true, dtype=float)
        y_prob = np.array(y_prob, dtype=float)
        return round(float(np.mean((y_prob - y_true) ** 2)), 6)

    @staticmethod
    def calculate_score_iv_table(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        max_n_bins: int = 8,
        min_bin_size: float = 0.05,
    ) -> pd.DataFrame | None:
        """
        对模型预测分数进行 IV 最优分箱，生成评分卡风格的分箱表。

        使用 OptimalBinning 对预测概率做最优分箱，计算每箱的
        样本数、占比、正样本数、正样本率、WoE、IV。

        Args:
            y_true: 真实标签 (0/1)
            y_prob: 预测正类概率
            max_n_bins: 最大分箱数
            min_bin_size: 最小分箱比例

        Returns:
            分箱明细 DataFrame（含合计行），optbinning 未安装时返回 None
        """
        if not HAS_OPTBINNING:
            return None

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        # 过滤有效样本
        mask = ~np.isnan(y_true_arr) & ~np.isnan(y_prob_arr)
        y_true_valid = y_true_arr[mask]
        y_prob_valid = y_prob_arr[mask]

        if len(y_true_valid) == 0:
            return None

        total_positive_rate = y_true_valid.mean()

        try:
            optb = OptimalBinning(
                name="score_iv",
                dtype="numerical",
                max_n_bins=max_n_bins,
                min_bin_size=min_bin_size,
            )
            optb.fit(y_prob_valid, y_true_valid)
            binning_table = optb.binning_table
            bin_table = binning_table.build()

            # 重命名列与特征 IV 模块保持一致
            rename_dict = {
                "Count (%)": "ratio",
                "Count": "total",
                "Event rate": "positive_rate",
                "WoE": "woe",
                "IV": "iv",
            }
            bin_table = bin_table.rename(columns=rename_dict)

            # 计算正样本数
            bin_table["positive"] = (bin_table["total"] * bin_table["positive_rate"]).round().astype(int)

            # 选择并排序输出列
            result = bin_table[["Bin", "total", "positive", "positive_rate", "ratio", "woe", "iv"]].copy()

            # 格式化
            result["positive_rate"] = result["positive_rate"].map("{:.2%}".format)
            result["ratio"] = result["ratio"].map("{:.2%}".format)
            result["woe"] = result["woe"].map("{:.4f}".format)
            result["iv"] = result["iv"].map("{:.4f}".format)

            # 重命名列名为中文（与报告展示一致）
            result.columns = ["区间", "样本数", "正样本数", "正样本率", "占比", "WoE", "IV"]

            return result

        except Exception:
            return None

    @staticmethod
    def calculate_calibration_data(
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        n_bins: int = 10,
    ) -> dict:
        """
        计算校准曲线数据（Calibration Curve）

        将预测概率分成 n_bins 个区间，对比每个区间内的预测均值与实际正样本率。
        完美校准时两条曲线重合。

        适配前端 echarts artifact 渲染：
        返回格式可直接嵌入 echarts line 图，展示预测概率 vs 实际正样本率。

        Returns:
            {
                "mean_predicted": [...],      # 每箱预测概率均值
                "fraction_positive": [...],   # 每箱实际正样本率
                "counts": [...],              # 每箱样本数
                "brier_score": float,
            }
        """
        y_true = np.array(y_true, dtype=float)
        y_prob = np.array(y_prob, dtype=float)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins[1:-1])

        mean_predicted = []
        fraction_positive = []
        counts = []

        for i in range(n_bins):
            mask = bin_ids == i
            if mask.sum() == 0:
                continue
            mean_predicted.append(round(float(y_prob[mask].mean()), 6))
            fraction_positive.append(round(float(y_true[mask].mean()), 6))
            counts.append(int(mask.sum()))

        brier = ModelEvaluator.calculate_brier_score(y_true, y_prob)

        return {
            "mean_predicted": mean_predicted,
            "fraction_positive": fraction_positive,
            "counts": counts,
            "brier_score": brier,
        }

    @staticmethod
    def compare_feature_sets(results_dict: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """
        对比多个特征集合的模型表现

        Args:
            results_dict: {
                feature_set_name: {
                    'train': {auc, ks, gini},
                    'test': {auc, ks, gini},
                    'features': [feature_list]
                }
            }

        Returns:
            对比 DataFrame
        """
        rows = []
        for name, res in results_dict.items():
            row = {
                'feature_set': name,
                'n_features': len(res.get('features', [])),
                'train_auc': res['train']['auc'],
                'test_auc': res['test']['auc'],
                'train_ks': res['train']['ks'],
                'test_ks': res['test']['ks'],
                'auc_gap': round(res['train']['auc'] - res['test']['auc'], 6),
                'ks_gap': round(res['train']['ks'] - res['test']['ks'], 6),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('test_auc', ascending=False).reset_index(drop=True)
        return df
