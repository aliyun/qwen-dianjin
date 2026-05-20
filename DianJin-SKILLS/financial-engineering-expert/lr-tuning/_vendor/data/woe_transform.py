# -*- coding: utf-8 -*-
"""WoE (Weight of Evidence) 编码桥接层。

职责：基于 OptimalBinning 对原始特征进行 WoE 编码，供 LR / ScoreCard 等算法使用。
复用系统已有的 optbinning 依赖（evaluator.py 已使用 OptimalBinning）。

核心用法：
    encoder = WoEEncoder()
    encoder.fit(X_train, y_train, feature_names)
    X_train_woe = encoder.transform(X_train)
    X_val_woe = encoder.transform(X_val)

设计要点：
1. fit 阶段用 OptimalBinning 学习每列的最优分箱 + WoE 映射
2. transform 阶段用学到的映射编码新数据
3. 自动处理缺失值（OptimalBinning 内置支持）
4. 暴露分箱明细供 ScoreCard 系数表使用
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from optbinning import OptimalBinning
    HAS_OPTBINNING = True
except ImportError:
    HAS_OPTBINNING = False

logger = logging.getLogger(__name__)

@dataclass
class BinningResult:
    """单特征的分箱结果。"""
    feature: str
    iv: float
    n_bins: int
    woe_values: List[float]       # 每个 bin 的 WoE
    bin_edges: List[str]          # 分箱区间描述
    is_fitted: bool = True

class WoEEncoder:
    """WoE 编码器：fit 学习分箱映射，transform 输出 WoE 编码后的 DataFrame。

    Attributes:
        binning_results: 每个特征的分箱结果
        fitted_features: 已成功拟合的特征列表
        iv_table: 各特征 IV 值汇总
    """

    def __init__(
        self,
        max_n_bins: int = 8,
        min_bin_size: float = 0.05,
        iv_threshold: float = 0.02,
        selection_mode: str = "all",
    ) -> None:
        """初始化 WoE 编码器。

        Args:
            max_n_bins: 最大分箱数
            min_bin_size: 最小分箱比例
            iv_threshold: IV 筛选阈值（低于此值的特征可选择性排除）
            selection_mode: "all" 保留所有特征 | "iv_filter" 按 iv_threshold 筛选
        """
        if not HAS_OPTBINNING:
            raise ImportError(
                "WoEEncoder 依赖 optbinning 库，请安装: pip install optbinning"
            )
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.iv_threshold = iv_threshold
        self.selection_mode = selection_mode

        self._binners: Dict[str, OptimalBinning] = {}
        self.binning_results: Dict[str, BinningResult] = {}
        self.fitted_features: List[str] = []
        self.iv_table: Dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        features: Optional[List[str]] = None,
    ) -> "WoEEncoder":
        """学习各特征的最优分箱与 WoE 映射。

        Args:
            X: 训练集特征 DataFrame
            y: 训练集标签（0/1）
            features: 指定要编码的特征列表；None 则使用 X 的全部列

        Returns:
            self（支持链式调用）
        """
        y_arr = np.array(y)
        features = features or list(X.columns)

        fitted = []
        skipped = []

        for feat in features:
            if feat not in X.columns:
                logger.warning(f"[WoE] 特征 '{feat}' 不在 DataFrame 中，跳过")
                skipped.append(feat)
                continue

            x_col = X[feat].values

            try:
                optb = OptimalBinning(
                    name=feat,
                    dtype="numerical",
                    max_n_bins=self.max_n_bins,
                    min_bin_size=self.min_bin_size,
                )
                optb.fit(x_col, y_arr)

                # 获取 IV
                binning_table = optb.binning_table
                table_df = binning_table.build()
                iv_total = float(table_df["IV"].sum())

                # IV 筛选
                if self.selection_mode == "iv_filter" and iv_total < self.iv_threshold:
                    skipped.append(feat)
                    continue

                self._binners[feat] = optb
                self.iv_table[feat] = round(iv_total, 6)

                # 构建 BinningResult
                woe_values = table_df["WoE"].tolist()
                bin_edges = table_df["Bin"].astype(str).tolist()
                self.binning_results[feat] = BinningResult(
                    feature=feat,
                    iv=iv_total,
                    n_bins=len(table_df) - 2,  # 减去 Missing 和 Special 行
                    woe_values=woe_values,
                    bin_edges=bin_edges,
                )
                fitted.append(feat)

            except Exception as e:
                logger.warning(f"[WoE] 特征 '{feat}' 分箱失败: {e}，跳过")
                skipped.append(feat)

        self.fitted_features = fitted
        self._is_fitted = True

        logger.info(
            f"[WoE] fit 完成: {len(fitted)} 个特征成功, "
            f"{len(skipped)} 个跳过"
        )
        if fitted:
            iv_sorted = sorted(self.iv_table.items(), key=lambda x: x[1], reverse=True)
            top5 = iv_sorted[:5]
            logger.info(f"[WoE] Top5 IV: {top5}")

        return self

    def transform(
        self,
        X: pd.DataFrame,
        metric: str = "woe",
    ) -> pd.DataFrame:
        """对数据进行 WoE 编码。

        Args:
            X: 待编码的 DataFrame
            metric: 编码指标，"woe" 或 "event_rate"

        Returns:
            WoE 编码后的 DataFrame（列名不变）
        """
        if not self._is_fitted:
            raise RuntimeError("WoEEncoder 未 fit，请先调用 fit()")

        result = pd.DataFrame(index=X.index)

        for feat in self.fitted_features:
            if feat not in X.columns:
                logger.warning(f"[WoE] transform 时特征 '{feat}' 不存在，填充 0")
                result[feat] = 0.0
                continue
            optb = self._binners[feat]
            result[feat] = optb.transform(X[feat].values, metric=metric)

        return result

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        features: Optional[List[str]] = None,
        metric: str = "woe",
    ) -> pd.DataFrame:
        """fit + transform 一步完成。"""
        self.fit(X, y, features)
        return self.transform(X, metric=metric)

    def get_iv_summary(self, sort: bool = True) -> pd.DataFrame:
        """返回各特征 IV 汇总表。

        Returns:
            DataFrame with columns [feature, iv, iv_level, n_bins]
        """
        if not self._is_fitted:
            raise RuntimeError("WoEEncoder 未 fit")

        rows = []
        for feat, iv in self.iv_table.items():
            if iv >= 0.3:
                level = "极强"
            elif iv >= 0.1:
                level = "强"
            elif iv >= 0.02:
                level = "中等"
            else:
                level = "弱"
            n_bins = self.binning_results[feat].n_bins if feat in self.binning_results else 0
            rows.append({"feature": feat, "iv": iv, "iv_level": level, "n_bins": n_bins})

        df = pd.DataFrame(rows)
        if sort and len(df) > 0:
            df = df.sort_values("iv", ascending=False).reset_index(drop=True)
        return df

    def get_scorecard_table(self) -> pd.DataFrame:
        """生成评分卡格式的完整分箱-WoE 映射表。

        适用于 LR 评分卡部署：每个特征的每个分箱对应一个 WoE 值，
        乘以 LR 系数即为该箱的分数贡献。

        Returns:
            DataFrame with columns [feature, bin, woe, iv]
        """
        if not self._is_fitted:
            raise RuntimeError("WoEEncoder 未 fit")

        rows = []
        for feat in self.fitted_features:
            optb = self._binners[feat]
            table_df = optb.binning_table.build()
            for _, row in table_df.iterrows():
                rows.append({
                    "feature": feat,
                    "bin": str(row["Bin"]),
                    "count": int(row["Count"]) if pd.notna(row["Count"]) else 0,
                    "event_rate": round(float(row["Event rate"]), 6) if pd.notna(row["Event rate"]) else 0,
                    "woe": round(float(row["WoE"]), 6) if pd.notna(row["WoE"]) else 0,
                    "iv": round(float(row["IV"]), 6) if pd.notna(row["IV"]) else 0,
                })

        return pd.DataFrame(rows)

__all__ = ["WoEEncoder", "BinningResult"]
