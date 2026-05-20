# -*- coding: utf-8 -*-
"""shared.data.missing — DNN 缺失值处理工具。

遵循 DNN 缺失值处理规范：
  - 中位数填充（使用训练集中位数，避免数据泄漏）
  - 中等缺失率（默认 5% ~ 80%）自动添加缺失指示列
  - 极低缺失率（<5%）仅填充，不加指示列（避免常量列）
  - 极高缺失率（>80%）仅填充，不加指示列（也建议上游整列删除）
  - 输出缺失率摘要，便于报告记录
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 默认阈值 ─────────────────────────────────────────────────────────
DEFAULT_INDICATOR_LOW = 0.05   # 缺失率下界：低于不加指示列
DEFAULT_INDICATOR_HIGH = 0.80  # 缺失率上界：高于不加指示列

@dataclass
class MissingSummary:
    """单特征缺失处理摘要。"""
    feature: str
    miss_rate: float
    fill_value: float
    has_indicator: bool

@dataclass
class DNNImputer:
    """DNN 缺失值处理器。

    使用方式::

        imputer = DNNImputer(features=features)
        X_train_filled, feat_out = imputer.fit_transform(train_data)
        X_val_filled, _ = imputer.transform(val_data)
        X_oot_filled, _ = imputer.transform(oot_data)

    说明：
      - fit_transform 返回填充后的 DataFrame 和扩展后的特征列表（含 _missing 列）
      - transform 使用 fit 阶段已确定的 median 与指示列集合，保证三段数据一致
      - summaries 字段记录每个原始特征的处理信息，便于报告落盘
    """

    features: List[str]
    indicator_low: float = DEFAULT_INDICATOR_LOW
    indicator_high: float = DEFAULT_INDICATOR_HIGH

    # fit 阶段产出（运行时填充）
    medians_: Dict[str, float] = field(default_factory=dict)
    indicator_cols_: List[str] = field(default_factory=list)  # 需要加指示列的原始特征
    output_features_: List[str] = field(default_factory=list)  # 最终输出列（原特征 + *_missing）
    summaries_: List[MissingSummary] = field(default_factory=list)
    fitted_: bool = False

    # ── 公共接口 ─────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "DNNImputer":
        """基于训练集计算中位数并决定指示列。"""
        self.medians_.clear()
        self.indicator_cols_.clear()
        self.summaries_.clear()

        for col in self.features:
            series = df[col]
            miss_rate = float(series.isna().mean())
            median = float(series.median()) if series.notna().any() else 0.0
            # NaN 的 median 可能仍为 NaN（全空列）
            if not np.isfinite(median):
                median = 0.0
            self.medians_[col] = median

            has_indicator = self.indicator_low <= miss_rate <= self.indicator_high
            if has_indicator:
                self.indicator_cols_.append(col)

            self.summaries_.append(
                MissingSummary(
                    feature=col,
                    miss_rate=miss_rate,
                    fill_value=median,
                    has_indicator=has_indicator,
                )
            )

        # 输出特征：原始特征 + 指示列（顺序固定）
        self.output_features_ = list(self.features) + [
            f"{c}_missing" for c in self.indicator_cols_
        ]
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """对单段数据做填充 + 指示列生成。

        返回: (扩展后的 DataFrame[output_features_], output_features_)
        """
        if not self.fitted_:
            raise RuntimeError("DNNImputer 未 fit，请先调用 fit 或 fit_transform。")

        # 先构造指示列（在填充之前，以免 NaN 被覆盖）
        indicator_frame = {}
        for col in self.indicator_cols_:
            indicator_frame[f"{col}_missing"] = df[col].isna().astype(np.int8).values

        # 填充原始特征
        filled = df[self.features].copy()
        for col in self.features:
            filled[col] = filled[col].fillna(self.medians_[col])

        # 合并指示列
        if indicator_frame:
            indicator_df = pd.DataFrame(indicator_frame, index=filled.index)
            filled = pd.concat([filled, indicator_df], axis=1)

        return filled[self.output_features_], list(self.output_features_)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        self.fit(df)
        return self.transform(df)

    # ── 报告辅助 ─────────────────────────────────────────────────────
    def summary_table(self, top_n: Optional[int] = None) -> List[Dict]:
        """导出摘要（按缺失率降序），用于报告/模型卡。"""
        rows = [
            {
                "feature": s.feature,
                "miss_rate": round(s.miss_rate, 4),
                "fill_value": round(s.fill_value, 6),
                "has_indicator": s.has_indicator,
            }
            for s in sorted(self.summaries_, key=lambda x: -x.miss_rate)
        ]
        if top_n is not None:
            rows = rows[:top_n]
        return rows

    def overview(self) -> Dict:
        """全局概览：用于模型卡/报告头部。"""
        total = len(self.summaries_)
        n_with_miss = sum(1 for s in self.summaries_ if s.miss_rate > 0)
        n_indicator = len(self.indicator_cols_)
        max_miss = max((s.miss_rate for s in self.summaries_), default=0.0)
        return {
            "total_features": total,
            "features_with_missing": n_with_miss,
            "indicator_cols_added": n_indicator,
            "max_miss_rate": round(max_miss, 4),
            "indicator_low": self.indicator_low,
            "indicator_high": self.indicator_high,
            "strategy": "median_fill + missing_indicator",
        }

__all__ = [
    "DNNImputer",
    "MissingSummary",
    "DEFAULT_INDICATOR_LOW",
    "DEFAULT_INDICATOR_HIGH",
]
