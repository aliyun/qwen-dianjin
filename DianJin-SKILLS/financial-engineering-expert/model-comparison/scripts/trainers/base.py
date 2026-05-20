# -*- coding: utf-8 -*-
"""训练器契约 — TrainerBase 抽象协议 + TrainResult 数据类。

设计原则：
  - TrainResult 是各模块之间唯一的"通用货币"，新增算法只需实现 TrainerBase 并返回它
  - 三段式概率全部保留，供下游叠加曲线、一致性分析使用
  - feature_importance 统一为 [(name, score)] 降序列表（跨算法可对齐）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 基础评估指标（train/val/oot 各一份）─────────────────────────────
@dataclass
class EvalMetrics:
    auc: float = 0.0
    ks: float = 0.0
    gini: float = 0.0
    # 高阶指标（analysis/metrics.py 补全）
    brier: float = 0.0
    logloss: float = 0.0
    bcr_top10: float = 0.0       # Bad Capture Rate @ top 10%
    precision_top10: float = 0.0
    recall_top10: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {k: round(getattr(self, k), 6) for k in self.__dataclass_fields__}

# ── 训练产物（核心契约）─────────────────────────────────────────────
@dataclass
class TrainResult:
    # 元信息
    algorithm: str                      # "XGBoost" / "LR + WoE" / "DNN (MLP)"
    short: str                          # "xgb" / "lr" / "dnn"
    interpretability: str               # "强" / "中" / "弱"
    n_features: int                     # 实际入模特征数

    # 三段式评估
    train: EvalMetrics = field(default_factory=EvalMetrics)
    val: EvalMetrics = field(default_factory=EvalMetrics)
    oot: EvalMetrics = field(default_factory=EvalMetrics)

    # 原始概率（下游分析/曲线使用）
    prob_train: Optional[np.ndarray] = None
    prob_val: Optional[np.ndarray] = None
    prob_oot: Optional[np.ndarray] = None

    # 模型产物（持久化用）
    model_artifact: Any = None
    # 特征重要性（降序）；用于跨算法特征重叠分析
    feature_importance: Optional[List[Tuple[str, float]]] = None

    # 稳定性分析结果（analysis/stability.py 填充）
    psi: float = 0.0                    # OOT vs Train 分数 PSI
    monthly_ks: Optional[List[Dict[str, Any]]] = None  # [{month, ks, auc}]

    # 保留扩展位
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── 便捷属性 ──
    @property
    def ks_gap(self) -> float:
        """Train KS - OOT KS，衡量过拟合。"""
        return round(self.train.ks - self.oot.ks, 6)

    def summary_row(self) -> Dict[str, Any]:
        """用于对比总表的一行摘要。"""
        return {
            "algorithm": self.algorithm,
            "short": self.short,
            "n_features": self.n_features,
            "oot_auc": self.oot.auc,
            "oot_ks": self.oot.ks,
            "oot_gini": self.oot.gini,
            "oot_bcr10": self.oot.bcr_top10,
            "oot_brier": self.oot.brier,
            "ks_gap": self.ks_gap,
            "psi": self.psi,
            "interpretability": self.interpretability,
        }

# ── 训练器协议 ────────────────────────────────────────────────────
class TrainerBase(ABC):
    """算法训练器抽象基类。

    子类只需实现 train()，其他公共能力（指标补全、重要性归一化）由子类在 train()
    最后 return 之前调用 helper 完成，或由主入口统一调用 analysis/metrics.py 补全。
    """

    #: 算法短名（xgb/lr/dnn）
    short: str = ""
    #: 人类可读名
    algorithm: str = ""
    #: 可解释性标签
    interpretability: str = ""

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        oot_data: pd.DataFrame,
        features: List[str],
        target: str,
        config: Dict[str, Any],
    ) -> TrainResult:
        """训练并产出 TrainResult。失败时抛异常由主入口捕获降级。"""
        ...

__all__ = ["EvalMetrics", "TrainResult", "TrainerBase"]
