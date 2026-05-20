# -*- coding: utf-8 -*-
"""ModelCard & WarmStartBundle — 模型知识闭环。

ModelCard 是训练产出的完整元数据快照，包含：
  - 数据指纹（可比对不同训练的数据来源是否一致）
  - 特征清单 + 四大算子指标（IV / PSI / SHAP / 缺失率）
  - 超参数 + Optuna 搜索轨迹
  - 评估结果（train/val/oot 三段）
  - 诊断报告（DiagnosticReport）
  - KS 切点历史（追踪切点随时间漂移）
  - Champion/Challenger 标记

WarmStartBundle 供 experience-advisor → xgb-tuning 传递历史先验。

使用示例：
    from model_card import ModelCard, DataFingerprint
    card = ModelCard.create(model_id="baseline_v3", ...)
    card.to_dict()  # JSON-safe dict，可直接存 SQLite
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 辅助数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataFingerprint:
    """数据指纹 — 标识训练数据来源。

    两个 DataFingerprint 可通过 field_hash 判断是否来自同一数据集。
    """
    n_samples: int                  # 总样本量
    n_features: int                 # 特征数
    pos_rate: float                 # 正样本率
    time_span: Optional[str] = None # 时间跨度（如 "2024-01-01 ~ 2024-06-30"）
    field_hash: str = ""            # 字段名排序后的 MD5 哈希

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_col: str = "label",
        time_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> DataFingerprint:
        """从 DataFrame 构造指纹。"""
        cols = list(feature_cols) if feature_cols else [
            c for c in df.columns if c not in (target_col, time_col)
        ]
        n_features = len(cols)
        n_samples = len(df)
        pos_rate = float(df[target_col].mean()) if target_col in df.columns else 0.0

        time_span = None
        if time_col and time_col in df.columns:
            ts = pd.to_datetime(df[time_col])
            time_span = f"{ts.min().date()} ~ {ts.max().date()}"

        col_str = ",".join(sorted(cols))
        field_hash = hashlib.md5(col_str.encode()).hexdigest()[:12]

        return cls(
            n_samples=n_samples,
            n_features=n_features,
            pos_rate=round(pos_rate, 6),
            time_span=time_span,
            field_hash=field_hash,
        )

@dataclass(frozen=True)
class FeatureSpec:
    """单个特征的元数据。"""
    name: str
    dtype: str = "float64"
    iv: Optional[float] = None          # Information Value
    psi: Optional[float] = None         # Population Stability Index
    shap_importance: Optional[float] = None
    missing_rate: Optional[float] = None
    description: str = ""

@dataclass(frozen=True)
class TrialRecord:
    """单次 Optuna trial 记录。"""
    trial_id: int
    params: Dict[str, Any]
    score: float                        # 主指标值
    metric: str = "ks"
    gap: Optional[float] = None
    duration_sec: Optional[float] = None

@dataclass(frozen=True)
class CutPoint:
    """KS 切点快照 — 追踪切点随时间的漂移。"""
    dataset: str                        # "train" | "val" | "oot"
    peak_quantile: float                # KS 峰值所处分位数
    ks_value: float                     # 该切点的 KS 值
    timestamp: str = ""                 # 记录时间

# ---------------------------------------------------------------------------
# ModelCard 主体
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """训练产出的完整元数据快照。

    Attributes:
        model_id: 模型唯一标识（如 "baseline_v3"）
        created_at: 创建时间
        data_fingerprint: 数据指纹
        scene_tags: 场景标签（如 ["credit_scoring", "small_sample"]）
        features: 特征清单 + 四大算子指标
        params: 最终超参数
        eval_results: 评估结果 {"train": {...}, "val": {...}, "oot": {...}}
        diagnostic: 诊断报告序列化 dict（来自 DiagnosticReport.to_dict()）
        search_trajectory: Optuna trial 全量记录
        ks_cut_history: KS 切点历史
        champion: 是否为 champion 模型
        notes: 附加备注
    """
    model_id: str
    created_at: str = ""
    data_fingerprint: Optional[DataFingerprint] = None
    scene_tags: List[str] = field(default_factory=list)
    features: List[FeatureSpec] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    eval_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagnostic: Optional[Dict[str, Any]] = None
    search_trajectory: List[TrialRecord] = field(default_factory=list)
    ks_cut_history: List[CutPoint] = field(default_factory=list)
    champion: bool = False
    notes: str = ""

    @classmethod
    def create(
        cls,
        model_id: str,
        params: Dict[str, Any],
        eval_results: Dict[str, Dict[str, Any]],
        *,
        data_fingerprint: Optional[DataFingerprint] = None,
        diagnostic: Optional[Dict[str, Any]] = None,
        features: Optional[List[FeatureSpec]] = None,
        search_trajectory: Optional[List[TrialRecord]] = None,
        ks_cut_history: Optional[List[CutPoint]] = None,
        scene_tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> ModelCard:
        """工厂方法 — 自动填充 created_at。"""
        return cls(
            model_id=model_id,
            created_at=datetime.now().isoformat(),
            data_fingerprint=data_fingerprint,
            scene_tags=scene_tags or [],
            features=features or [],
            params=params,
            eval_results=eval_results,
            diagnostic=diagnostic,
            search_trajectory=search_trajectory or [],
            ks_cut_history=ks_cut_history or [],
            notes=notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe 序列化。"""
        d: Dict[str, Any] = {
            "model_id": self.model_id,
            "created_at": self.created_at,
            "champion": self.champion,
            "notes": self.notes,
            "scene_tags": self.scene_tags,
            "params": self.params,
            "eval_results": self.eval_results,
        }
        if self.data_fingerprint is not None:
            d["data_fingerprint"] = asdict(self.data_fingerprint)
        if self.features:
            d["features"] = [asdict(f) for f in self.features]
        if self.diagnostic is not None:
            d["diagnostic"] = self.diagnostic
        if self.search_trajectory:
            d["search_trajectory"] = [asdict(t) for t in self.search_trajectory]
        if self.ks_cut_history:
            d["ks_cut_history"] = [asdict(c) for c in self.ks_cut_history]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModelCard:
        """从 dict 反序列化。"""
        fp = None
        if "data_fingerprint" in d and d["data_fingerprint"]:
            fp = DataFingerprint(**d["data_fingerprint"])

        features = []
        for f in d.get("features", []):
            features.append(FeatureSpec(**f))

        trajectory = []
        for t in d.get("search_trajectory", []):
            trajectory.append(TrialRecord(**t))

        cuts = []
        for c in d.get("ks_cut_history", []):
            cuts.append(CutPoint(**c))

        return cls(
            model_id=d["model_id"],
            created_at=d.get("created_at", ""),
            data_fingerprint=fp,
            scene_tags=d.get("scene_tags", []),
            features=features,
            params=d.get("params", {}),
            eval_results=d.get("eval_results", {}),
            diagnostic=d.get("diagnostic"),
            search_trajectory=trajectory,
            ks_cut_history=cuts,
            champion=d.get("champion", False),
            notes=d.get("notes", ""),
        )

    def to_json(self) -> str:
        """JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

# ---------------------------------------------------------------------------
# WarmStartBundle — 历史先验注入
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WarmStartBundle:
    """experience-advisor → xgb-tuning 的历史先验包。

    Attributes:
        baseline_params: 推荐基线参数
        enqueue_trials: 从历史 trial 中筛选的 Top N 参数集合，
            xgb-tuning 启动时自动 optuna.enqueue_trial 注入
        param_bounds_override: 参数搜索空间覆写（如历史经验发现
            某些参数范围更优）
        expected_cut_quantile_range: 历史 KS 切点分位的期望范围
        source_model_id: 来源模型 ID
        confidence: 推荐置信度（0~1）
    """
    baseline_params: Dict[str, Any] = field(default_factory=dict)
    enqueue_trials: List[Dict[str, Any]] = field(default_factory=list)
    param_bounds_override: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    expected_cut_quantile_range: Tuple[float, float] = (0.05, 0.40)
    source_model_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_params": self.baseline_params,
            "enqueue_trials": self.enqueue_trials,
            "param_bounds_override": {
                k: list(v) for k, v in self.param_bounds_override.items()
            },
            "expected_cut_quantile_range": list(self.expected_cut_quantile_range),
            "source_model_id": self.source_model_id,
            "confidence": self.confidence,
        }

__all__ = [
    "DataFingerprint",
    "FeatureSpec",
    "TrialRecord",
    "CutPoint",
    "ModelCard",
    "WarmStartBundle",
]
