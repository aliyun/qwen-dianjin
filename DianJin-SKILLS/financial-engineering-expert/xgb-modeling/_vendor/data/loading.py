# -*- coding: utf-8 -*-
"""数据加载与转换工具。

职责：读取数据文件、推断特征、DataFrame → (X, y) 转换。
不涉及切分逻辑（见 splitting.py）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Iterable, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def read_dataset(
    path: str,
    label_filter: Optional[Iterable] = (0, 1),
    target: Optional[str] = None,
) -> pd.DataFrame:
    """仅 load 不切分；供 feature-analysis / univariate-analysis 等复用。

    Args:
        path: parquet 或 csv 路径
        label_filter: 仅保留 target 在该集合内的样本；为 None 时跳过
        target: 标签列名；label_filter 非 None 时必须给出
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"不支持的文件格式: {p.suffix}（仅支持 .parquet / .csv）")

    if label_filter is not None:
        if target is None:
            raise ValueError("label_filter 非 None 时必须传入 target 列名")
        if target not in df.columns:
            raise ValueError(f"目标变量列不存在: {target}")
        df = df[df[target].isin(list(label_filter))].reset_index(drop=True)

    return df

def infer_features(
    df: pd.DataFrame,
    target: str,
    exclude_cols: Optional[List[str]] = None,
    numeric_only: bool = True,
) -> List[str]:
    """统一特征推断。"""
    exclude_set = set(exclude_cols or []) | {target}
    if numeric_only:
        return [
            c for c in df.columns
            if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
        ]
    return [c for c in df.columns if c not in exclude_set]

def to_xy(
    df: pd.DataFrame,
    features: List[str],
    target: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """从 DataFrame 派生 (X, y)。"""
    return df[features], df[target].values

def to_indices(sub_df: pd.DataFrame, full_df: pd.DataFrame) -> np.ndarray:
    """获取 sub_df 在 full_df 中的整数位置索引。

    依赖 sub_df 保留了 full_df 的原始 index（load_and_split_data 已保证）。
    """
    pos = full_df.index.get_indexer(sub_df.index)
    if (pos < 0).any():
        raise ValueError("sub_df 含不在 full_df 中的索引，无法派生位置索引")
    return pos

def build_xgb_eval_set(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: List[str],
    target: str,
) -> List[Tuple[pd.DataFrame, np.ndarray]]:
    """统一构造 XGBoost 的 eval_set=[(X_train,y_train),(X_val,y_val)]。

    硬约束：val 必须非空。任何把 oot 传进 val 位置的尝试都会被拒绝。
    """
    if val is None or len(val) == 0:
        raise ValueError(
            "经典三段式范式要求 val 非空：early stopping 必须用 val。"
            "请检查 val_ratio 或数据切分结果。"
        )
    return [
        (train[features], train[target].values),
        (val[features], val[target].values),
    ]
