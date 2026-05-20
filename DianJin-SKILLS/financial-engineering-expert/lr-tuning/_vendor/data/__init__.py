# -*- coding: utf-8 -*-
"""shared.data — 数据加载与切分子包。

按关注点从 data_utils.py 分桶：
  - loading: 数据读取、特征推断、X/y 转换
  - splitting: 经典三段式切分（train/val/oot）
"""
from .loading import (
    read_dataset,
    infer_features,
    to_xy,
    to_indices,
    build_xgb_eval_set,
)
from .splitting import (
    SplitResult,
    load_and_split_data,
    render_split_section,
    DEFAULT_OOT_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_RANDOM_SEED,
)
from .missing import (
    DNNImputer,
    MissingSummary,
    DEFAULT_INDICATOR_LOW,
    DEFAULT_INDICATOR_HIGH,
)

__all__ = [
    "read_dataset",
    "infer_features",
    "to_xy",
    "to_indices",
    "build_xgb_eval_set",
    "SplitResult",
    "load_and_split_data",
    "render_split_section",
    "DEFAULT_OOT_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_RANDOM_SEED",
    "DNNImputer",
    "MissingSummary",
    "DEFAULT_INDICATOR_LOW",
    "DEFAULT_INDICATOR_HIGH",
]
