## -*- coding: utf-8 -*-
"""兼容门面：实际实现已迁移至 shared.data.* 子包。

此文件保留所有原公开接口的重导出，确保现有 `from data_utils import xxx`
的调用方无需任何改动。新代码请直接 import shared.data.*。
"""
from data.loading import (  # noqa: F401,F403
    read_dataset,
    infer_features,
    to_xy,
    to_indices,
    build_xgb_eval_set,
)
from data.splitting import (  # noqa: F401,F403
    SplitResult,
    load_and_split_data,
    render_split_section,
    DEFAULT_OOT_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_RANDOM_SEED,
    # 私有函数也需重导出，因 test_data_utils.py 可能引用
    _validate_ratios,
    _split_explicit,
    _split_by_time,
    _split_random,
    _split_train_val,
    _validate_splits,
    _build_meta,
    _pos_rate,
)
from data.missing import (  # noqa: F401,F403
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
