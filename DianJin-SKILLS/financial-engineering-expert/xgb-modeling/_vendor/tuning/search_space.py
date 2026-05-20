# -*- coding: utf-8 -*-
"""搜索空间构造 — 诊断驱动的约束空间 + 参数差值工具。

职责：
  - PARAM_BOUNDS: 硬边界（算法允许范围）
  - _build_constrained_space: 把 DiagnosisResult 翻译为 {param: (lo, hi)} 约束
  - build_params_delta: 构造参数向量差异，用于 tried_directions
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tuning.diagnose import DiagnosisResult

# ---------------------------------------------------------------------------
# 硬边界（算法学允许范围）
# ---------------------------------------------------------------------------

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "max_depth": (2, 8),
    "learning_rate": (0.005, 0.15),
    "n_estimators": (100, 1500),
    "subsample": (0.5, 0.95),
    "colsample_bytree": (0.5, 0.95),
    "min_child_weight": (10, 500),
    "reg_alpha": (1e-3, 2.0),
    "reg_lambda": (0.1, 10.0),
    "scale_pos_weight": (1.0, 20.0),
}

def _clip(value: float, key: str) -> float:
    lo, hi = PARAM_BOUNDS[key]
    return max(lo, min(hi, value))

def _build_constrained_space(
    diagnosis: DiagnosisResult,
    current_params: Dict[str, Any],
    tried_directions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """把诊断结论翻译为 {param: (lo, hi)} 的约束搜索空间。

    约束策略（替代旧的 ×2/×1.5 启发式）：
      - 过拟合       : max_depth 上限收紧、reg_* 下限抬高、min_child_weight 抬高
      - 轻微过拟合   : 仅 reg_* 温和抬高、max_depth 微收紧
      - 欠拟合       : max_depth/n_estimators 下限抬高、reg_* 放松
      - 拟合良好     : 全参数 ±20% 微调窗口
    同时根据 tried_directions 剔除已无效的调整方向（缩成一个点，即冻结该参数）。
    """
    tried_directions = tried_directions or []
    space: Dict[str, Tuple[float, float]] = {}

    def add(key: str, lo: float, hi: float) -> None:
        lo_c = _clip(lo, key)
        hi_c = _clip(hi, key)
        if hi_c < lo_c:
            lo_c, hi_c = hi_c, lo_c
        space[key] = (lo_c, hi_c)

    md = current_params.get("max_depth", 4)
    lr = current_params.get("learning_rate", 0.05)
    ne = current_params.get("n_estimators", 500)
    ss = current_params.get("subsample", 0.8)
    cs = current_params.get("colsample_bytree", 0.9)
    mcw = current_params.get("min_child_weight", 100)
    ra = current_params.get("reg_alpha", 0.1)
    rl = current_params.get("reg_lambda", 0.5)

    status = diagnosis.status

    if status == "过拟合":
        add("max_depth", max(2, md - 2), md)
        add("reg_alpha", ra, ra * 3)
        add("reg_lambda", rl, rl * 2)
        add("min_child_weight", mcw, mcw * 2)
        add("subsample", ss - 0.1, ss)
        add("colsample_bytree", cs - 0.1, cs)
        add("learning_rate", lr * 0.7, lr)
        add("n_estimators", ne * 0.8, ne)

    elif status == "轻微过拟合":
        add("max_depth", max(2, md - 1), md)
        add("reg_alpha", ra, ra * 2)
        add("reg_lambda", rl, rl * 1.5)
        add("min_child_weight", mcw, mcw * 1.5)
        add("subsample", ss - 0.05, ss)
        add("colsample_bytree", cs - 0.05, cs)
        add("learning_rate", lr * 0.8, lr * 1.1)
        add("n_estimators", ne * 0.9, ne * 1.1)

    elif status == "欠拟合":
        add("max_depth", md, min(8, md + 2))
        add("n_estimators", ne, ne * 1.5)
        add("reg_alpha", ra * 0.3, ra)
        add("reg_lambda", rl * 0.5, rl)
        add("min_child_weight", mcw * 0.5, mcw)
        add("subsample", ss, min(0.95, ss + 0.1))
        add("colsample_bytree", cs, min(0.95, cs + 0.1))
        add("learning_rate", lr, lr * 1.5)

    else:  # 拟合良好 / 收敛 → 微调
        add("max_depth", max(2, md - 1), min(8, md + 1))
        add("reg_alpha", ra * 0.8, ra * 1.2)
        add("reg_lambda", rl * 0.8, rl * 1.2)
        add("min_child_weight", mcw * 0.8, mcw * 1.2)
        add("subsample", ss - 0.05, ss + 0.05)
        add("colsample_bytree", cs - 0.05, cs + 0.05)
        add("learning_rate", lr * 0.8, lr * 1.2)
        add("n_estimators", ne * 0.8, ne * 1.2)

    # 根据 tried_directions 冻结已尝试无效的方向
    for record in tried_directions:
        if record.get("improved"):
            continue
        delta: Dict[str, float] = record.get("params_delta", {}) or {}
        for key, d in delta.items():
            if key not in space:
                continue
            cur = current_params.get(key)
            if cur is None:
                continue
            lo, hi = space[key]
            if d > 0:  # 上次增加未改善 → 禁止继续增加
                space[key] = (lo, min(hi, float(cur)))
            elif d < 0:  # 上次减少未改善 → 禁止继续减少
                space[key] = (max(lo, float(cur)), hi)

    # Task 7-4: n_estimators 固定为上限，由 early_stopping 自动选树数
    if "n_estimators" in space:
        upper = PARAM_BOUNDS["n_estimators"][1]  # 1500
        space["n_estimators"] = (upper, upper)  # 冻结

    # Task 7-5: 强不均衡场景（正样本率 < 5%）纳入 scale_pos_weight
    spw = current_params.get("scale_pos_weight")
    if spw is not None and spw > 1.0:
        add("scale_pos_weight", max(1.0, spw * 0.5), min(20.0, spw * 2.0))

    return space

def _default_batch_space() -> Dict[str, Tuple[float, float]]:
    """run_batch 默认使用全量硬边界作为搜索空间。"""
    return dict(PARAM_BOUNDS)

# ---------------------------------------------------------------------------
# 对外工具函数：参数向量差异
# ---------------------------------------------------------------------------

def build_params_delta(
    prev_params: Dict[str, Any],
    new_params: Dict[str, Any],
) -> Dict[str, float]:
    """构造 {param: new - prev} 的差值字典，仅保留数值型且值有变化的键。

    用于 tried_directions 记录每轮参数调整方向，供 _build_constrained_space
    根据历史效果冻结已尝试无效的方向。

    Example:
        >>> build_params_delta({"max_depth": 4}, {"max_depth": 5})
        {'max_depth': 1}
    """
    delta: Dict[str, float] = {}
    for key, new_val in new_params.items():
        if key not in prev_params:
            continue
        prev_val = prev_params[key]
        if (
            isinstance(new_val, (int, float))
            and isinstance(prev_val, (int, float))
            and new_val != prev_val
        ):
            delta[key] = new_val - prev_val
    return delta

__all__ = [
    "PARAM_BOUNDS",
    "_clip",
    "_build_constrained_space",
    "_default_batch_space",
    "build_params_delta",
]
