# -*- coding: utf-8 -*-
"""加载各算法的"最优参数"供多模型对比使用。

查找优先级（从高到低）：
  1. 显式 --tuned_params_file 传入的 JSON：
     {"xgb": {...}, "lr": {...}, "dnn": {...}}
  2. 模型目录下最新的 tuning meta：
     - XGB: *_meta.json 里的 params
     - LR:  *_tuning_best_*_meta.json 里的 best_params
     - DNN: *_tuning_best_*_meta.json 里的 best_params
  3. 找不到 → 返回空 dict（各 trainer 走默认）

设计：纯函数、无副作用，返回 `{algo_short: params_dict}`。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── 扫描规则 ─────────────────────────────────────────────────
_SCAN_PATTERNS = {
    "xgb": ["xgb_tuning_best_*_meta.json", "*xgb*tuned*_meta.json"],
    "lr": ["lr_tuning_best_*_meta.json"],
    "dnn": ["dnn_tuning_best_*_meta.json"],
}

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("读取 %s 失败: %s", path, e)
        return None

def _extract_params(meta: Dict[str, Any], algo: str) -> Optional[Dict[str, Any]]:
    """不同算法的 meta 结构可能不同，统一优先 best_params → params → tuning.best_params。"""
    if not isinstance(meta, dict):
        return None
    for key in ("best_params", "params"):
        if isinstance(meta.get(key), dict):
            return meta[key]
    # 嵌套：xgb_tuning/lr_tuning/dnn_tuning
    sub = meta.get(f"{algo}_tuning") or meta.get("tuning")
    if isinstance(sub, dict):
        for key in ("best_params", "params"):
            if isinstance(sub.get(key), dict):
                return sub[key]
    return None

def _scan_latest(models_dir: Path, algo: str) -> Optional[Dict[str, Any]]:
    """扫描 models_dir 下最新的 tuning meta 文件。"""
    candidates = []
    for pattern in _SCAN_PATTERNS.get(algo, []):
        candidates.extend(models_dir.glob(pattern))
    if not candidates:
        return None
    # 按 mtime 降序，取最新
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        meta = _load_json(path)
        params = _extract_params(meta or {}, algo)
        if params:
            logger.info("[%s] 发现调参最优参数: %s", algo.upper(), path.name)
            return params
    return None

def load_tuned_params(
    models_dir: Optional[Path] = None,
    tuned_params_file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """按优先级加载三类算法的最优参数。

    Args:
        models_dir: 模型目录（如 data/models），用于自动扫描
        tuned_params_file: 显式 JSON 路径
    Returns:
        {"xgb": {...}, "lr": {...}, "dnn": {...}}  空值表示未找到
    """
    result: Dict[str, Dict[str, Any]] = {"xgb": {}, "lr": {}, "dnn": {}}

    # 1. 显式文件 —— 最高优先级
    if tuned_params_file:
        explicit = _load_json(Path(tuned_params_file))
        if isinstance(explicit, dict):
            for algo in result:
                if isinstance(explicit.get(algo), dict):
                    result[algo] = explicit[algo]
                    logger.info("[%s] 使用显式参数（来自 %s）", algo.upper(), tuned_params_file)

    # 2. 模型目录扫描 —— 仅填补上一步未覆盖的算法
    if models_dir and models_dir.exists():
        for algo in result:
            if not result[algo]:
                params = _scan_latest(models_dir, algo)
                if params:
                    result[algo] = params

    return result
