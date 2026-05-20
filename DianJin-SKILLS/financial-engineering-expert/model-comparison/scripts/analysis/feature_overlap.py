# -*- coding: utf-8 -*-
"""跨算法特征重要性重叠分析。

输入各算法的 feature_importance 列表，输出：
  - Top-N 交集/并集
  - 两两 Jaccard
  - 共识特征（出现在 ≥K 个算法 Top-N 的特征）

DNN 没有 feature_importance 时自动跳过。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from trainers.base import TrainResult

def _top_set(result: TrainResult, top_n: int) -> Optional[set]:
    fi = result.feature_importance
    if not fi:
        return None
    return set(name for name, _ in fi[:top_n])

def feature_importance_overlap(
    results: Dict[str, TrainResult],
    top_n: int = 20,
) -> Dict[str, object]:
    """返回 overlap 分析结果。

    Returns:
        {
            "top_n": int,
            "participants": [algo_short, ...],     # 有 feature_importance 的算法
            "pairwise_jaccard": {(a, b): score},
            "consensus": [(feat, count), ...],     # 共识：至少两算法都上榜，按次数降序
            "union_size": int,
            "intersection_size": int,
        }
    """
    sets: Dict[str, set] = {}
    for k, r in results.items():
        s = _top_set(r, top_n)
        if s is not None:
            sets[k] = s

    if len(sets) < 2:
        return {
            "top_n": top_n,
            "participants": list(sets.keys()),
            "pairwise_jaccard": {},
            "consensus": [],
            "union_size": sum(len(s) for s in sets.values()),
            "intersection_size": 0,
        }

    # 两两 Jaccard
    keys = list(sets.keys())
    pairwise: Dict[Tuple[str, str], float] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = sets[keys[i]], sets[keys[j]]
            jac = len(a & b) / max(len(a | b), 1)
            pairwise[(keys[i], keys[j])] = round(jac, 6)

    # 共识：出现在 ≥2 个算法 Top-N 的特征
    counter: Dict[str, int] = {}
    for s in sets.values():
        for feat in s:
            counter[feat] = counter.get(feat, 0) + 1
    consensus: List[Tuple[str, int]] = sorted(
        [(f, c) for f, c in counter.items() if c >= 2],
        key=lambda x: (-x[1], x[0]),
    )

    union: set = set().union(*sets.values())
    intersection: set = set.intersection(*sets.values()) if sets else set()

    return {
        "top_n": top_n,
        "participants": keys,
        "pairwise_jaccard": pairwise,
        "consensus": consensus,
        "union_size": len(union),
        "intersection_size": len(intersection),
    }
