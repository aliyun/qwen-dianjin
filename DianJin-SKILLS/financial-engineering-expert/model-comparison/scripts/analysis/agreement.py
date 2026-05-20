# -*- coding: utf-8 -*-
"""样本级一致性分析 — 两两算法打分是否对齐。

关注三个维度：
  1. Top-K% 集合重叠（Jaccard）—— 关注"挑的坏人是否一致"
  2. Spearman 秩相关    —— 关注"整体排序是否一致"
  3. 分歧样本数          —— 两算法判断相反的样本数量（>median vs <median）
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

TOP_RATIO = 0.10  # Top 10% 对齐度

def _top_k_indices(prob: np.ndarray, ratio: float) -> set:
    k = max(int(len(prob) * ratio), 1)
    return set(np.argsort(-prob)[:k].tolist())

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return round(len(a & b) / max(len(a | b), 1), 6)

def _disagreement_count(p_a: np.ndarray, p_b: np.ndarray) -> int:
    """二值化后标签分歧数 —— 用各自中位数做阈值，避免分数绝对值差异。"""
    tag_a = (p_a >= np.median(p_a)).astype(int)
    tag_b = (p_b >= np.median(p_b)).astype(int)
    return int((tag_a != tag_b).sum())

def pairwise_agreement(prob_map: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """对 prob_map 中所有算法两两计算一致性指标。

    Returns:
        {(a, b): {top10_jaccard, spearman_rho, spearman_p, disagreement}}
    """
    keys = list(prob_map.keys())
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if len(keys) < 2:
        return out

    # 预计算每个算法的 top-k 集合
    top_sets = {k: _top_k_indices(prob_map[k], TOP_RATIO) for k in keys}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            p_a, p_b = prob_map[a], prob_map[b]
            rho, pval = spearmanr(p_a, p_b)
            out[(a, b)] = {
                "top10_jaccard": _jaccard(top_sets[a], top_sets[b]),
                "spearman_rho": round(float(rho) if rho == rho else 0.0, 6),  # NaN-safe
                "spearman_p": round(float(pval) if pval == pval else 1.0, 6),
                "disagreement": _disagreement_count(p_a, p_b),
                "n_samples": int(len(p_a)),
            }
    return out
