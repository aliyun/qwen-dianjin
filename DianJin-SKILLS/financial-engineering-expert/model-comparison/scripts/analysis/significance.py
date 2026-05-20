# -*- coding: utf-8 -*-
"""DeLong 检验 — 两两算法 AUC 差异是否统计显著。

实现参考 Xu Sun & Weichao Xu (2014) 的 fast-DeLong 算法，O(N log N)。
避免在 N=10w+ 时 Bootstrap 过慢。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Fast midrank —— 对相同值赋予平均秩。"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based midrank
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(y_true: np.ndarray, probs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (aucs, cov)：K 个 AUC 以及 K×K 协方差矩阵。"""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    label = y_true[order]
    m = int(label.sum())  # positives
    n = len(label) - m    # negatives
    if m == 0 or n == 0:
        k = len(probs)
        return np.zeros(k), np.zeros((k, k))

    K = len(probs)
    pos = np.zeros((K, m))
    neg = np.zeros((K, n))
    for i, p in enumerate(probs):
        p_sorted = np.asarray(p)[order]
        pos[i] = p_sorted[:m]
        neg[i] = p_sorted[m:]

    tx = np.empty((K, m))
    ty = np.empty((K, n))
    tz = np.empty((K, m + n))
    for i in range(K):
        tx[i] = _compute_midrank(pos[i])
        ty[i] = _compute_midrank(neg[i])
        tz[i] = _compute_midrank(np.concatenate([pos[i], neg[i]]))

    aucs = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:  # K==1
        sx = sx.reshape(1, 1)
        sy = sy.reshape(1, 1)
    cov = sx / m + sy / n
    return aucs, cov

def pairwise_delong(
    y_true: np.ndarray,
    prob_map: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """对 prob_map 中所有算法两两做 DeLong 检验。

    Args:
        y_true: 真实标签（通常用 OOT）
        prob_map: {algo_short: probability_array}
    Returns:
        {(a, b): {auc_a, auc_b, delta, z, p_value, significant}}
    """
    keys = list(prob_map.keys())
    if len(keys) < 2:
        return {}

    aucs, cov = _fast_delong(y_true, [prob_map[k] for k in keys])
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            delta = aucs[i] - aucs[j]
            var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            if var <= 0:
                z = 0.0
                p = 1.0
            else:
                z = float(delta / np.sqrt(var))
                p = float(2 * (1 - stats.norm.cdf(abs(z))))
            out[(keys[i], keys[j])] = {
                "auc_a": round(float(aucs[i]), 6),
                "auc_b": round(float(aucs[j]), 6),
                "delta": round(float(delta), 6),
                "z": round(z, 4),
                "p_value": round(p, 6),
                "significant": bool(p < 0.05),
            }
    return out
