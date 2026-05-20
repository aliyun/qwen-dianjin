# -*- coding: utf-8 -*-
"""高阶评估指标补全。

基础 AUC/KS/Gini 已在 xgb_core.ModelEvaluator 内完成；此处补：
  - Brier score（概率校准程度）
  - logloss（分类损失）
  - Precision@top10% / Recall@top10% / BCR@top10%（业务落地口径）

这些指标对三类算法同等适用（只依赖 y_true + prob），单独抽出便于复用与单测。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from trainers.base import EvalMetrics, TrainResult

TOP_RATIO = 0.10  # 固定 10% 分位口径（业务常用）

def _top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray, ratio: float) -> Tuple[float, float, float]:
    """返回 (bcr, precision, recall) at top-ratio 分位。

    - BCR (Bad Capture Rate) = 前 k% 打分最高人群中坏人数 / 总坏人数
    - Precision = 前 k% 人群中坏人占比
    - Recall    = BCR（在正负样本定义下两者等价），这里独立给出是为了文案清晰
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    k = max(int(n * ratio), 1)

    # 按分数降序取 top-k
    order = np.argsort(-y_prob)
    top_idx = order[:k]
    y_top = y_true[top_idx]

    total_bad = float(y_true.sum())
    top_bad = float(y_top.sum())

    bcr = top_bad / total_bad if total_bad > 0 else 0.0
    precision = top_bad / k
    recall = bcr  # 正负样本口径一致
    return round(bcr, 6), round(precision, 6), round(recall, 6)

def _fill_one(m: EvalMetrics, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """原地补全一份 EvalMetrics 的高阶指标。"""
    if y_prob is None or len(y_prob) == 0:
        return
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-7, 1 - 1e-7)

    try:
        m.brier = round(float(brier_score_loss(y_true, y_prob)), 6)
    except Exception:
        m.brier = 0.0
    try:
        m.logloss = round(float(log_loss(y_true, y_prob)), 6)
    except Exception:
        m.logloss = 0.0

    bcr, prec, rec = _top_k_metrics(y_true, y_prob, TOP_RATIO)
    m.bcr_top10 = bcr
    m.precision_top10 = prec
    m.recall_top10 = rec

def enrich_metrics(
    result: TrainResult,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_oot: np.ndarray,
) -> TrainResult:
    """在 result 上原地补全 train/val/oot 三段式的高阶指标。返回同对象方便链式。"""
    _fill_one(result.train, y_train, result.prob_train)
    _fill_one(result.val, y_val, result.prob_val)
    _fill_one(result.oot, y_oot, result.prob_oot)
    return result
