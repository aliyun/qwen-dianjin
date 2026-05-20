# -*- coding: utf-8 -*-
"""KS 结构诊断 — 识别隐性过拟合。

通过分析 KS 曲线的统计结构（而非仅看 Gap 数值），发现以下隐性风险：
  1. 切点漂移 (peak_drift) > 5% → 模型在不同数据集上依赖不同人群，泛化不可靠
  2. 曲线陡峭度 (curve_sharpness) 过高 → KS 依赖极少数样本，方差大
  3. KS ↑ 但 AUC ↓ → 排序能力与区分力不一致，需人工确认

核心原理：
  - KS = max|CDF_pos(s) - CDF_neg(s)|，仅取决于概率分布的一个切点
  - AUC 是对所有阈值的积分，衡量全局排序能力
  - 两者不一致时，说明模型区分力集中在局部概率区间
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 切点漂移阈值：train/val/oot 间 peak_quantile 最大差异
PEAK_DRIFT_THRESHOLD = 0.05

# 曲线陡峭度阈值：KS 峰值 ±10 percentile 的 IQR
SHARPNESS_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KSStructureDiagnosis:
    """KS 结构诊断结果。

    Attributes:
        peak_quantile_train: 训练集 KS 峰值所处分位数（0~1，越小 = 模型排分越集中在头部）
        peak_quantile_val: 验证集 KS 峰值分位数
        peak_quantile_oot: OOT KS 峰值分位数（可选，无 OOT 时为 None）
        peak_drift: 最大漂移幅度 = max(|train-val|, |train-oot|, |val-oot|)
        curve_sharpness: 陡峭度指标（峰值 ±10 分位的 KS 增量集中度）
        ks_auc_consistency: KS 排名与 AUC 排名是否一致
        is_hidden_overfit: 综合判定 — 隐性过拟合
        warnings: 诊断过程中发现的问题
    """
    peak_quantile_train: float
    peak_quantile_val: float
    peak_quantile_oot: Optional[float] = None
    peak_drift: float = 0.0
    curve_sharpness: float = 0.0
    ks_auc_consistency: bool = True
    is_hidden_overfit: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_quantile_train": self.peak_quantile_train,
            "peak_quantile_val": self.peak_quantile_val,
            "peak_quantile_oot": self.peak_quantile_oot,
            "peak_drift": self.peak_drift,
            "curve_sharpness": self.curve_sharpness,
            "ks_auc_consistency": self.ks_auc_consistency,
            "is_hidden_overfit": self.is_hidden_overfit,
            "warnings": list(self.warnings),
        }

# ---------------------------------------------------------------------------
# 计算工具
# ---------------------------------------------------------------------------

def _compute_peak_quantile(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """计算 KS 峰值的分位数和陡峭度。

    Returns:
        (peak_quantile, sharpness)
        - peak_quantile: KS 取到最大值时，对应概率阈值在 y_prob 分布中的分位数
        - sharpness: 峰值 ±10 分位范围内 KS 值占整体 KS 的比例
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n = len(y_prob)
    if n == 0:
        return 0.5, 0.0

    # 按预测概率降序排列
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    # 累积分布
    n_pos = y_sorted.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    tpr_curve = np.cumsum(y_sorted) / n_pos
    fpr_curve = np.cumsum(1 - y_sorted) / n_neg
    ks_curve = np.abs(tpr_curve - fpr_curve)

    peak_idx = int(np.argmax(ks_curve))
    peak_quantile = (peak_idx + 1) / n  # Top X% 的样本处取得 KS 峰值

    # 陡峭度：峰值 ±10% 范围内 KS 值占峰值的比例
    ks_max = ks_curve[peak_idx]
    if ks_max < 1e-8:
        return peak_quantile, 0.0

    window = max(1, int(n * 0.10))
    lo_idx = max(0, peak_idx - window)
    hi_idx = min(n - 1, peak_idx + window)
    ks_in_window = ks_curve[lo_idx:hi_idx + 1]
    # 陡峭度 = 窗口内 KS ≥ 80% 峰值的样本比例（越高 = 越尖锐）
    sharpness = float(np.mean(ks_in_window >= 0.8 * ks_max))

    return float(peak_quantile), float(sharpness)

def _check_ks_auc_consistency(
    ks_train: float, ks_val: float,
    auc_train: float, auc_val: float,
) -> bool:
    """检查 KS 和 AUC 的变化方向是否一致。

    如果 KS 从 train→val 上升但 AUC 下降（或反之），说明两个指标矛盾。
    """
    ks_delta = ks_val - ks_train
    auc_delta = auc_val - auc_train

    # 两者同向（都上升或都下降）→ 一致
    # 一者几乎不变（< 0.005）→ 忽略
    if abs(ks_delta) < 0.005 or abs(auc_delta) < 0.005:
        return True
    return (ks_delta > 0) == (auc_delta > 0)

# ---------------------------------------------------------------------------
# 诊断入口
# ---------------------------------------------------------------------------

def run_ks_structure_check(
    y_true_train: np.ndarray,
    y_prob_train: np.ndarray,
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    y_true_oot: Optional[np.ndarray] = None,
    y_prob_oot: Optional[np.ndarray] = None,
    *,
    ks_train: Optional[float] = None,
    ks_val: Optional[float] = None,
    auc_train: Optional[float] = None,
    auc_val: Optional[float] = None,
) -> KSStructureDiagnosis:
    """执行 KS 结构诊断。

    Args:
        y_true_train / y_prob_train: 训练集标签和预测概率
        y_true_val / y_prob_val: 验证集
        y_true_oot / y_prob_oot: OOT（可选）
        ks_train / ks_val / auc_train / auc_val: 预计算的指标值（可选，
            传入可避免重复计算；不传则跳过 KS-AUC 一致性检查）

    Returns:
        KSStructureDiagnosis
    """
    warnings: List[str] = []

    # 1. 计算各数据集的 peak_quantile 和 sharpness
    pq_train, sharp_train = _compute_peak_quantile(y_true_train, y_prob_train)
    pq_val, sharp_val = _compute_peak_quantile(y_true_val, y_prob_val)

    pq_oot: Optional[float] = None
    sharp_oot = 0.0
    if y_true_oot is not None and y_prob_oot is not None:
        pq_oot, sharp_oot = _compute_peak_quantile(y_true_oot, y_prob_oot)

    # 2. 切点漂移
    drifts = [abs(pq_train - pq_val)]
    if pq_oot is not None:
        drifts.append(abs(pq_train - pq_oot))
        drifts.append(abs(pq_val - pq_oot))
    peak_drift = max(drifts)

    if peak_drift > PEAK_DRIFT_THRESHOLD:
        warnings.append(
            f"KS 切点漂移 {peak_drift:.1%} > {PEAK_DRIFT_THRESHOLD:.0%}，"
            "模型在不同数据集上依赖不同人群排序"
        )

    # 3. 陡峭度（取最大值）
    sharpness = max(sharp_train, sharp_val, sharp_oot)
    if sharpness > SHARPNESS_THRESHOLD:
        warnings.append(
            f"KS 曲线陡峭度 {sharpness:.2f} > {SHARPNESS_THRESHOLD}，"
            "KS 指标高度依赖少数样本"
        )

    # 4. KS-AUC 一致性
    ks_auc_ok = True
    if all(v is not None for v in [ks_train, ks_val, auc_train, auc_val]):
        ks_auc_ok = _check_ks_auc_consistency(ks_train, ks_val, auc_train, auc_val)
        if not ks_auc_ok:
            warnings.append("KS 与 AUC 变化方向不一致，需人工确认模型有效性")

    # 5. 综合判定
    is_hidden_overfit = (
        peak_drift > PEAK_DRIFT_THRESHOLD
        or sharpness > SHARPNESS_THRESHOLD
        or not ks_auc_ok
    )

    return KSStructureDiagnosis(
        peak_quantile_train=pq_train,
        peak_quantile_val=pq_val,
        peak_quantile_oot=pq_oot,
        peak_drift=peak_drift,
        curve_sharpness=sharpness,
        ks_auc_consistency=ks_auc_ok,
        is_hidden_overfit=is_hidden_overfit,
        warnings=warnings,
    )

__all__ = ["KSStructureDiagnosis", "run_ks_structure_check"]
