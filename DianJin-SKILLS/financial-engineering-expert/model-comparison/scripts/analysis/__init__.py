# -*- coding: utf-8 -*-
"""多维度分析器 — 纯函数，无状态，可独立单测。

模块职责（对应报告章节）：
  - metrics: 补全 Brier/Precision@K/Recall@K/BCR（EvalMetrics 升级）
  - significance: DeLong 检验（两两算法 AUC 差异显著性）
  - stability: 真实分数 PSI (OOT vs Train) + 可选分月 KS
  - agreement: 样本级一致性（Top% 重叠 / Spearman 秩相关 / 分歧案例）
  - feature_overlap: 跨算法 Top-N 特征重叠（Jaccard + 共识特征）
"""
from analysis.metrics import enrich_metrics
from analysis.significance import pairwise_delong
from analysis.stability import compute_psi, compute_monthly_performance
from analysis.agreement import pairwise_agreement
from analysis.feature_overlap import feature_importance_overlap

__all__ = [
    "enrich_metrics",
    "pairwise_delong",
    "compute_psi",
    "compute_monthly_performance",
    "pairwise_agreement",
    "feature_importance_overlap",
]
