# -*- coding: utf-8 -*-
"""报告倒金字塔装配 — 薄胶水，只决定章节顺序。

设计原则：
  - 最重要的客观结论放最前（执行摘要 + 视觉化对比）
  - 技术细节（指标总表、Pareto 前沿、显著性、稳定性）居中
  - 可逐段扩展的"深度分析"（一致性、特征重叠）垫底
  - LLM 推理引导 + 适用场景知识放最后

注：本报告**不做主观推荐**，仅陈述事实，由 LLM/业务方基于事实推理决策。
任何一个章节失败，尽量不影响其他章节（try-except 软降级）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from analysis import feature_importance_overlap, pairwise_agreement, pairwise_delong
from recommend.ranker import ComparisonFacts
from reporting import sections
from trainers.base import TrainResult

logger = logging.getLogger(__name__)

def _safe_section(fn, *args, **kwargs) -> str:
    """一个章节失败不拖累整份报告。"""
    try:
        return fn(*args, **kwargs) or ""
    except Exception as e:  # pragma: no cover
        logger.exception("章节生成失败 %s: %s", fn.__name__, e)
        return f"\n> ⚠️ 章节 `{fn.__name__}` 生成失败：{e}\n\n"

def build_report(
    *,
    data_path: str,
    target: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    results: Dict[str, TrainResult],
    facts: ComparisonFacts,
    split_meta: Dict[str, Any] = None,
) -> str:
    """按倒金字塔顺序装配多模型对比报告。"""
    y_oot = oot_data[target].values
    prob_map_oot = {r.algorithm: r.prob_oot for r in results.values() if r.prob_oot is not None}
    algo_name = {r.short: r.algorithm for r in results.values()}

    # ── 预计算：下游章节复用（避免重复算）──
    prob_short_map = {r.short: r.prob_oot for r in results.values() if r.prob_oot is not None}
    delong = _safe_dict(pairwise_delong, y_oot, prob_short_map)
    agreement = _safe_dict(pairwise_agreement, prob_short_map)
    overlap = _safe_dict_single(feature_importance_overlap, results, 20)

    parts = []
    # 1. 执行摘要（含 Pareto 前沿标记的客观指标速览）
    parts.append(_safe_section(sections.section_exec_summary, results, facts, data_path, target))
    parts.append("---\n")

    # 2. 视觉化对比（叠加曲线，已删雷达图）
    parts.append(_safe_section(sections.section_overlap_curves, y_oot, prob_map_oot))
    parts.append("---\n")

    # 3. 数据概览
    parts.append(_safe_section(sections.section_data_overview, train_data, val_data, oot_data, target, split_meta))
    parts.append("---\n")

    # 4. 详细指标总表
    parts.append(_safe_section(sections.section_metrics_table, results))
    parts.append("---\n")

    # 5. Pareto 前沿候选集（替代旧『场景化打分』）
    parts.append(_safe_section(sections.section_pareto_frontier, facts))
    parts.append("---\n")

    # 6. 显著性
    if delong:
        parts.append(_safe_section(sections.section_significance, delong, algo_name))
        parts.append("---\n")

    # 7. 稳定性
    parts.append(_safe_section(sections.section_stability, results))
    parts.append("---\n")

    # 8. 一致性
    if agreement:
        parts.append(_safe_section(sections.section_agreement, agreement, algo_name))
        parts.append("---\n")

    # 9. 特征重叠
    if overlap and overlap.get("participants"):
        parts.append(_safe_section(sections.section_feature_overlap, overlap, algo_name))
        parts.append("---\n")

    # 10. 每算法独立评估
    parts.append(_safe_section(sections.section_per_algo_detail, results, y_oot))
    parts.append("---\n")

    # 11. LLM 推理引导（事实清单 + 开放问题）
    parts.append(_safe_section(sections.section_facts_for_llm, facts))
    parts.append("---\n")

    # 12. 各算法适用场景背景
    parts.append(_safe_section(sections.section_usage_guidance, results))

    return "\n".join(p for p in parts if p)

def _safe_dict(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        logger.exception("分析失败 %s: %s", fn.__name__, e)
        return {}

def _safe_dict_single(fn, results, top_n):
    try:
        return fn(results, top_n=top_n)
    except Exception as e:
        logger.exception("分析失败 %s: %s", fn.__name__, e)
        return {}
