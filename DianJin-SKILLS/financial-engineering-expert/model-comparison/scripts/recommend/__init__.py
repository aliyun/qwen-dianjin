# -*- coding: utf-8 -*-
"""对比上下文子包 — 仅产出客观事实清单（Pareto 前沿 + 原始指标），不做主观评分。"""
from recommend.scenarios import SCENARIOS, get_scenario
from recommend.ranker import (
    compute_facts,
    ComparisonFacts,
    AlgoFacts,
    # 兼容旧名
    rank_algorithms,
    RankVerdict,
    AlgoScoring,
)

__all__ = [
    "SCENARIOS",
    "get_scenario",
    "compute_facts",
    "ComparisonFacts",
    "AlgoFacts",
    "rank_algorithms",
    "RankVerdict",
    "AlgoScoring",
]
