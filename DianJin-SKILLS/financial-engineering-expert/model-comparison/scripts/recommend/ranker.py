# -*- coding: utf-8 -*-
"""客观事实计算器 — 不做主观打分，只输出可被 LLM 消费的事实清单。

设计动机：
  历史版本里有「五维加权评分 + 硬门禁淘汰」，本质是把多个不可比指标
  用拍脑袋的权重压成一个数，丢失信息且引入主观假设。
  现改为「只陈述客观事实」：
    - 每个算法的原始指标快照（OOT AUC/KS/BCR/Brier/KS Gap/PSI/特征数）
    - Pareto 前沿（在所有目标维度上不被任何算法严格优于的算法集合）
    - 关键事实陈述（OOT AUC 范围、过拟合风险、漂移强弱、Pareto 候选集）
  剩下的「哪个更适合」交给 LLM 在对话里基于事实推理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from recommend.scenarios import Scenario
from trainers.base import TrainResult

# ── Pareto 前沿目标维度配置 ─────────────────────────────────
# (字段名, 方向: "max" 越大越好 / "min" 越小越好)
_PARETO_DIMENSIONS = (
    ("oot_auc", "max"),
    ("oot_ks", "max"),
    ("oot_bcr10", "max"),
    ("ks_gap", "min"),
    ("psi", "min"),
)

@dataclass
class AlgoFacts:
    """单个算法的客观事实快照（无任何打分）。"""
    short: str
    algorithm: str
    n_features: int
    interpretability: str

    # 原始指标快照
    oot_auc: float
    oot_ks: float
    oot_gini: float
    oot_bcr10: float
    oot_brier: float
    ks_gap: float
    psi: float

    # 客观标签
    is_pareto_optimal: bool = False
    overfit_label: str = ""    # "良好" / "需关注" / "过拟合风险"
    psi_label: str = ""        # "稳定" / "轻微漂移" / "显著漂移"

@dataclass
class ComparisonFacts:
    """整体对比的事实陈述（替代旧 RankVerdict）。"""
    scenario_key: str
    scenario_name: str
    scenario_description: str
    algos: List[AlgoFacts]                # 已按 OOT AUC 降序
    pareto_frontier: List[str]            # short 列表
    facts_for_llm: List[str]              # 中文事实陈述（供 LLM 与人类共同消费）
    summary: str                          # 一句话事实陈述

# ── 标签函数（仅文本映射，无评分含义）─────────────────────────
def _overfit_label(ks_gap: float) -> str:
    if ks_gap > 0.10:
        return "过拟合风险（KS Gap > 0.10）"
    if ks_gap > 0.05:
        return "需关注（KS Gap > 0.05）"
    return "良好"

def _psi_label(psi: float) -> str:
    if psi >= 0.25:
        return "显著漂移（PSI ≥ 0.25）"
    if psi >= 0.10:
        return "轻微漂移（PSI ≥ 0.10）"
    return "稳定"

# ── Pareto 前沿计算 ────────────────────────────────────────
def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """a 是否严格支配 b：所有维度上 a ≥ b（按方向），且至少一个 > 。"""
    strictly_better_at_least_one = False
    for key, direction in _PARETO_DIMENSIONS:
        av, bv = a[key], b[key]
        if direction == "max":
            if av < bv:
                return False
            if av > bv:
                strictly_better_at_least_one = True
        else:  # "min"
            if av > bv:
                return False
            if av < bv:
                strictly_better_at_least_one = True
    return strictly_better_at_least_one

def _compute_pareto_frontier(facts: List[AlgoFacts]) -> List[str]:
    """返回 Pareto 前沿上算法的 short 列表（保持原顺序）。"""
    if len(facts) <= 1:
        return [f.short for f in facts]
    snapshots = {
        f.short: {
            "oot_auc": f.oot_auc,
            "oot_ks": f.oot_ks,
            "oot_bcr10": f.oot_bcr10,
            "ks_gap": f.ks_gap,
            "psi": f.psi,
        }
        for f in facts
    }
    frontier: List[str] = []
    for f in facts:
        dominated = any(
            _dominates(snapshots[other.short], snapshots[f.short])
            for other in facts if other.short != f.short
        )
        if not dominated:
            frontier.append(f.short)
    return frontier

# ── 入口 ──────────────────────────────────────────────────
def _facts_from_result(r: TrainResult) -> AlgoFacts:
    return AlgoFacts(
        short=r.short,
        algorithm=r.algorithm,
        n_features=r.n_features,
        interpretability=r.interpretability,
        oot_auc=round(r.oot.auc, 6),
        oot_ks=round(r.oot.ks, 6),
        oot_gini=round(r.oot.gini, 6),
        oot_bcr10=round(r.oot.bcr_top10, 6),
        oot_brier=round(r.oot.brier, 6),
        ks_gap=round(r.ks_gap, 6),
        psi=round(r.psi, 6),
        overfit_label=_overfit_label(r.ks_gap),
        psi_label=_psi_label(r.psi),
    )

def _build_facts_for_llm(facts: List[AlgoFacts], frontier: List[str]) -> List[str]:
    """组织一段「LLM 友好」的事实清单，供下游模型推理。

    刻意不出现 "推荐 / 综合分 / 加权" 等主观词；让 LLM 自己看事实下结论。
    """
    facts_lines: List[str] = []

    # 1) OOT AUC 范围
    aucs = [f.oot_auc for f in facts]
    facts_lines.append(
        f"OOT AUC 范围：{min(aucs):.4f} ~ {max(aucs):.4f}"
        f"（共 {len(facts)} 个算法）"
    )

    # 2) 哪个 AUC 最高
    top_auc = max(facts, key=lambda f: f.oot_auc)
    facts_lines.append(f"OOT AUC 最高：{top_auc.algorithm}（{top_auc.oot_auc:.4f}）")

    # 3) 哪个最稳（PSI 最小）
    min_psi = min(facts, key=lambda f: f.psi)
    facts_lines.append(
        f"OOT vs Train 分布漂移最小：{min_psi.algorithm}"
        f"（PSI={min_psi.psi:.4f}，{min_psi.psi_label}）"
    )

    # 4) 过拟合风险情况
    risky = [f for f in facts if f.ks_gap > 0.05]
    if risky:
        facts_lines.append(
            "存在过拟合迹象的算法："
            + "；".join(f"{f.algorithm}（KS Gap={f.ks_gap:.4f}）" for f in risky)
        )

    # 5) 数据漂移情况
    drift = [f for f in facts if f.psi >= 0.10]
    if drift:
        facts_lines.append(
            "存在分布漂移的算法："
            + "；".join(f"{f.algorithm}（PSI={f.psi:.4f}，{f.psi_label}）" for f in drift)
        )

    # 6) Pareto 前沿
    if frontier:
        names = "、".join(
            next(f.algorithm for f in facts if f.short == s) for s in frontier
        )
        if len(frontier) == 1:
            facts_lines.append(
                f"Pareto 前沿（在 OOT AUC/KS/BCR、KS Gap、PSI 全维度无人严格优于）"
                f"仅有 1 个候选：{names}"
            )
        else:
            facts_lines.append(
                f"Pareto 前沿（在 OOT AUC/KS/BCR、KS Gap、PSI 全维度无人严格优于）"
                f"共 {len(frontier)} 个候选：{names}（彼此各有侧重，无客观最优）"
            )

    # 7) 极端低 AUC 提醒（接近随机）
    near_random = [f for f in facts if f.oot_auc < 0.55]
    if near_random:
        facts_lines.append(
            "OOT AUC 接近随机（<0.55）的算法："
            + "；".join(f"{f.algorithm}（{f.oot_auc:.4f}）" for f in near_random)
            + "——可能是数据漂移、未来信息泄漏或样本不足，建议先做数据/特征诊断"
        )

    return facts_lines

def compute_facts(
    results: Dict[str, TrainResult],
    scenario: Scenario,
) -> ComparisonFacts:
    """给一组训练结果，返回客观事实陈述（不做任何主观打分/推荐）。"""
    facts = [_facts_from_result(r) for r in results.values()]
    facts.sort(key=lambda f: f.oot_auc, reverse=True)

    frontier = _compute_pareto_frontier(facts)
    for f in facts:
        f.is_pareto_optimal = f.short in frontier

    facts_for_llm = _build_facts_for_llm(facts, frontier)

    if not frontier:
        summary = f"场景『{scenario.name}』：共 {len(facts)} 个算法已对比，无 Pareto 前沿"
    elif len(frontier) == 1:
        only = next(f for f in facts if f.short == frontier[0])
        summary = (
            f"场景『{scenario.name}』：Pareto 前沿仅 1 个候选 — {only.algorithm}"
            f"（OOT AUC={only.oot_auc:.4f}）"
        )
    else:
        names = "、".join(
            next(f.algorithm for f in facts if f.short == s) for s in frontier
        )
        summary = (
            f"场景『{scenario.name}』：Pareto 前沿 {len(frontier)} 个候选 — {names}"
            f"（各有侧重，请基于业务取舍由 LLM 推理）"
        )

    return ComparisonFacts(
        scenario_key=scenario.key,
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        algos=facts,
        pareto_frontier=frontier,
        facts_for_llm=facts_for_llm,
        summary=summary,
    )

# ── 旧名称向后兼容（外部 import 不立即崩）──────────────────────
RankVerdict = ComparisonFacts
AlgoScoring = AlgoFacts
rank_algorithms = compute_facts

__all__ = [
    "AlgoFacts",
    "ComparisonFacts",
    "compute_facts",
    # 兼容名
    "AlgoScoring",
    "RankVerdict",
    "rank_algorithms",
]
