# -*- coding: utf-8 -*-
"""报告各章节生成器 — 每个函数返回一段 markdown，可独立测试、重组顺序。

章节清单（按倒金字塔顺序消费）：
  1. section_exec_summary    执行摘要（按 OOT AUC 排序的事实表，⭐ 标 Pareto 前沿）
  2. section_overlap_curves  叠加曲线（ROC/KS/Lift）
  3. section_data_overview   数据切分概览
  4. section_metrics_table   详细指标总表（train/val/oot × 8 指标）
  5. section_pareto_frontier Pareto 前沿候选集（无主观权重）
  6. section_significance    DeLong 显著性两两检验
  7. section_stability       PSI + 分月 KS 趋势
  8. section_agreement       样本级一致性
  9. section_feature_overlap 跨算法特征重叠
 10. section_per_algo_detail 每个算法的独立评估图
 11. section_facts_for_llm   LLM 推理引导（事实清单+开放问题）
 12. section_usage_guidance  各算法适用场景背景
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from analysis.stability import interpret_psi
from data_utils import render_split_section
from recommend.ranker import ComparisonFacts
from report_artifact import (
    render_bad_capture_rate,
    render_feature_importance,
    render_ks_curve,
    render_lift_chart,
    render_roc_curve,
)
from reporting.renderer import (
    render_multi_ks_curve,
    render_multi_lift_curve,
    render_multi_roc_curve,
)
from trainers.base import TrainResult

# ── 1) 执行摘要 ─────────────────────────────────────────────────
def section_exec_summary(
    results: Dict[str, TrainResult],
    facts: ComparisonFacts,
    data_path: str,
    target: str,
) -> str:
    lines: List[str] = []
    lines.append("# 多模型效果对比报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 数据文件：{data_path}")
    lines.append(f"> 目标变量：{target}")
    lines.append(f"> 对比算法：{' / '.join(r.algorithm for r in results.values())}")
    lines.append(f"> 业务场景：**{facts.scenario_name}** — {facts.scenario_description}")
    lines.append("")
    lines.append("## 🎯 客观指标速览")
    lines.append("")
    lines.append(
        "> 本报告**不做主观推荐**，仅陈述事实。⭐ 标记表示该算法位于 Pareto 前沿"
        "（在 OOT AUC/KS/BCR、KS Gap、PSI 全部维度上无人严格优于）。"
        "请由 LLM 或业务方基于事实自行取舍。"
    )
    lines.append("")
    lines.append(f"**{facts.summary}**")
    lines.append("")

    lines.append(
        "| 算法 | 入模特征 | OOT AUC | OOT KS | OOT BCR@10% | KS Gap | PSI | 可解释性 |"
    )
    lines.append("|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for f in facts.algos:
        marker = " ⭐" if f.is_pareto_optimal else ""
        lines.append(
            f"| {f.algorithm}{marker} | {f.n_features} | "
            f"{f.oot_auc:.4f} | {f.oot_ks:.4f} | {f.oot_bcr10:.4f} | "
            f"{f.ks_gap:.4f} | {f.psi:.4f} | {f.interpretability} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"

# ── 2) 叠加曲线（去除雷达图）─────────────────────────────────────
def section_overlap_curves(
    y_oot,
    prob_map_oot: Dict[str, Any],
) -> str:
    if len(prob_map_oot) < 2:
        return ""
    lines = ["## 🎨 多算法叠加曲线（OOT）", ""]
    lines.append(render_multi_roc_curve(y_oot, prob_map_oot))
    lines.append("")
    lines.append(render_multi_ks_curve(y_oot, prob_map_oot))
    lines.append("")
    lines.append(render_multi_lift_curve(y_oot, prob_map_oot))
    lines.append("")
    return "\n".join(lines)

# ── 3) 数据概览 ─────────────────────────────────────────────────
def section_data_overview(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    target: str,
    split_meta: Dict[str, Any] = None,
) -> str:
    lines = ["## 📊 数据概览", ""]
    if split_meta:
        lines.append(render_split_section(split_meta))
        lines.append("")
    lines.append("| 数据集 | 样本量 | 正样本量 | 正样本率 |")
    lines.append("|--------|:------:|:--------:|:--------:|")
    for name, df in [("Train", train_data), ("Val", val_data), ("OOT", oot_data)]:
        lines.append(
            f"| {name} | {len(df):,} | {int(df[target].sum()):,} | {df[target].mean():.4%} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"

# ── 4) 详细指标总表 ─────────────────────────────────────────────
def section_metrics_table(results: Dict[str, TrainResult]) -> str:
    lines = ["## 📐 详细指标总表", ""]
    for r in results.values():
        lines.append(f"### {r.algorithm}")
        lines.append("")
        lines.append("| 指标 | Train | Val | OOT |")
        lines.append("|------|:-----:|:---:|:---:|")
        for mkey, mname in [
            ("auc", "AUC"),
            ("ks", "KS"),
            ("gini", "Gini"),
            ("bcr_top10", "BCR@10%"),
            ("precision_top10", "Precision@10%"),
            ("recall_top10", "Recall@10%"),
            ("brier", "Brier"),
            ("logloss", "LogLoss"),
        ]:
            t = getattr(r.train, mkey)
            v = getattr(r.val, mkey)
            o = getattr(r.oot, mkey)
            lines.append(f"| {mname} | {t:.4f} | {v:.4f} | {o:.4f} |")
        lines.append("")
        gap = r.ks_gap
        risk = "⚠️ 过拟合风险" if gap > 0.10 else ("需关注" if gap > 0.05 else "良好")
        lines.append(
            f"- 入模特征：{r.n_features} | KS Gap：{gap:.4f}（{risk}） | "
            f"可解释性：{r.interpretability} | 分数 PSI：{r.psi:.4f}（{interpret_psi(r.psi)}）"
        )
        lines.append("")
    return "\n".join(lines)

# ── 5) Pareto 前沿候选集（替代旧『场景化打分』）──────────────
def section_pareto_frontier(facts: ComparisonFacts) -> str:
    lines = ["## 🧭 Pareto 前沿候选集（无主观权重）", ""]
    lines.append(
        "> Pareto 前沿：在 OOT AUC、OOT KS、OOT BCR@10%、KS Gap、PSI 这些「方向明确」"
        "的目标上，**没有任何其他算法能严格优于**它的算法集合。"
        "前沿内的算法彼此各有侧重，不存在客观最优。"
    )
    lines.append("")

    if not facts.pareto_frontier:
        lines.append("- 当前结果未识别出 Pareto 前沿。")
        return "\n".join(lines) + "\n"

    name_map = {f.short: f.algorithm for f in facts.algos}
    lines.append(f"**前沿候选（{len(facts.pareto_frontier)} 个）**：")
    for short in facts.pareto_frontier:
        f = next(x for x in facts.algos if x.short == short)
        lines.append(
            f"- ⭐ **{f.algorithm}** — OOT AUC={f.oot_auc:.4f} / OOT KS={f.oot_ks:.4f} / "
            f"BCR@10%={f.oot_bcr10:.4f} / KS Gap={f.ks_gap:.4f} / PSI={f.psi:.4f}"
        )
    lines.append("")

    dominated = [f for f in facts.algos if not f.is_pareto_optimal]
    if dominated:
        lines.append("**被支配（在所有维度上至少有一个算法不差且至少一项更好）**：")
        for f in dominated:
            lines.append(f"- {f.algorithm}")
        lines.append("")

    return "\n".join(lines) + "\n"

# ── 6) 显著性检验 ─────────────────────────────────────────────
def section_significance(
    delong: Dict[Tuple[str, str], Dict[str, float]],
    algo_name: Dict[str, str],
) -> str:
    if not delong:
        return ""
    lines = ["## 🔬 AUC 差异显著性（DeLong 检验）", ""]
    lines.append("| 算法 A | 算法 B | AUC_A | AUC_B | ΔAUC | p-value | 是否显著(α=0.05) |")
    lines.append("|--------|--------|:---:|:---:|:---:|:---:|:---:|")
    for (a, b), r in delong.items():
        sig = "✅ 显著" if r["significant"] else "❌ 不显著"
        lines.append(
            f"| {algo_name.get(a, a)} | {algo_name.get(b, b)} | {r['auc_a']:.4f} | "
            f"{r['auc_b']:.4f} | {r['delta']:+.4f} | {r['p_value']:.4f} | {sig} |"
        )
    lines.append("")
    lines.append(
        "> DeLong 在原始 AUC 差值上做渐近正态检验，用于判断"
        "「效果差距是统计显著还是采样波动」。"
    )
    lines.append("")
    return "\n".join(lines)

# ── 7) 稳定性 ───────────────────────────────────────────────────
def section_stability(results: Dict[str, TrainResult]) -> str:
    lines = ["## 📦 稳定性分析（分数 PSI）", ""]
    lines.append("| 算法 | OOT vs Train PSI | 判定 | 分月 KS 波动(σ) |")
    lines.append("|------|:---:|:---:|:---:|")
    for r in results.values():
        if r.monthly_ks:
            import numpy as np
            ks_list = [m["ks"] for m in r.monthly_ks]
            std = float(np.std(ks_list)) if ks_list else 0.0
            ks_std = f"{std:.4f}"
        else:
            ks_std = "—"
        lines.append(
            f"| {r.algorithm} | {r.psi:.4f} | {interpret_psi(r.psi)} | {ks_std} |"
        )
    lines.append("")
    lines.append(
        "> PSI < 0.10 稳定 / 0.10–0.25 轻微漂移 / ≥ 0.25 显著漂移。"
        "真实 PSI 在 OOT vs Train 分数分布上计算，不再用 AUC gap 近似。"
    )
    lines.append("")
    return "\n".join(lines)

# ── 8) 样本一致性 ───────────────────────────────────────────────
def section_agreement(
    agreement: Dict[Tuple[str, str], Dict[str, float]],
    algo_name: Dict[str, str],
) -> str:
    if not agreement:
        return ""
    lines = ["## 🤝 样本级一致性", ""]
    lines.append("| 算法 A | 算法 B | Top10% 重叠(Jaccard) | Spearman 秩相关 | 标签分歧数 |")
    lines.append("|--------|--------|:---:|:---:|:---:|")
    for (a, b), r in agreement.items():
        lines.append(
            f"| {algo_name.get(a, a)} | {algo_name.get(b, b)} | "
            f"{r['top10_jaccard']:.4f} | {r['spearman_rho']:.4f} | {r['disagreement']:,} |"
        )
    lines.append("")
    lines.append(
        "> Jaccard 越高说明"
        "「挑出的 Top 10% 坏客户」越一致；"
        "Spearman 越接近 1 说明整体排序高度一致；"
        "分歧数是两算法在中位数阈值下标签相反的样本数。"
    )
    lines.append("")
    return "\n".join(lines)

# ── 9) 特征重叠 ─────────────────────────────────────────────────
def section_feature_overlap(overlap: Dict[str, Any], algo_name: Dict[str, str]) -> str:
    participants = overlap.get("participants", [])
    if len(participants) < 2:
        return ""
    lines = ["## 🧬 跨算法特征重要性重叠", ""]
    lines.append(
        f"- 参与算法：{' / '.join(algo_name.get(p, p) for p in participants)}"
        f" | Top-{overlap['top_n']} 共识集大小：{overlap['intersection_size']}"
        f"（并集 {overlap['union_size']}）"
    )
    lines.append("")
    pairs = overlap.get("pairwise_jaccard", {})
    if pairs:
        lines.append("### 两两 Jaccard 相似度")
        lines.append("")
        lines.append("| 算法 A | 算法 B | Jaccard |")
        lines.append("|--------|--------|:---:|")
        for (a, b), jac in pairs.items():
            lines.append(f"| {algo_name.get(a, a)} | {algo_name.get(b, b)} | {jac:.4f} |")
        lines.append("")

    consensus = overlap.get("consensus", [])
    if consensus:
        lines.append(f"### 共识特征（至少出现在 2 个算法 Top-{overlap['top_n']}）")
        lines.append("")
        lines.append("| 特征 | 命中算法数 |")
        lines.append("|------|:---:|")
        for feat, cnt in consensus[:20]:
            lines.append(f"| {feat} | {cnt} |")
        lines.append("")
    return "\n".join(lines)

# ── 10) 每个算法独立评估图 ─────────────────────────────────────
def section_per_algo_detail(results: Dict[str, TrainResult], y_oot) -> str:
    lines = ["## 🔍 每个算法的独立评估图", ""]
    for r in results.values():
        lines.append(f"### {r.algorithm}")
        lines.append("")
        lines.append(render_roc_curve(y_oot, r.prob_oot, prefix=f"{r.short.upper()} OOT "))
        lines.append(render_ks_curve(y_oot, r.prob_oot, prefix=f"{r.short.upper()} OOT "))
        lines.append(render_lift_chart(y_oot, r.prob_oot, prefix=f"{r.short.upper()} OOT "))
        lines.append(render_bad_capture_rate(y_oot, r.prob_oot, prefix=f"{r.short.upper()} OOT "))
        if r.feature_importance:
            imp_dict = {k: v for k, v in r.feature_importance[:20]}
            lines.append(render_feature_importance(imp_dict, top_n=20, prefix=f"{r.short.upper()} "))
        lines.append("")
    return "\n".join(lines)

# ── 11) LLM 推理引导：事实清单 + 开放问题 ──────────────────────
def section_facts_for_llm(facts: ComparisonFacts) -> str:
    lines = ["## 🧠 LLM 推理引导（基于事实）", ""]
    lines.append(
        "> 以下事实由代码客观计算，**不含任何主观打分**。"
        "请基于这些事实结合业务背景，回答下方开放问题。"
    )
    lines.append("")
    lines.append("### 已知事实")
    for line in facts.facts_for_llm:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("### 待 LLM 推理的开放问题")
    lines.append(
        "1. **是否存在数据问题？**（OOT AUC 是否接近随机、KS Gap 是否过大、PSI 是否显著漂移）；"
        "如有，建议先做 `data-profiling` / `feature-analysis` 还是直接淘汰当前数据切分？"
    )
    lines.append(
        "2. **若上线，会选谁？** 在 Pareto 前沿候选集里，"
        "结合业务对（解释性 / 漂移容忍度 / 黑盒接受度 / 部署成本）的优先级，给出取舍理由。"
    )
    lines.append(
        "3. **下一步动作**：是否需要再调一轮参数（哪个算法、调哪些超参）？"
        "是否需要重新切分 OOT 时间窗？是否考虑算法融合（Stacking / Blending）？"
    )
    lines.append("")
    return "\n".join(lines)

# ── 12) 适用场景建议（背景知识，非主观打分）──────────────────
_USAGE_TIPS = {
    "xgb": "**XGBoost**：追求效果极致、特征交互复杂、可接受黑盒；配合 SHAP 做局部解释",
    "lr": "**LR + WoE**：白盒可解释、监管合规、评分卡部署；IV 低的特征会被自动剔除",
    "dnn": "**DNN (MLP)**：高维交互、大数据量、可与 XGB 做 Ensemble；需要充足训练样本",
}

def section_usage_guidance(results: Dict[str, TrainResult]) -> str:
    lines = ["## 📚 各算法适用场景背景", ""]
    for key in ("xgb", "lr", "dnn"):
        if key in results:
            lines.append(f"- {_USAGE_TIPS[key]}")
    lines.append("")
    return "\n".join(lines)
