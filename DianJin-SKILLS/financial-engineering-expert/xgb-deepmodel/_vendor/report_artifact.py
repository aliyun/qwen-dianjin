# -*- coding: utf-8 -*-
"""
XGB 评估图表渲染工具 — KS / ROC / Lift / 特征重要性 / BCR / 概率校准 六件套

职责边界（important）：
  - 纯渲染：每个 render_* 返回一段 markdown 片段（quote 摘要行 + <artifact/> 标签），
    可直接拼入最终报告；前端在渲染时从注释包裹的 artifact 标签取数据。
  - 不含业务解读：Lift/Brier/特征集中度等文案留给调用方，本模块只输出"数据图表"本身。
  - 调用方自由组合：modeling.py 需要在图之间穿插业务解读 → 逐个调 render_*；
    tuner.py / sub_trainer.py / stacker.py 不在乎中间文案 → 直接用 build_full_eval_section。

过渡说明：
  render_ks/roc/lift/bcr/calibration 依赖 xgb_core.evaluator.ModelEvaluator 的
  calculate_*_data() 做底层计算。xgb_core 由 python_executor 预置
  PYTHONPATH=backend/libs 暴露。

使用示例：
    from report_artifact import (
        render_ks_curve, render_roc_curve, render_lift_chart,
        render_feature_importance, render_bad_capture_rate, render_calibration_curve,
        build_full_eval_section,
    )

    # 模式 A：自由组合（modeling.py 风格）
    md = []
    md.append(render_ks_curve(y_oot, prob_oot, prefix="OOT "))
    md.append("> KS 解读：...")   # 业务文案由调用方负责
    md.append(render_roc_curve(y_oot, prob_oot, prefix="OOT "))
    ...

    # 模式 B：一把梭（tuner.py 风格）
    md = build_full_eval_section(y_oot, prob_oot, importance_dict, prefix="最优模型 ")
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# xgb_core 由 python_executor 预置 PYTHONPATH=backend/libs 暴露
# 开发期直接 python 调试请设置：PYTHONPATH=backend/libs:backend/skills/shared
from xgb_core import ModelEvaluator

# ---------------------------------------------------------------------------
# 内部工具：artifact 双格式包裹（HTML 注释 + <artifact/> 标签）
# ---------------------------------------------------------------------------

def _artifact_block(artifact_type: str, title: str, payload: Dict[str, Any]) -> str:
    """生成一段 artifact 标签（纯文本，调用方自行与摘要行拼接）。

    双格式：HTML 注释包裹，保证 Markdown 渲染不可见、前端正常提取。
    """
    return (
        f'<!--ARTIFACT_START--><artifact type="{artifact_type}" title="{title}">'
        + json.dumps(payload, ensure_ascii=False)
        + "</artifact><!--ARTIFACT_END-->"
    )

# ---------------------------------------------------------------------------
# 6 个独立渲染函数
# ---------------------------------------------------------------------------

def render_ks_curve(y_true, y_prob, *, prefix: str = "") -> str:
    """KS 曲线 artifact 段。

    Args:
        y_true, y_prob : 真实标签与预测概率（支持 ndarray/Series/list）
        prefix         : artifact title 前缀（如 "OOT "、"最优模型 "）
    Returns:
        markdown 片段：一行摘要 quote + artifact 标签 + 末尾空行
    """
    data = ModelEvaluator.calculate_ks_curve_data(y_true, y_prob)
    ks_val = data["ks_value"]
    summary = f"> 📉 **KS 曲线**（KS = {ks_val:.4f}）\n"
    block = _artifact_block("ks-curve", f"{prefix}KS 曲线（KS={ks_val:.4f}）", data)
    return f"{summary}\n{block}\n"

def render_roc_curve(y_true, y_prob, *, prefix: str = "") -> str:
    """ROC 曲线 artifact 段。"""
    data = ModelEvaluator.calculate_roc_data(y_true, y_prob)
    auc_val = data["auc"]
    summary = f"> 📈 **ROC 曲线**（AUC = {auc_val:.4f}）\n"
    block = _artifact_block("roc-curve", f"{prefix}ROC 曲线（AUC={auc_val:.4f}）", data)
    return f"{summary}\n{block}\n"

def render_lift_chart(y_true, y_prob, *, prefix: str = "") -> str:
    """Lift Chart artifact 段（仅图表；Lift 表与解读由调用方负责）。"""
    data = ModelEvaluator.calculate_lift_chart_data(y_true, y_prob)
    summary = "> 📊 **Lift Chart**\n"
    block = _artifact_block("lift-chart", f"{prefix}Lift Chart", data)
    return f"{summary}\n{block}\n"

def render_lift_chart_optbinning(
    y_true, y_prob, *, prefix: str = "", max_n_bins: int = 8, min_bin_size: float = 0.05
) -> str:
    """Lift Chart artifact 段（IV 最优分箱版本）。"""
    data = ModelEvaluator.calculate_lift_chart_data_optbinning(
        y_true, y_prob, max_n_bins=max_n_bins, min_bin_size=min_bin_size
    )
    if data is None:
        return "> ⚠️ **Lift Chart（IV 最优分箱）**：optbinning 未安装或计算失败\n\n"
    summary = "> 📊 **Lift Chart（IV 最优分箱）**\n"
    block = _artifact_block("lift-chart", f"{prefix}Lift Chart（IV 最优分箱）", data)
    return f"{summary}\n{block}\n"

def render_feature_importance(
    importance: Dict[str, float],
    *,
    top_n: int = 15,
    prefix: str = "",
) -> str:
    """特征重要性 artifact 段。

    Args:
        importance: {feature_name: importance} 字典；顺序可不保证，内部会降序截断。
                    有意不依赖 XGBTrainer，以便 sklearn / LightGBM / 线性模型复用。
        top_n:      截取前 N 个特征
    """
    # 过滤掉 None / NaN，按重要性降序
    items: Iterable[Tuple[str, float]] = (
        (str(k), float(v))
        for k, v in (importance or {}).items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    )
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
    payload = {
        "features": [k for k, _ in sorted_items],
        "importances": [v for _, v in sorted_items],
    }
    summary = f"> 🎯 **特征重要性（Top {len(sorted_items)}）**\n"
    block = _artifact_block("feature-importance", f"{prefix}特征重要性", payload)
    return f"{summary}\n{block}\n"

def render_bad_capture_rate(y_true, y_prob, *, prefix: str = "") -> str:
    """Bad Capture Rate 折线图 artifact 段（echarts）。"""
    bcr = ModelEvaluator.calculate_bad_capture_rate(y_true, y_prob)
    topk_pcts = bcr["topk_pcts"]
    bcr_rates = bcr["bad_capture_rates"]
    option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["BCR", "随机基线"], "bottom": 0},
        "grid": {
            "left": "3%", "right": "4%",
            "bottom": "15%", "top": "10%",
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": [f"Top{p}%" for p in topk_pcts],
            "name": "拒绝比例",
        },
        "yAxis": {
            "type": "value", "name": "坏客户捕获率",
            "min": 0, "max": 1,
            "axisLabel": {"formatter": "{value}"},
        },
        "series": [
            {
                "name": "BCR", "type": "line",
                "data": [round(r, 4) for r in bcr_rates],
                "smooth": True,
                "lineStyle": {"color": "#6366f1", "width": 2},
                "itemStyle": {"color": "#6366f1"},
                "symbol": "circle", "symbolSize": 6,
            },
            {
                "name": "随机基线", "type": "line",
                "data": [round(p / 100, 4) for p in topk_pcts],
                "smooth": False,
                "lineStyle": {"color": "#9ca3af", "type": "dashed", "width": 1.5},
                "itemStyle": {"color": "#9ca3af"},
                "symbol": "none",
            },
        ],
    }
    summary = "> 🎣 **Bad Capture Rate**\n"
    block = _artifact_block("echarts", f"{prefix}Bad Capture Rate", option)
    return f"{summary}\n{block}\n"

def render_calibration_curve(y_true, y_prob, *, prefix: str = "") -> str:
    """概率校准曲线 artifact 段（echarts 散点 + 对角线基准）。"""
    cal = ModelEvaluator.calculate_calibration_data(y_true, y_prob)
    brier = cal["brier_score"]
    option = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["预测概率", "完美校准"], "bottom": 0},
        "grid": {
            "left": "3%", "right": "4%",
            "bottom": "15%", "top": "10%",
            "containLabel": True,
        },
        "xAxis": {"type": "value", "name": "预测概率", "min": 0, "max": 1},
        "yAxis": {"type": "value", "name": "实际正样本率", "min": 0, "max": 1},
        "series": [
            {
                "name": "预测概率", "type": "scatter",
                "data": list(zip(cal["mean_predicted"], cal["fraction_positive"])),
                "symbolSize": 8,
                "itemStyle": {"color": "#6366f1"},
            },
            {
                "name": "完美校准", "type": "line",
                "data": [[0, 0], [1, 1]],
                "lineStyle": {"color": "#9ca3af", "type": "dashed"},
                "symbol": "none",
            },
        ],
    }
    summary = f"> 🎯 **概率校准曲线**（Brier Score = {brier:.4f}）\n"
    block = _artifact_block(
        "echarts", f"{prefix}校准曲线（Brier={brier:.4f}）", option
    )
    return f"{summary}\n{block}\n"

# ---------------------------------------------------------------------------
# 便捷一把梭：给 tuner/sub_trainer/stacker 这类"不在意中间文案"场景
# ---------------------------------------------------------------------------

def build_full_eval_section(
    y_true,
    y_prob,
    importance: Dict[str, float],
    *,
    prefix: str = "",
    top_n: int = 15,
) -> str:
    """顺序拼接 6 件套：KS → ROC → Lift → 特征重要性 → BCR → 校准曲线。

    这是一个 thin wrapper；如果需要在图之间插业务文案，请直接调用 render_*。
    """
    return "\n".join([
        render_ks_curve(y_true, y_prob, prefix=prefix),
        render_roc_curve(y_true, y_prob, prefix=prefix),
        render_lift_chart(y_true, y_prob, prefix=prefix),
        render_feature_importance(importance, top_n=top_n, prefix=prefix),
        render_bad_capture_rate(y_true, y_prob, prefix=prefix),
        render_calibration_curve(y_true, y_prob, prefix=prefix),
    ])

# ---------------------------------------------------------------------------
# 扩展：表格段落（评分卡风格的 IV 分箱表、Lift 明细表、BCR 明细表）
#   LR / DNN 在补齐 XGB 报告结构时复用。
# ---------------------------------------------------------------------------

def _heading(level: int, text: str) -> str:
    level = max(1, min(6, int(level)))
    return "#" * level + " " + text

def render_score_iv_table_section(
    y_true,
    y_prob,
    *,
    heading: str = "模型分数 IV 最优分箱（OOT）",
    heading_level: int = 3,
) -> str:
    """模型分数 IV 最优分箱表（带 IV 等级解读）。

    未安装 optbinning / 分箱失败时返回降级提示，不抛异常。
    """
    score_iv_table = ModelEvaluator.calculate_score_iv_table(y_true, y_prob)
    lines: List[str] = [_heading(heading_level, heading), ""]
    if score_iv_table is None or score_iv_table.empty:
        lines.append("> 未计算（optbinning 未安装或分箱失败）")
        lines.append("")
        return "\n".join(lines)

    lines.append("> 基于 OptimalBinning 对模型预测概率进行最优分箱，评估分数区分能力")
    lines.append("")
    lines.append(score_iv_table.to_markdown(index=False))
    lines.append("")

    total_iv = score_iv_table[score_iv_table["区间"] == "Total"]["IV"].values
    if len(total_iv) > 0:
        try:
            iv_val = float(total_iv[0])
        except (TypeError, ValueError):
            iv_val = 0.0
        if iv_val >= 0.5:
            iv_level = "很强"
        elif iv_val >= 0.3:
            iv_level = "较强"
        elif iv_val >= 0.1:
            iv_level = "中等"
        elif iv_val >= 0.02:
            iv_level = "较差"
        else:
            iv_level = "无预测能力"
        lines.append(f"> **模型分数 IV = {iv_val:.4f}**，预测能力 **{iv_level}**")
        lines.append("")
    return "\n".join(lines)

def render_lift_detail_table_section(
    y_true,
    y_prob,
    *,
    heading: str = "Lift 明细表（OOT）",
    heading_level: int = 3,
    n_bins: int = 10,
) -> str:
    """Lift 明细表（含累计正样本率 / 累计 Lift）+ 业务解读。"""
    import pandas as pd  # 局部引入，避免顶层依赖

    lift_table = ModelEvaluator.calculate_lift_table(y_true, y_prob, n_bins=n_bins)
    total_pos = lift_table["positive"].sum()
    total_count = lift_table["count"].sum()
    overall_rate = total_pos / total_count if total_count > 0 else 0

    lift_table = lift_table.copy()
    lift_table["cum_positive"] = lift_table["positive"].cumsum()
    lift_table["cum_positive_rate"] = lift_table["cum_positive"] / total_pos if total_pos > 0 else 0
    lift_table["cum_count"] = lift_table["count"].cumsum()
    lift_table["cum_lift"] = (
        (lift_table["cum_positive"] / lift_table["cum_count"]) / overall_rate
        if overall_rate > 0
        else 0
    )

    lines: List[str] = [_heading(heading_level, heading), ""]
    lines.append("| 分箱 | 样本数 | 正样本数 | 正样本率 | 累计正样本率 | Lift | 累计Lift |")
    lines.append("|------|--------|----------|----------|--------------|------|----------|")
    for idx, row in lift_table.iterrows():
        lines.append(
            f"| {idx} | {int(row['count'])} | {int(row['positive'])} | "
            f"{row['positive_rate']:.4%} | {row['cum_positive_rate']:.2%} | "
            f"{row['lift']:.2f} | {row['cum_lift']:.2f} |"
        )
    lines.append("")

    # 业务解读
    if len(lift_table) > 0:
        top1_lift = float(lift_table.iloc[0]["lift"])
        top1_pos_rate = float(lift_table.iloc[0]["positive_rate"])
        top2_cum_lift = float(lift_table.iloc[1]["cum_lift"]) if len(lift_table) > 1 else top1_lift
        lines.append("**Lift 解读**:")
        lines.append("")
        lines.append(f"- Top 10% 人群中正样本率为 {top1_pos_rate:.2%}，是整体的 **{top1_lift:.1f} 倍**")
        lines.append(f"- Top 20% 人群累计捕获效率为随机的 **{top2_cum_lift:.1f} 倍**")
        if top1_lift >= 3:
            lines.append("- 模型区分度良好，头部人群高度富集")
        elif top1_lift >= 2:
            lines.append("- 模型具备一定区分能力，可用于粗筛")
        else:
            lines.append("- 模型区分度较弱，建议优化特征或调参")
        lines.append("")
    return "\n".join(lines)

def render_lift_table_optbinning(
    y_true,
    y_prob,
    *,
    heading: str = "Lift 明细表（IV 最优分箱）",
    heading_level: int = 3,
    max_n_bins: int = 8,
    min_bin_size: float = 0.05,
) -> str:
    """Lift 明细表（IV 最优分箱版本）+ 业务解读。"""
    lift_table = ModelEvaluator.calculate_lift_table_optbinning(
        y_true, y_prob, max_n_bins=max_n_bins, min_bin_size=min_bin_size
    )
    if lift_table is None:
        lines: List[str] = [_heading(heading_level, heading), ""]
        lines.append("> optbinning 未安装或计算失败，无法生成 IV 最优分箱 Lift 表")
        lines.append("")
        return "\n".join(lines)

    total_pos = lift_table["positive"].sum()
    total_count = lift_table["count"].sum()
    overall_rate = total_pos / total_count if total_count > 0 else 0

    lift_table = lift_table.copy()
    lift_table["cum_positive"] = lift_table["positive"].cumsum()
    lift_table["cum_positive_rate"] = lift_table["cum_positive"] / total_pos if total_pos > 0 else 0
    lift_table["cum_count"] = lift_table["count"].cumsum()
    lift_table["cum_lift"] = (
        (lift_table["cum_positive"] / lift_table["cum_count"]) / overall_rate
        if overall_rate > 0
        else 0
    )

    lines: List[str] = [_heading(heading_level, heading), ""]
    lines.append("| 分箱 | 区间 | 样本数 | 正样本数 | 正样本率 | 累计正样本率 | Lift | 累计Lift |")
    lines.append("|------|------|--------|----------|----------|--------------|------|----------|")
    for idx, row in lift_table.iterrows():
        min_s = row["min_score"]
        max_s = row["max_score"]
        if min_s is not None and max_s is not None:
            bin_range = f"[{min_s:.4f}, {max_s:.4f})"
        elif min_s is not None:
            bin_range = f"[{min_s:.4f}, +∞)"
        elif max_s is not None:
            bin_range = f"(-∞, {max_s:.4f})"
        else:
            bin_range = "-"
        lines.append(
            f"| {idx} | {bin_range} | {int(row['count'])} | {int(row['positive'])} | "
            f"{row['positive_rate']:.4%} | {row['cum_positive_rate']:.2%} | "
            f"{row['lift']:.2f} | {row['cum_lift']:.2f} |"
        )
    lines.append("")

    # 业务解读
    if len(lift_table) > 0:
        top1_lift = float(lift_table.iloc[0]["lift"])
        top1_pos_rate = float(lift_table.iloc[0]["positive_rate"])
        top2_cum_lift = float(lift_table.iloc[1]["cum_lift"]) if len(lift_table) > 1 else top1_lift
        lines.append("**Lift 解读（IV 最优分箱）**:")
        lines.append("")
        lines.append(
            f"- 最优分箱 Top 箱正样本率为 {top1_pos_rate:.2%}，"
            f"是整体的 **{top1_lift:.1f} 倍**"
        )
        lines.append(
            f"- 前两箱累计捕获效率为随机的 **{top2_cum_lift:.1f} 倍**"
        )
        if top1_lift >= 3:
            lines.append("- 模型区分度良好，头部人群高度富集")
        elif top1_lift >= 2:
            lines.append("- 模型具备一定区分能力，可用于粗筛")
        else:
            lines.append("- 模型区分度较弱，建议优化特征或调参")
        lines.append("")
    return "\n".join(lines)

def render_bcr_detail_table_section(
    y_true,
    y_prob,
    *,
    heading: str = "坏客户捕获率（OOT）",
    heading_level: int = 3,
) -> str:
    """BCR 明细表（拒绝比例 / 坏客户捕获率 / 说明）。"""
    bcr_data = ModelEvaluator.calculate_bad_capture_rate(y_true, y_prob)
    lines: List[str] = [_heading(heading_level, heading), ""]
    lines.append("> 按得分从高到低排序，拒绝前 K% 人群能捕获多少比例的坏客户")
    lines.append("")
    lines.append("| 拒绝比例 | 坏客户捕获率 | 说明 |")
    lines.append("|---------|-------------|------|")
    for pct, rate in zip(bcr_data["topk_pcts"], bcr_data["bad_capture_rates"]):
        if rate >= 0.5:
            comment = "捕获过半坏客户"
        elif rate >= 0.3:
            comment = "良好"
        else:
            comment = ""
        lines.append(f"| 拒绝 {pct}% | {rate:.2%} | {comment} |")
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 扩展：稳定性分析段落（按月 AUC/KS 表 + Mann-Kendall 趋势 + 月度 AUC echarts + 月度 PSI 表）
#   与 xgb-modeling/modeling.py 的 "## 4. 稳定性分析" 结构对齐，供 LR / DNN 复用。
# ---------------------------------------------------------------------------

_TREND_MAP = {
    "increasing": "↑ 显著上升（p={:.3f}，tau={:.3f}）",
    "decreasing": "↓ **显著下降**（p={:.3f}，tau={:.3f}）⚠️ 建议关注",
    "no trend": "→ 无显著趋势（p={:.3f}，tau={:.3f}）",
    "insufficient data": "数据不足（至少需要 4 个月）",
    "unknown": "未计算",
}

def render_monthly_stability_section(
    stability_report: Optional[Dict[str, Any]],
    *,
    heading: str = "稳定性分析",
    heading_level: int = 2,
) -> str:
    """完整稳定性段落 —— 若 stability_report=None 则输出占位行。"""
    lines: List[str] = [_heading(heading_level, heading), ""]
    if not stability_report:
        lines.append("> 未执行稳定性分析（缺少 time_col 或时间窗口不足）")
        lines.append("")
        return "\n".join(lines)

    sub_level = heading_level + 1
    perf = stability_report.get("performance_by_month")
    if perf is not None and len(perf) > 0:
        lines.append(_heading(sub_level, "按月 AUC / KS"))
        lines.append("")
        baseline_auc = (
            perf.head(3)["auc"].mean() if len(perf) >= 3 else perf["auc"].mean()
        )
        has_ci = "auc_ci_lower" in perf.columns and perf["auc_ci_lower"].notna().any()

        if has_ci:
            lines.append("| 月份 | 样本数 | 正样本率 | AUC | 95% CI | KS | 较基线变化 | 状态 |")
            lines.append("|------|--------|----------|-----|--------|-----|------------|------|")
        else:
            lines.append("| 月份 | 样本数 | 正样本率 | AUC | KS | 较基线变化 | 状态 |")
            lines.append("|------|--------|----------|-----|-----|------------|------|")
        for idx, row in perf.iterrows():
            auc_diff = row["auc"] - baseline_auc
            status = "✅" if abs(auc_diff) < 0.02 else ("⚠️" if auc_diff < -0.03 else "✅")
            change_str = "基线" if idx < 3 else f"{auc_diff:+.4f}"
            if has_ci and not (row.get("auc_ci_lower") != row.get("auc_ci_lower")):
                ci_str = f"[{row['auc_ci_lower']:.4f}, {row['auc_ci_upper']:.4f}]"
                lines.append(
                    f"| {row['month']} | {row['samples']} | {row['positive_rate']:.4%} | "
                    f"{row['auc']:.4f} | {ci_str} | {row['ks']:.4f} | {change_str} | {status} |"
                )
            else:
                lines.append(
                    f"| {row['month']} | {row['samples']} | {row['positive_rate']:.4%} | "
                    f"{row['auc']:.4f} | {row['ks']:.4f} | {change_str} | {status} |"
                )
        lines.append("")
        lines.append(
            f"> AUC 标准差: {stability_report.get('auc_std', 0):.4f}，"
            f"KS 标准差: {stability_report.get('ks_std', 0):.4f}"
        )
        lines.append("")

        # Mann-Kendall 趋势检验
        auc_trend = stability_report.get("auc_trend", {}) or {}
        ks_trend = stability_report.get("ks_trend", {}) or {}
        lines.append("**趋势检验（Mann-Kendall）**:")
        lines.append("")
        auc_t = _TREND_MAP.get(auc_trend.get("trend", "unknown"), "未知")
        if "{" in auc_t:
            auc_t = auc_t.format(auc_trend.get("p_value", 0), auc_trend.get("tau", 0))
        lines.append(f"- AUC 趋势：{auc_t}")
        ks_t = _TREND_MAP.get(ks_trend.get("trend", "unknown"), "未知")
        if "{" in ks_t:
            ks_t = ks_t.format(ks_trend.get("p_value", 0), ks_trend.get("tau", 0))
        lines.append(f"- KS 趋势：{ks_t}")
        lines.append("")

        # 月度 AUC 趋势 echarts artifact
        months_list = perf["month"].tolist()
        auc_list = [round(float(v), 4) for v in perf["auc"].tolist()]
        series: List[Dict[str, Any]] = [
            {"name": "AUC", "type": "line", "data": auc_list, "smooth": True},
        ]
        if has_ci:
            try:
                ci_lower = perf["auc_ci_lower"].fillna(method="ffill").tolist()
                ci_upper = perf["auc_ci_upper"].fillna(method="ffill").tolist()
            except Exception:
                ci_lower, ci_upper = [], []
            if ci_upper and ci_lower:
                series.append({
                    "name": "95% CI 上界",
                    "type": "line",
                    "data": [round(float(v), 4) for v in ci_upper],
                    "lineStyle": {"opacity": 0},
                    "areaStyle": {"opacity": 0.15, "color": "#5b9bd5"},
                    "symbol": "none",
                })
                series.append({
                    "name": "95% CI 下界",
                    "type": "line",
                    "data": [round(float(v), 4) for v in ci_lower],
                    "lineStyle": {"opacity": 0},
                    "areaStyle": {"opacity": 0},
                    "symbol": "none",
                })
        option = {
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["AUC"] + (["95% CI 上界", "95% CI 下界"] if has_ci else []),
            },
            "xAxis": {"type": "category", "data": months_list},
            "yAxis": {"type": "value", "name": "AUC"},
            "series": series,
        }
        lines.append(_artifact_block("echarts", "月度 AUC 趋势（含 95% CI）", option))
        lines.append("")

    # 月度 PSI 分布表
    psi_monthly = stability_report.get("score_psi_by_month")
    if psi_monthly is not None and len(psi_monthly) > 0:
        lines.append(_heading(sub_level, "分数 PSI"))
        lines.append("")
        baseline_months = stability_report.get("baseline_months", ["Train前3月"])
        train_month_str = ", ".join(list(baseline_months)[:3])
        lines.append(f"> **PSI 基线**：{train_month_str}")
        lines.append("")
        lines.append("| 月份 | 样本数 | 平均分数 | PSI | 漂移等级 |")
        lines.append("|------|--------|----------|-----|----------|")
        for _, row in psi_monthly.iterrows():
            psi_val = float(row["psi"])
            if psi_val >= 0.25:
                drift_level = "⚠️ 显著漂移"
            elif psi_val >= 0.1:
                drift_level = "⚠️ 轻微漂移"
            else:
                drift_level = "✅ 稳定"
            lines.append(
                f"| {row['month']} | {row['samples']} | {row['mean_score']:.4f} | "
                f"{psi_val:.4f} | {drift_level} |"
            )
        lines.append("")
        overall_psi = float(stability_report.get("overall_score_psi", 0))
        lines.append(f"> 整体分数 PSI: {overall_psi:.4f}")
        lines.append("")
        if overall_psi < 0.1:
            lines.append("**PSI 解读**: 整体 PSI < 0.1，分数分布稳定")
        elif overall_psi < 0.25:
            lines.append("**PSI 解读**: 整体 PSI 处于临界范围，建议持续监控")
        else:
            lines.append("**PSI 解读**: 整体 PSI > 0.25，分数分布存在显著漂移，建议重新评估模型")
        lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 扩展：Permutation Feature Importance（DNN / 任意不可直接取 importance 的模型通用）
# ---------------------------------------------------------------------------

def calculate_permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    n_repeats: int = 3,
    sample_size: int = 2000,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Permutation Feature Importance（AUC 下降量作为重要性）。

    Args:
        predict_fn   : 接受 (n, d) ndarray 返回 (n,) 概率的函数
        X, y         : 待评估数据（通常用 val 集，避免训练集过拟合引入偏差）
        feature_names: 列名列表，长度需与 X.shape[1] 一致
        n_repeats    : 每个特征打乱次数，取平均
        sample_size  : 若样本数过大则随机下采样，控制耗时
    Returns:
        {feature_name: importance}，importance = baseline_auc - mean(shuffled_auc)
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape
    if d != len(feature_names):
        raise ValueError(
            f"feature_names 长度({len(feature_names)}) 与 X.shape[1]({d}) 不一致"
        )

    if sample_size and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X_use = X[idx]
        y_use = y[idx]
    else:
        X_use = X
        y_use = y

    baseline_prob = predict_fn(X_use)
    try:
        baseline_auc = roc_auc_score(y_use, baseline_prob)
    except ValueError:
        return {name: 0.0 for name in feature_names}

    importance: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        drops: List[float] = []
        for _ in range(max(1, n_repeats)):
            X_perm = X_use.copy()
            rng.shuffle(X_perm[:, i])  # 就地打乱第 i 列
            try:
                shuffled_auc = roc_auc_score(y_use, predict_fn(X_perm))
            except ValueError:
                shuffled_auc = baseline_auc
            drops.append(float(baseline_auc - shuffled_auc))
        importance[name] = float(np.mean(drops))

    return importance

__all__ = [
    "render_ks_curve",
    "render_roc_curve",
    "render_lift_chart",
    "render_lift_chart_optbinning",
    "render_feature_importance",
    "render_bad_capture_rate",
    "render_calibration_curve",
    "build_full_eval_section",
    "render_score_iv_table_section",
    "render_lift_detail_table_section",
    "render_lift_table_optbinning",
    "render_bcr_detail_table_section",
    "render_monthly_stability_section",
    "calculate_permutation_importance",
]
