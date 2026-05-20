# -*- coding: utf-8 -*-
"""多算法叠加曲线渲染 — ECharts line 透传。

设计要点：
  - 直接产出 <artifact type="echarts">，前端一次性渲染多条曲线
  - ROC 对齐到统一 FPR 网格（线性插值）；KS/Lift 的 decile 坐标天然一致
  - 每条曲线的 legend 带上 AUC/KS/Lift@K 等关键指标，一眼可比
"""
from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np

from xgb_core import ModelEvaluator

# ── 艺术化样式常量（保持与前端 SERIES_COLORS 兼容视觉）─────────────
_DEFAULT_COLORS = ["#2563eb", "#f97316", "#16a34a", "#a855f7", "#ef4444"]

def _echarts_block(title: str, payload: dict) -> str:
    """包裹 artifact 标签（HTML 注释 + <artifact/>，与 report_artifact 一致）。"""
    return (
        f'<!--ARTIFACT_START--><artifact type="echarts" title="{title}">'
        + json.dumps(payload, ensure_ascii=False)
        + "</artifact><!--ARTIFACT_END-->"
    )

def _line_series(name: str, x: list, y: list, color: str) -> dict:
    return {
        "name": name,
        "type": "line",
        "smooth": True,
        "showSymbol": False,
        "lineStyle": {"width": 2, "color": color},
        "itemStyle": {"color": color},
        # 通过 xAxisIndex 默认 0 省略
        "data": [[round(float(xi), 6), round(float(yi), 6)] for xi, yi in zip(x, y)],
    }

def _base_line_option(title: str, xlabel: str, ylabel: str) -> dict:
    return {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "type": "scroll"},
        "grid": {"left": "8%", "right": "4%", "top": "15%", "bottom": "18%", "containLabel": True},
        "xAxis": {"type": "value", "name": xlabel, "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "value", "name": ylabel, "nameLocation": "middle", "nameGap": 40},
        "series": [],
    }

# ── 1) 叠加 ROC ────────────────────────────────────────────────
def _align_roc_grid(y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 50) -> Tuple[list, list, float]:
    """返回 (fpr_grid, tpr_interp, auc)，fpr 统一到 [0, 1] n_points 等分。"""
    data = ModelEvaluator.calculate_roc_data(y_true, y_prob, max_points=n_points * 2)
    fpr = np.array(data["fpr"])
    tpr = np.array(data["tpr"])
    # 保证严格单调递增以便插值
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    grid = np.linspace(0, 1, n_points)
    tpr_interp = np.interp(grid, fpr, tpr)
    return grid.tolist(), tpr_interp.tolist(), float(data["auc"])

def render_multi_roc_curve(
    y_true: np.ndarray,
    prob_map: Dict[str, np.ndarray],
    *,
    title: str = "多算法 ROC 曲线（OOT）",
) -> str:
    """叠加多算法 ROC 曲线。prob_map: {label_in_legend: prob_array}。"""
    option = _base_line_option(title, "FPR", "TPR")

    for i, (label, prob) in enumerate(prob_map.items()):
        fpr, tpr, auc = _align_roc_grid(y_true, prob)
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        option["series"].append(_line_series(
            f"{label} (AUC={auc:.4f})", fpr, tpr, color
        ))

    # 对角参考线
    option["series"].append({
        "name": "Random",
        "type": "line",
        "showSymbol": False,
        "lineStyle": {"type": "dashed", "color": "#9ca3af", "width": 1},
        "data": [[0, 0], [1, 1]],
    })

    summary = f"> 📈 **多算法 ROC 叠加**（{len(prob_map)} 条曲线）\n"
    return f"{summary}\n{_echarts_block(title, option)}\n"

# ── 2) 叠加 KS ─────────────────────────────────────────────────
def render_multi_ks_curve(
    y_true: np.ndarray,
    prob_map: Dict[str, np.ndarray],
    *,
    title: str = "多算法 KS 曲线（OOT）",
) -> str:
    """叠加多算法 KS 曲线（x = 分位十分位 0~10）。"""
    option = _base_line_option(title, "分位（Decile）", "累计占比")
    option["xAxis"] = {
        "type": "category",
        "data": list(range(0, 11)),
        "name": "分位（Decile）",
        "nameLocation": "middle",
        "nameGap": 30,
    }
    option["yAxis"] = {"type": "value", "name": "累计占比", "nameLocation": "middle", "nameGap": 40, "max": 1.0}
    option["series"] = []

    for i, (label, prob) in enumerate(prob_map.items()):
        data = ModelEvaluator.calculate_ks_curve_data(y_true, prob)
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        # 两条：累计 Bad / 累计 Good
        option["series"].append({
            "name": f"{label} 累计坏 (KS={data['ks_value']:.4f})",
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 2, "color": color},
            "itemStyle": {"color": color},
            "data": data["cum_pos_rate"],
        })
        option["series"].append({
            "name": f"{label} 累计好",
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 1.5, "color": color, "type": "dashed"},
            "itemStyle": {"color": color},
            "data": data["cum_neg_rate"],
        })

    summary = f"> 📉 **多算法 KS 叠加**（实线=累计坏，虚线=累计好）\n"
    return f"{summary}\n{_echarts_block(title, option)}\n"

# ── 3) 叠加 Lift ───────────────────────────────────────────────
def render_multi_lift_curve(
    y_true: np.ndarray,
    prob_map: Dict[str, np.ndarray],
    *,
    title: str = "多算法 Lift 曲线（OOT 累计）",
) -> str:
    """叠加多算法的累计 Lift（x = 1..10 分位）。"""
    option = _base_line_option(title, "分位（Decile）", "累计 Lift")
    option["xAxis"] = {
        "type": "category",
        "data": list(range(1, 11)),
        "name": "分位（Decile）",
        "nameLocation": "middle",
        "nameGap": 30,
    }

    for i, (label, prob) in enumerate(prob_map.items()):
        data = ModelEvaluator.calculate_lift_chart_data(y_true, prob)
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        lift_top1 = data["lift"][0] if data["lift"] else 0.0
        option["series"].append({
            "name": f"{label} (Lift@10%={lift_top1:.2f})",
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 2, "color": color},
            "itemStyle": {"color": color},
            "data": data["cum_lift"],
        })

    # 基准线 1.0
    option["series"].append({
        "name": "Baseline",
        "type": "line",
        "showSymbol": False,
        "lineStyle": {"type": "dashed", "color": "#9ca3af", "width": 1},
        "data": [1.0] * 10,
    })

    summary = "> 📊 **多算法累计 Lift 叠加**（基线=1.0）\n"
    return f"{summary}\n{_echarts_block(title, option)}\n"
