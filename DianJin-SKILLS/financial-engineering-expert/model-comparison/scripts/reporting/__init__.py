# -*- coding: utf-8 -*-
"""报告与可视化子包 — 多算法叠加曲线、章节、倒金字塔装配。"""
from reporting.renderer import (
    render_multi_roc_curve,
    render_multi_ks_curve,
    render_multi_lift_curve,
)
from reporting.builder import build_report

__all__ = [
    "render_multi_roc_curve",
    "render_multi_ks_curve",
    "render_multi_lift_curve",
    "build_report",
]
