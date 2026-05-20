# -*- coding: utf-8 -*-
"""shared.evaluation — 业务评估层。

基于 xgb_core.ModelEvaluator（纯计算原语）之上，提供：
  - MetricsBundle: KS 为主的评估结构（带 Bootstrap CI）
  - evaluate_with_ci: 统一评估入口
  - bootstrap: Bootstrap 置信区间工具
  - significance: 统计显著性检验（从 auto-experiment 下沉）
"""
from .metrics import MetricsBundle, evaluate_with_ci
from .bootstrap import bootstrap_ks_ci, bootstrap_auc_ci, paired_bootstrap_delta_ci
from .significance import SignificanceTester, CrossValidationTester

__all__ = [
    "MetricsBundle",
    "evaluate_with_ci",
    "bootstrap_ks_ci",
    "bootstrap_auc_ci",
    "paired_bootstrap_delta_ci",
    "SignificanceTester",
    "CrossValidationTester",
]
