# -*- coding: utf-8 -*-
"""
xgb_core —— XGBoost 建模通用能力库

该包承载原本散落在 xgb-modeling skill 内的核心类（训练、评估、稳定性），
通过 python_executor 预置的 PYTHONPATH 对所有 skill（含 custom/）统一可见，
以消除跨 skill 的 sys.path.insert 反模式。

使用方式：
    from xgb_core import XGBTrainer, ModelEvaluator, ModelStabilityAnalyzer
"""
from .trainer import XGBTrainer
from .evaluator import ModelEvaluator
from .stability import ModelStabilityAnalyzer

__all__ = [
    "XGBTrainer",
    "ModelEvaluator",
    "ModelStabilityAnalyzer",
]
