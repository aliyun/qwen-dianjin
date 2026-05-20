# -*- coding: utf-8 -*-
"""算法训练器子包。

每个训练器实现 TrainerBase 协议，产出统一的 TrainResult 契约。
"""
from trainers.base import TrainerBase, TrainResult, EvalMetrics
from trainers.xgb_trainer import XGBTrainerAdapter
from trainers.lr_trainer import LRTrainerAdapter
from trainers.dnn_trainer import DNNTrainerAdapter

# 算法注册表：key → Trainer 类
REGISTRY = {
    "xgb": XGBTrainerAdapter,
    "lr": LRTrainerAdapter,
    "dnn": DNNTrainerAdapter,
}

__all__ = [
    "TrainerBase",
    "TrainResult",
    "EvalMetrics",
    "XGBTrainerAdapter",
    "LRTrainerAdapter",
    "DNNTrainerAdapter",
    "REGISTRY",
]
