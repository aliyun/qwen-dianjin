# -*- coding: utf-8 -*-
"""XGBoost 训练器适配。

复用 xgb_core.XGBTrainer，只做"产出 TrainResult"的编排。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from xgb_core import XGBTrainer, ModelEvaluator

from trainers.base import EvalMetrics, TrainResult, TrainerBase

logger = logging.getLogger(__name__)

def _to_metrics(raw: Dict[str, float]) -> EvalMetrics:
    """xgb_core.evaluate() 返回 dict → EvalMetrics。高阶指标留空，后续补全。"""
    return EvalMetrics(
        auc=float(raw.get("auc", 0.0)),
        ks=float(raw.get("ks", 0.0)),
        gini=float(raw.get("gini", 0.0)),
    )

class XGBTrainerAdapter(TrainerBase):
    short = "xgb"
    algorithm = "XGBoost"
    interpretability = "中（SHAP 可解释）"

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        oot_data: pd.DataFrame,
        features: List[str],
        target: str,
        config: Dict[str, Any],
    ) -> TrainResult:
        logger.info("[XGB] 训练开始 (features=%d)", len(features))

        X_train, y_train = train_data[features], train_data[target]
        X_val, y_val = val_data[features], val_data[target]
        X_oot, y_oot = oot_data[features], oot_data[target]

        params = config.get("xgb") or None
        trainer = XGBTrainer(params)
        trainer.train(X_train, y_train, X_val, y_val)

        p_train = trainer.predict_proba(X_train)
        p_val = trainer.predict_proba(X_val)
        p_oot = trainer.predict_proba(X_oot)

        # 特征重要性：xgb importance → [(feat, score)] 降序
        imp_map = trainer.get_feature_importance(sort=True) or {}
        fi = list(imp_map.items())  # 已按降序排序

        m_train = ModelEvaluator.evaluate(y_train, p_train)
        m_val = ModelEvaluator.evaluate(y_val, p_val)
        m_oot = ModelEvaluator.evaluate(y_oot, p_oot)
        logger.info("[XGB] 完成 OOT AUC=%.4f KS=%.4f", m_oot["auc"], m_oot["ks"])

        return TrainResult(
            algorithm=self.algorithm,
            short=self.short,
            interpretability=self.interpretability,
            n_features=len(features),
            train=_to_metrics(m_train),
            val=_to_metrics(m_val),
            oot=_to_metrics(m_oot),
            prob_train=p_train,
            prob_val=p_val,
            prob_oot=p_oot,
            model_artifact=trainer,
            feature_importance=fi,
        )
