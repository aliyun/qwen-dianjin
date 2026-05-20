# -*- coding: utf-8 -*-
"""LR + WoE 训练器适配。

WoE 编码 + 逻辑回归，特征重要性 = 归一化后的 |系数| × 平均 WoE 幅度。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from xgb_core import ModelEvaluator

from trainers.base import EvalMetrics, TrainResult, TrainerBase

logger = logging.getLogger(__name__)

def _to_metrics(raw: Dict[str, float]) -> EvalMetrics:
    return EvalMetrics(
        auc=float(raw.get("auc", 0.0)),
        ks=float(raw.get("ks", 0.0)),
        gini=float(raw.get("gini", 0.0)),
    )

def _lr_importance(coef: np.ndarray, feature_names: List[str]) -> List[tuple]:
    """LR 特征重要性：|系数| 降序。WoE 后所有特征同量级，直接 |coef| 即可。"""
    abs_coef = np.abs(coef.ravel())
    # 归一化到 [0, 1] 便于跨算法对比
    if abs_coef.max() > 0:
        norm = abs_coef / abs_coef.max()
    else:
        norm = abs_coef
    return sorted(zip(feature_names, norm.tolist()), key=lambda x: x[1], reverse=True)

class LRTrainerAdapter(TrainerBase):
    short = "lr"
    algorithm = "LR + WoE"
    interpretability = "强（系数 × WoE）"

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        oot_data: pd.DataFrame,
        features: List[str],
        target: str,
        config: Dict[str, Any],
    ) -> TrainResult:
        from data.woe_transform import WoEEncoder
        from sklearn.linear_model import LogisticRegression

        logger.info("[LR] 训练开始（WoE + Logistic Regression）")

        cfg = config.get("lr", {})
        max_n_bins = cfg.get("max_n_bins", 8)
        iv_threshold = cfg.get("iv_threshold", 0.02)
        regularization = cfg.get("regularization", "l2")
        C = cfg.get("C", 1.0)

        y_train = train_data[target].values
        y_val = val_data[target].values
        y_oot = oot_data[target].values

        # WoE 编码（只在 train 上拟合，防止泄漏）
        encoder = WoEEncoder(
            max_n_bins=max_n_bins,
            iv_threshold=iv_threshold,
            selection_mode="iv_filter",
        )
        encoder.fit(train_data, y_train, features)
        if not encoder.fitted_features:
            raise RuntimeError("LR: 没有特征通过 IV 筛选，请降低 iv_threshold 重试")

        X_tr = encoder.transform(train_data)
        X_va = encoder.transform(val_data)
        X_ot = encoder.transform(oot_data)

        solver = "saga" if regularization == "l1" else "lbfgs"
        model = LogisticRegression(
            penalty=regularization,
            C=C,
            solver=solver,
            max_iter=1000,
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
        )
        model.fit(X_tr, y_train)

        p_train = model.predict_proba(X_tr)[:, 1]
        p_val = model.predict_proba(X_va)[:, 1]
        p_oot = model.predict_proba(X_ot)[:, 1]

        m_train = ModelEvaluator.evaluate(y_train, p_train)
        m_val = ModelEvaluator.evaluate(y_val, p_val)
        m_oot = ModelEvaluator.evaluate(y_oot, p_oot)
        logger.info(
            "[LR] 完成 fitted=%d OOT AUC=%.4f KS=%.4f",
            len(encoder.fitted_features), m_oot["auc"], m_oot["ks"],
        )

        return TrainResult(
            algorithm=self.algorithm,
            short=self.short,
            interpretability=self.interpretability,
            n_features=len(encoder.fitted_features),
            train=_to_metrics(m_train),
            val=_to_metrics(m_val),
            oot=_to_metrics(m_oot),
            prob_train=p_train,
            prob_val=p_val,
            prob_oot=p_oot,
            model_artifact={"model": model, "woe_encoder": encoder},
            feature_importance=_lr_importance(model.coef_, encoder.fitted_features),
        )
