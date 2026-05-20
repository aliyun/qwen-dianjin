# -*- coding: utf-8 -*-
"""DNN (MLP) 训练器适配。

关键点：
  - 缺失值处理遵循《DNN 数据预处理规范》：DNNImputer（中位数 + 阈值指示列）
  - 不提供 feature_importance（permutation 太贵；需要时由上层单独触发）
  - 固定使用 CPU 以避免 GPU 差异，若有 GPU 自动切换
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from xgb_core import ModelEvaluator
from data_utils import DNNImputer

from trainers.base import EvalMetrics, TrainResult, TrainerBase

logger = logging.getLogger(__name__)

# ── 网络：独立函数，方便测试 ──────────────────────────────────────
def _build_mlp(input_dim: int, hidden_dims: List[int], dropout: float):
    import torch.nn as nn
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

def _to_metrics(raw: Dict[str, float]) -> EvalMetrics:
    return EvalMetrics(
        auc=float(raw.get("auc", 0.0)),
        ks=float(raw.get("ks", 0.0)),
        gini=float(raw.get("gini", 0.0)),
    )

class DNNTrainerAdapter(TrainerBase):
    short = "dnn"
    algorithm = "DNN (MLP)"
    interpretability = "弱"

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        oot_data: pd.DataFrame,
        features: List[str],
        target: str,
        config: Dict[str, Any],
    ) -> TrainResult:
        import torch
        import torch.nn as nn
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        cfg = config.get("dnn", {})
        hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
        dropout = cfg.get("dropout", 0.3)
        lr = cfg.get("learning_rate", 0.001)
        batch_size = cfg.get("batch_size", 512)
        epochs = cfg.get("epochs", 100)
        patience = cfg.get("patience", 10)
        weight_decay = cfg.get("weight_decay", 1e-4)

        logger.info("[DNN] 训练开始 hidden=%s dropout=%s", hidden_dims, dropout)

        # 1. 缺失值处理（中位数 + 阈值指示列）
        imputer = DNNImputer(features=features)
        X_tr_df, model_features = imputer.fit_transform(train_data)
        X_va_df, _ = imputer.transform(val_data)
        X_ot_df, _ = imputer.transform(oot_data)

        # 2. 标准化（仅用 train 拟合）
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_df.values).astype(np.float32)
        X_va = scaler.transform(X_va_df.values).astype(np.float32)
        X_ot = scaler.transform(X_ot_df.values).astype(np.float32)

        y_tr = train_data[target].values.astype(np.float32)
        y_va = val_data[target].values.astype(np.float32)
        y_ot = oot_data[target].values.astype(np.float32)

        pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1e-6)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _build_mlp(len(model_features), hidden_dims, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
            batch_size=batch_size, shuffle=True,
        )

        # 3. 训练循环 + 早停
        best_auc, best_state, wait = 0.0, None, 0
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(-1), yb)
                loss.backward()
                optimizer.step()

            # val
            model.eval()
            with torch.no_grad():
                val_prob = torch.sigmoid(
                    model(torch.FloatTensor(X_va).to(device)).squeeze(-1)
                ).cpu().numpy()
            val_auc = roc_auc_score(y_va, val_prob)
            scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                logger.info("[DNN] 早停 @epoch=%d best_val_auc=%.4f", ep, best_auc)
                break

        if best_state:
            model.load_state_dict(best_state)

        # 4. 预测
        model.eval()
        with torch.no_grad():
            def _pred(X):
                return torch.sigmoid(
                    model(torch.FloatTensor(X).to(device)).squeeze(-1)
                ).cpu().numpy()
            p_train = _pred(X_tr)
            p_val = _pred(X_va)
            p_oot = _pred(X_ot)

        m_train = ModelEvaluator.evaluate(y_tr, p_train)
        m_val = ModelEvaluator.evaluate(y_va, p_val)
        m_oot = ModelEvaluator.evaluate(y_ot, p_oot)
        logger.info("[DNN] 完成 OOT AUC=%.4f KS=%.4f", m_oot["auc"], m_oot["ks"])

        return TrainResult(
            algorithm=self.algorithm,
            short=self.short,
            interpretability=self.interpretability,
            n_features=len(model_features),
            train=_to_metrics(m_train),
            val=_to_metrics(m_val),
            oot=_to_metrics(m_oot),
            prob_train=p_train,
            prob_val=p_val,
            prob_oot=p_oot,
            model_artifact={
                "model": model,
                "scaler": scaler,
                "imputer": imputer,
                "model_features": model_features,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
            },
            feature_importance=None,
        )
