# -*- coding: utf-8 -*-
"""DNN 调参引擎 — DNNTuningEngine。

继承 BaseTuningEngine，实现 PyTorch MLP 的超参搜索：
  - 网络架构（层数、宽度、dropout）
  - 训练参数（learning_rate、weight_decay、batch_size）

搜索期间使用缩减 epochs（加速），最终训练使用完整 epochs。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tuning.engine import BaseTuningEngine
from tuning.diagnose import DiagnosisResult, _DIAG_THRESHOLDS

logger = logging.getLogger(__name__)

# DNN 参数硬边界
DNN_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "n_layers": (2, 4),
    "layer_width": (32, 256),
    "dropout": (0.1, 0.5),
    "learning_rate": (1e-4, 0.01),
    "weight_decay": (1e-5, 1e-3),
}

DNN_BATCH_SIZE_CHOICES = [128, 256, 512, 1024]

def _clip_dnn(value: float, key: str) -> float:
    lo, hi = DNN_PARAM_BOUNDS[key]
    return max(lo, min(hi, value))

class DNNTuningEngine(BaseTuningEngine):
    """PyTorch MLP 调参引擎。

    搜索空间：
      - n_layers: 整数 2-4（隐藏层数）
      - layer_width: 整数 32-256（各层宽度基准，递减结构）
      - dropout: 线性 0.1-0.5
      - learning_rate: 对数 1e-4 ~ 0.01
      - weight_decay: 对数 1e-5 ~ 1e-3
      - batch_size: 分类 128/256/512/1024

    Args:
        X_train/y_train: 原始训练数据（StandardScaler 在 __init__ 内部做）
        X_eval/y_eval: 原始验证数据
        base_params: 基线参数
        metric: "auc" 或 "ks"
        features: 特征列表
        search_epochs: 搜索期间缩减的 epochs 数（默认 30）
        already_imputed: 调用方是否已完成缺失值填充（True 跳过内部 imputer）
    """

    INT_PARAMS = {"n_layers", "layer_width"}
    LOG_PARAMS = {"learning_rate", "weight_decay"}
    CAT_PARAMS = {"batch_size": DNN_BATCH_SIZE_CHOICES}

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_eval: pd.DataFrame,
        y_eval: np.ndarray,
        base_params: Dict[str, Any],
        metric: str = "auc",
        features: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        search_epochs: int = 30,
        already_imputed: bool = False,
    ) -> None:
        super().__init__(X_train, y_train, X_eval, y_eval, base_params, metric, sample_weight)
        self.features = features or list(X_train.columns)
        self.search_epochs = search_epochs

        # ── 缺失值处理：遵循 DNN 缺失值处理规范（中位数填充 + 指示列）──
        if already_imputed:
            # 调用方已处理（如 tuner.py），直接取值
            train_filled = X_train[self.features]
            eval_filled = X_eval[self.features]
            self.imputer = None
        else:
            # 独立使用场景：引擎内部自动做缺失处理
            from data.missing import DNNImputer
            self.imputer = DNNImputer(features=self.features)
            train_filled, model_features = self.imputer.fit_transform(X_train)
            eval_filled, _ = self.imputer.transform(X_eval)
            # 更新特征列表为含指示列的最终顺序
            self.features = model_features

        # 预先标准化（避免每次 _evaluate 重复计算）
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(
            train_filled.values
        ).astype(np.float32)
        self.X_eval_scaled = self.scaler.transform(
            eval_filled.values
        ).astype(np.float32)

        # 正样本权重
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        self.pos_weight = neg_count / max(pos_count, 1)

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """构建 MLP + 训练 + 评估。"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        n_layers = int(params.get("n_layers", 3))
        layer_width = int(params.get("layer_width", 128))
        dropout = float(params.get("dropout", 0.3))
        learning_rate = float(params.get("learning_rate", 0.001))
        weight_decay = float(params.get("weight_decay", 1e-4))
        batch_size = int(params.get("batch_size", 512))

        # 构造递减 hidden_dims
        hidden_dims = []
        width = layer_width
        for i in range(n_layers):
            hidden_dims.append(max(16, width))
            width = max(16, width // 2)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_dim = self.X_train_scaled.shape[1]

        # 构建模型
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        model = nn.Sequential(*layers).to(device)

        # DataLoader
        train_ds = TensorDataset(
            torch.FloatTensor(self.X_train_scaled),
            torch.FloatTensor(self.y_train.astype(np.float32)),
        )
        eval_ds = TensorDataset(
            torch.FloatTensor(self.X_eval_scaled),
            torch.FloatTensor(self.y_eval.astype(np.float32)),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

        # 训练
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=device)
        )

        best_eval_auc = 0.0
        patience_counter = 0
        patience = 5  # 搜索期间用较小的 patience

        for epoch in range(1, self.search_epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch).squeeze(-1), y_batch)
                loss.backward()
                optimizer.step()

            # Eval
            model.eval()
            eval_preds = []
            with torch.no_grad():
                for X_batch, y_batch in eval_loader:
                    logits = model(X_batch.to(device)).squeeze(-1)
                    eval_preds.append(torch.sigmoid(logits).cpu().numpy())
            eval_preds = np.concatenate(eval_preds)
            eval_auc = roc_auc_score(self.y_eval, eval_preds)

            if eval_auc > best_eval_auc:
                best_eval_auc = eval_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # 返回评估指标
        if self.metric == "ks":
            from xgb_core import ModelEvaluator
            # 获取最终预测
            model.eval()
            with torch.no_grad():
                train_logits = model(torch.FloatTensor(self.X_train_scaled).to(device)).squeeze(-1)
                eval_logits = model(torch.FloatTensor(self.X_eval_scaled).to(device)).squeeze(-1)
                prob_train = torch.sigmoid(train_logits).cpu().numpy()
                prob_eval = torch.sigmoid(eval_logits).cpu().numpy()

            train_ks = float(ModelEvaluator.calculate_ks(self.y_train, prob_train))
            eval_ks = float(ModelEvaluator.calculate_ks(self.y_eval, prob_eval))
            gap = train_ks - eval_ks
            threshold = _DIAG_THRESHOLDS["ks"]["overfit_gap"]
            if gap > threshold:
                return 0.0
            return max(eval_ks, 0.0)
        else:
            return max(best_eval_auc, 0.0)

    def _get_default_space(self) -> Dict[str, Tuple[float, float]]:
        """DNN 全量硬边界搜索空间。"""
        return dict(DNN_PARAM_BOUNDS)

    def _build_constrained_space(
        self,
        diagnosis: DiagnosisResult,
        tried_directions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """DNN 诊断驱动的约束搜索空间。

        过拟合 → dropout 抬高、weight_decay 抬高、n_layers 减少
        欠拟合 → layer_width 增大、n_layers 增大、dropout 降低
        拟合良好 → 全参数 ±20% 微调
        """
        tried_directions = tried_directions or []
        space: Dict[str, Tuple[float, float]] = {}

        n_layers = self.base_params.get("n_layers", 3)
        layer_width = self.base_params.get("layer_width", 128)
        dropout_val = self.base_params.get("dropout", 0.3)
        lr_val = self.base_params.get("learning_rate", 0.001)
        wd_val = self.base_params.get("weight_decay", 1e-4)

        status = diagnosis.status

        if status == "过拟合":
            space["n_layers"] = (_clip_dnn(max(2, n_layers - 1), "n_layers"), _clip_dnn(n_layers, "n_layers"))
            space["layer_width"] = (_clip_dnn(max(32, layer_width // 2), "layer_width"), _clip_dnn(layer_width, "layer_width"))
            space["dropout"] = (_clip_dnn(dropout_val, "dropout"), _clip_dnn(min(0.5, dropout_val + 0.15), "dropout"))
            space["learning_rate"] = (_clip_dnn(lr_val * 0.5, "learning_rate"), _clip_dnn(lr_val, "learning_rate"))
            space["weight_decay"] = (_clip_dnn(wd_val, "weight_decay"), _clip_dnn(wd_val * 5, "weight_decay"))
        elif status == "轻微过拟合":
            space["n_layers"] = (_clip_dnn(max(2, n_layers - 1), "n_layers"), _clip_dnn(n_layers, "n_layers"))
            space["layer_width"] = (_clip_dnn(max(32, layer_width * 0.7), "layer_width"), _clip_dnn(layer_width, "layer_width"))
            space["dropout"] = (_clip_dnn(dropout_val, "dropout"), _clip_dnn(min(0.5, dropout_val + 0.1), "dropout"))
            space["learning_rate"] = (_clip_dnn(lr_val * 0.7, "learning_rate"), _clip_dnn(lr_val * 1.2, "learning_rate"))
            space["weight_decay"] = (_clip_dnn(wd_val, "weight_decay"), _clip_dnn(wd_val * 3, "weight_decay"))
        elif status == "欠拟合":
            space["n_layers"] = (_clip_dnn(n_layers, "n_layers"), _clip_dnn(min(4, n_layers + 1), "n_layers"))
            space["layer_width"] = (_clip_dnn(layer_width, "layer_width"), _clip_dnn(min(256, layer_width * 2), "layer_width"))
            space["dropout"] = (_clip_dnn(max(0.1, dropout_val - 0.1), "dropout"), _clip_dnn(dropout_val, "dropout"))
            space["learning_rate"] = (_clip_dnn(lr_val, "learning_rate"), _clip_dnn(lr_val * 3, "learning_rate"))
            space["weight_decay"] = (_clip_dnn(wd_val * 0.3, "weight_decay"), _clip_dnn(wd_val, "weight_decay"))
        else:
            # 拟合良好 / 收敛 → 微调
            space["n_layers"] = (_clip_dnn(max(2, n_layers - 1), "n_layers"), _clip_dnn(min(4, n_layers + 1), "n_layers"))
            space["layer_width"] = (_clip_dnn(max(32, layer_width * 0.8), "layer_width"), _clip_dnn(min(256, layer_width * 1.3), "layer_width"))
            space["dropout"] = (_clip_dnn(max(0.1, dropout_val - 0.05), "dropout"), _clip_dnn(min(0.5, dropout_val + 0.05), "dropout"))
            space["learning_rate"] = (_clip_dnn(lr_val * 0.7, "learning_rate"), _clip_dnn(lr_val * 1.5, "learning_rate"))
            space["weight_decay"] = (_clip_dnn(wd_val * 0.5, "weight_decay"), _clip_dnn(wd_val * 2, "weight_decay"))

        # 根据 tried_directions 冻结已尝试无效的方向
        for record in tried_directions:
            if record.get("improved"):
                continue
            delta = record.get("params_delta", {}) or {}
            for key, d in delta.items():
                if key not in space:
                    continue
                cur = self.base_params.get(key)
                if cur is None:
                    continue
                lo, hi = space[key]
                if d > 0:
                    space[key] = (lo, min(hi, float(cur)))
                elif d < 0:
                    space[key] = (max(lo, float(cur)), hi)

        return space

__all__ = ["DNNTuningEngine", "DNN_PARAM_BOUNDS"]
