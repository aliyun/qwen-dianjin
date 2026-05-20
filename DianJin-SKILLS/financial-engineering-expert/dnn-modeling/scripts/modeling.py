# -*- coding: utf-8 -*-
"""
DNN 深度学习建模主入口脚本

基于 PyTorch MLP 进行二分类建模。
复用平台已有的三段式数据切分、评估体系、稳定性分析。

Usage:
    python modeling.py --data_path ./data.parquet --target y_label

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from data_utils import load_and_split_data, render_split_section, DNNImputer  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402
from report_artifact import (  # noqa: E402
    render_ks_curve,
    render_roc_curve,
    render_lift_chart,
    render_lift_chart_optbinning,
    render_bad_capture_rate,
    render_feature_importance,
    render_score_iv_table_section,
    render_lift_detail_table_section,
    render_bcr_detail_table_section,
    render_monthly_stability_section,
    calculate_permutation_importance,
)
from safety_gate import SafetyGate  # noqa: E402
from xgb_core import ModelEvaluator, ModelStabilityAnalyzer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# CLI 参数
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DNN MLP Modeling")

    # 通用数据参数
    parser.add_argument("-d", "--data_path", required=True)
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("--time_col", default="busi_dt")
    parser.add_argument("--train_filter", default=None)
    parser.add_argument("--oot_filter", default=None)
    parser.add_argument("--oot_ratio", type=float, default=0.20)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--exclude_cols", default="")
    parser.add_argument("--features", default=None)

    # DNN 特有参数
    parser.add_argument("--hidden_dims", default="128,64,32", help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", default="auto", help="正样本权重，auto=自动")

    # 输出参数
    parser.add_argument("--output_dir", default=None, help="产物输出目录")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--report_output", default=None)
    parser.add_argument("--config", default=None)

    return parser

# ============================================================
# MLP 模型定义
# ============================================================

class BinaryMLP(nn.Module):
    """多层全连接网络（二分类）。

    结构：Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """输出正类概率。"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

# ============================================================
# 训练器
# ============================================================

class DNNTrainer:
    """DNN 训练器：封装训练循环、早停、学习率调度。"""

    def __init__(
        self,
        model: BinaryMLP,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        self.history: List[Dict[str, float]] = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """训练模型，返回训练信息。"""
        best_val_auc = 0.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ── Training ──
            self.model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels.append(y_batch.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_preds = np.concatenate(train_preds)
            train_labels = np.concatenate(train_labels)
            train_auc = roc_auc_score(train_labels, train_preds)

            # ── Validation ──
            self.model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_ks = float(ModelEvaluator.calculate_ks(val_labels, val_preds))

            # 学习率调度
            self.scheduler.step(val_auc)

            # 训练集 KS
            train_ks = float(ModelEvaluator.calculate_ks(train_labels, train_preds))

            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_auc": round(train_auc, 6),
                "val_auc": round(val_auc, 6),
                "train_ks": round(train_ks, 6),
                "val_ks": round(val_ks, 6),
            })

            # 早停
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                lr_current = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss={train_loss:.4f} AUC={train_auc:.4f} KS={train_ks:.4f} | "
                    f"Val Loss={val_loss:.4f} AUC={val_auc:.4f} KS={val_ks:.4f} | "
                    f"LR={lr_current:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"早停触发 @ Epoch {epoch}（最佳 Epoch {best_epoch}, Val AUC={best_val_auc:.4f}）")
                break

        # 恢复最佳权重
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_epoch": best_epoch,
            "best_val_auc": round(best_val_auc, 6),
            "total_epochs": epoch,
            "early_stopped": patience_counter >= patience,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测正类概率。"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy()
        return probs

# ============================================================
# 报告生成
# ============================================================

def generate_report(
    data_path: str,
    target: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    features: List[str],
    hidden_dims: List[int],
    training_info: Dict[str, Any],
    eval_results: Dict[str, Dict[str, float]],
    history: List[Dict[str, float]],
    stability_report: Optional[Dict[str, Any]],
    split_meta: Optional[Dict[str, Any]] = None,
    args: Any = None,
) -> str:
    """生成 DNN 建模报告。"""
    lines = []

    lines.append("# DNN 深度学习建模报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 数据文件：{data_path}")
    lines.append(f"> 目标变量：{target}")
    lines.append(f"> 算法：MLP (PyTorch)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 执行摘要
    oot_metrics = eval_results.get("oot", {})
    train_metrics = eval_results.get("train", {})
    gap = train_metrics.get("auc", 0) - oot_metrics.get("auc", 0)

    lines.append("## 执行摘要")
    lines.append("")
    lines.append(f"- **入模特征**：{len(features)} 个")
    lines.append(f"- **网络结构**：{len(features)} → {'→'.join(map(str, hidden_dims))} → 1")
    lines.append(f"- **OOT 效果**：AUC={oot_metrics.get('auc', 0):.4f}，KS={oot_metrics.get('ks', 0):.4f}")
    lines.append(f"- **泛化性**：Train-OOT Gap={gap:.4f}，"
                 f"{'无过拟合风险' if gap < 0.05 else '⚠️ 存在过拟合风险'}")
    lines.append(f"- **训练**：最佳 Epoch {training_info['best_epoch']}/{training_info['total_epochs']}，"
                 f"{'早停触发' if training_info['early_stopped'] else '跑满轮次'}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 数据概览
    lines.append("## 1. 数据概览")
    lines.append("")
    if split_meta:
        split_md = render_split_section(split_meta)
        if split_md:
            lines.append(split_md)
            lines.append("")
    lines.append("| 数据集 | 样本量 | 正样本量 | 正样本率 |")
    lines.append("|--------|--------|----------|----------|")
    for name, df in [("Train", train_data), ("Val", val_data), ("OOT", oot_data)]:
        n = len(df)
        pos = int(df[target].sum())
        rate = df[target].mean()
        lines.append(f"| {name} | {n:,} | {pos:,} | {rate:.4%} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 网络结构
    lines.append("## 2. 网络结构")
    lines.append("")
    lines.append(f"```")
    lines.append(f"Input({len(features)})")
    for i, dim in enumerate(hidden_dims):
        lines.append(f"  → Linear({dim}) → BatchNorm → ReLU → Dropout({args.dropout if args else 0.3})")
    lines.append(f"  → Linear(1) → Sigmoid")
    lines.append(f"```")
    lines.append("")
    total_params = sum(
        (len(features) if i == 0 else hidden_dims[i-1]) * hidden_dims[i] + hidden_dims[i]  # Linear + bias
        + hidden_dims[i] * 2  # BatchNorm
        for i in range(len(hidden_dims))
    ) + hidden_dims[-1] + 1  # 最后一层
    lines.append(f"- 总参数量：{total_params:,}")
    lines.append(f"- Dropout：{args.dropout if args else 0.3}")
    lines.append(f"- 学习率：{args.learning_rate if args else 0.001}")
    lines.append(f"- 权重衰减：{args.weight_decay if args else 1e-4}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 训练曲线数据（artifact 嵌入）
    lines.append("## 3. 训练曲线")
    lines.append("")
    if history:
        epochs_list = [h["epoch"] for h in history]
        train_auc_list = [h["train_auc"] for h in history]
        val_auc_list = [h["val_auc"] for h in history]
        chart_data = json.dumps({
            "labels": epochs_list,
            "series": [
                {"name": "Train AUC", "values": train_auc_list},
                {"name": "Val AUC", "values": val_auc_list},
            ]
        }, ensure_ascii=False)
        lines.append(f'<!--ARTIFACT_START--><artifact type="chart" chart-type="line" title="训练曲线 - AUC">')
        lines.append(chart_data)
        lines.append('</artifact><!--ARTIFACT_END-->')
        lines.append("")
        # KS 训练曲线
        has_ks = any(h.get("train_ks") is not None for h in history)
        if has_ks:
            train_ks_list = [h.get("train_ks", 0) or 0 for h in history]
            val_ks_list = [h.get("val_ks", 0) or 0 for h in history]
            ks_chart_data = json.dumps({
                "labels": epochs_list,
                "series": [
                    {"name": "Train KS", "values": train_ks_list},
                    {"name": "Val KS", "values": val_ks_list},
                ]
            }, ensure_ascii=False)
            lines.append(f'<!--ARTIFACT_START--><artifact type="chart" chart-type="line" title="训练曲线 - KS">')
            lines.append(ks_chart_data)
            lines.append('</artifact><!--ARTIFACT_END-->')
            lines.append("")

    # 评估指标
    lines.append("## 4. 模型评估")
    lines.append("")
    lines.append("| 指标 | Train | Val | OOT |")
    lines.append("|------|-------|-----|-----|")
    for metric in ["auc", "ks", "gini"]:
        tr = train_metrics.get(metric, 0)
        va = eval_results.get("val", {}).get(metric, 0)
        oot = oot_metrics.get(metric, 0)
        lines.append(f"| {metric.upper()} | {tr:.4f} | {va:.4f} | {oot:.4f} |")
    lines.append("")

    # 稳定性分析（完整段落：月度 AUC/KS 表 + 趋势 + echarts + 月度 PSI 表）
    lines.append(render_monthly_stability_section(
        stability_report,
        heading="5. 稳定性分析",
        heading_level=2,
    ))

    return "\n".join(lines)

# ============================================================
# 主函数
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # --config 合并
    if args.config:
        config_path = Path(args.config)
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, value)

    # ── output_dir ──
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="dnn-modeling", output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("DNN 深度学习建模开始")
    logger.info("=" * 60)

    # ── 设备选择 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"计算设备: {device}")

    # ── 随机种子 ──
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # ── 数据加载与切分 ──
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()] if args.exclude_cols else []

    split_result = load_and_split_data(
        data_path=args.data_path,
        target=args.target,
        time_col=args.time_col,
        train_filter=args.train_filter,
        oot_filter=args.oot_filter,
        oot_ratio=args.oot_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        return_features=True,
        exclude_cols=exclude_cols,
        return_meta=True,
    )
    train_data, val_data, oot_data = split_result.train, split_result.val, split_result.oot
    all_features = split_result.features
    split_meta = split_result.meta

    # ── 确定入模特征 ──
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        features = all_features
    logger.info(f"入模特征: {len(features)} 个")

    # ── SafetyGate ──
    gate = SafetyGate()
    gate_result = gate.check(train_data, val_data, oot_data, target_col=args.target, features=features)
    if not gate_result.passed:
        logger.warning(f"SafetyGate: {'; '.join(gate_result.blocking_reasons)}")

    # ── 特征缺失值处理（中位数填充 + 指示列）──
    imputer = DNNImputer(features=features)
    X_train_df, model_features = imputer.fit_transform(train_data)
    X_val_df, _ = imputer.transform(val_data)
    X_oot_df, _ = imputer.transform(oot_data)

    miss_overview = imputer.overview()
    logger.info(
        f"缺失处理: 总特征={miss_overview['total_features']}, "
        f"存在缺失={miss_overview['features_with_missing']}, "
        f"新增指示列={miss_overview['indicator_cols_added']}, "
        f"最大缺失率={miss_overview['max_miss_rate']:.2%}"
    )

    # ── 特征标准化 ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_val = scaler.transform(X_val_df.values)
    X_oot = scaler.transform(X_oot_df.values)

    y_train = train_data[args.target].values.astype(np.float32)
    y_val = val_data[args.target].values.astype(np.float32)
    y_oot = oot_data[args.target].values.astype(np.float32)

    # ── DataLoader ──
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── 正样本权重 ──
    if args.pos_weight == "auto":
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        pos_weight_val = neg_count / max(pos_count, 1)
    else:
        pos_weight_val = float(args.pos_weight)
    logger.info(f"正样本权重: {pos_weight_val:.2f}")

    # ── 构建模型 ──
    hidden_dims = [int(d.strip()) for d in args.hidden_dims.split(",")]
    model = BinaryMLP(
        input_dim=len(model_features),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )
    logger.info(f"网络结构: {len(model_features)} → {'→'.join(map(str, hidden_dims))} → 1")

    # ── 训练 ──
    trainer = DNNTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight_val,
        device=device,
    )
    training_info = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )
    logger.info(f"训练完成: Best Epoch={training_info['best_epoch']}, Val AUC={training_info['best_val_auc']:.4f}")

    # ── 预测与评估 ──
    prob_train = trainer.predict_proba(X_train)
    prob_val = trainer.predict_proba(X_val)
    prob_oot = trainer.predict_proba(X_oot)

    eval_train = ModelEvaluator.evaluate(y_train, prob_train)
    eval_val = ModelEvaluator.evaluate(y_val, prob_val)
    eval_oot = ModelEvaluator.evaluate(y_oot, prob_oot)
    eval_results = {"train": eval_train, "val": eval_val, "oot": eval_oot}

    logger.info(f"评估: Train AUC={eval_train['auc']:.4f}, Val AUC={eval_val['auc']:.4f}, OOT AUC={eval_oot['auc']:.4f}")
    logger.info(f"评估: Train KS={eval_train['ks']:.4f}, Val KS={eval_val['ks']:.4f}, OOT KS={eval_oot['ks']:.4f}")

    # ── 稳定性分析 ──
    stability_report = None
    if args.time_col and args.time_col in oot_data.columns:
        try:
            analyzer = ModelStabilityAnalyzer()
            all_data = pd.concat([train_data, val_data, oot_data], ignore_index=True)
            all_data["_dnn_score"] = np.concatenate([prob_train, prob_val, prob_oot])
            months = sorted(all_data[args.time_col].astype(str).str[:6].unique())
            if len(months) >= 2:
                baseline_months = months[:max(1, len(months) // 2)]
                stability_report = analyzer.full_stability_report(
                    all_data, args.time_col, args.target, "_dnn_score", baseline_months
                )
        except Exception as e:
            logger.warning(f"稳定性分析跳过: {e}")

    # ── 生成报告 ──
    report_content = generate_report(
        data_path=args.data_path,
        target=args.target,
        train_data=train_data,
        val_data=val_data,
        oot_data=oot_data,
        features=features,
        hidden_dims=hidden_dims,
        training_info=training_info,
        eval_results=eval_results,
        history=trainer.history,
        stability_report=stability_report,
        split_meta=split_meta,
        args=args,
    )

    # 评估图表与明细表（与 XGB 报告结构对齐）
    # DNN 特征重要性：在 val 集上做 Permutation Feature Importance
    try:
        logger.info("计算 Permutation Feature Importance ...")
        dnn_importance = calculate_permutation_importance(
            predict_fn=trainer.predict_proba,
            X=X_val,
            y=y_val,
            feature_names=list(model_features),
            n_repeats=3,
            sample_size=2000,
            random_seed=args.random_seed,
        )
    except Exception as e:
        logger.warning(f"Permutation Importance 计算失败: {e}")
        dnn_importance = {}

    report_content += "\n\n## 6. 评估图表\n\n"
    chart_renderers = [
        ("模型分数 IV 分箱", lambda: render_score_iv_table_section(
            y_oot, prob_oot, heading="6.1 模型分数 IV 最优分箱（OOT）", heading_level=3)),
        ("KS 曲线", lambda: render_ks_curve(y_oot, prob_oot, prefix="OOT ")),
        ("ROC 曲线", lambda: render_roc_curve(y_oot, prob_oot, prefix="OOT ")),
        ("Lift 明细表", lambda: render_lift_detail_table_section(
            y_oot, prob_oot, heading="6.2 Lift 明细表（OOT）", heading_level=3)),
        ("Lift Chart", lambda: render_lift_chart(y_oot, prob_oot, prefix="OOT ")),
        ("Lift Chart（IV最优分箱）", lambda: render_lift_chart_optbinning(y_oot, prob_oot, prefix="OOT ")),
        ("特征重要性", lambda: render_feature_importance(
            dnn_importance, top_n=15, prefix="DNN Permutation ")),
        ("坏客户捕获率 明细", lambda: render_bcr_detail_table_section(
            y_oot, prob_oot, heading="6.3 坏客户捕获率（OOT）", heading_level=3)),
        ("Bad Capture Rate", lambda: render_bad_capture_rate(y_oot, prob_oot, prefix="OOT ")),
    ]
    for chart_name, renderer in chart_renderers:
        try:
            report_content += renderer() + "\n"
        except Exception as e:
            logger.warning(f"图表 {chart_name} 渲染失败: {e}")
            report_content += f"> ⚠️ 图表 **{chart_name}** 渲染失败: {e}\n\n"

    # ── 保存产物 ──
    model_name = args.model_name or f"dnn_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_path = output_dir / f"{model_name}_report.md"
    report_path.write_text(report_content, encoding='utf-8')
    emitter.add_report(report_path)

    # 模型文件
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pt"

    model_bundle = {
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": len(model_features),
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "features": features,             # 原始入模特征
            "model_features": model_features, # 含缺失指示列的最终顺序
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "imputer": {
            "medians": imputer.medians_,
            "indicator_cols": imputer.indicator_cols_,
            "indicator_low": imputer.indicator_low,
            "indicator_high": imputer.indicator_high,
        },
        "training_info": training_info,
        "eval_results": eval_results,
    }
    torch.save(model_bundle, model_path)
    emitter.add_model(model_path, model_type="dnn_mlp")

    # 训练历史
    history_path = models_dir / f"{model_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.history, f, ensure_ascii=False, indent=2)
    emitter.add_file(history_path, role="asset", meta={"type": "training_history"})

    # ── State 更新 ──
    emitter.update_state({
        "model_result": {
            "algorithm": "dnn_mlp",
            "scheme_name": f"DNN MLP ({len(features)}特征)",
            "features": features,
            "n_features": len(features),
            "metrics": eval_results,
            "gap": round(eval_train["ks"] - eval_oot["ks"], 4),
            "gap_auc": round(eval_train["auc"] - eval_oot["auc"], 4),
            "psi": stability_report.get("overall_score_psi") if stability_report else None,
        },
    })

    emitter.add_artifact("metrics-table", "DNN MLP 评估指标", {
        "metrics": eval_results,
        "model_type": "dnn_mlp",
    })

    emitter.set_summary(f"DNN MLP建模完成: OOT AUC={eval_oot['auc']:.4f}, KS={eval_oot['ks']:.4f}")
    emitter.update_metrics({"oot_auc": eval_oot["auc"], "oot_ks": eval_oot["ks"], "features_count": len(features)})
    emitter.write(output_dir / "result.json")

    logger.info("=" * 60)
    logger.info("DNN 深度学习建模完成 ✓")
    logger.info(f"  模型: {model_path}")
    logger.info(f"  报告: {report_path}")
    logger.info(f"  result.json: {output_dir / 'result.json'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
