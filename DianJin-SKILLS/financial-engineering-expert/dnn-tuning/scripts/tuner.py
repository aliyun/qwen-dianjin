# -*- coding: utf-8 -*-
"""
DNN 深度学习超参调优脚本

基于 DNNTuningEngine 实现 PyTorch MLP 超参搜索，支持：
1. 交互式单轮调优（--round N）
2. AUTO 自动多轮调优（--auto）

Usage:
    # 基线
    python tuner.py --data_path ./data.parquet --target y_label

    # 自动调优
    python tuner.py --data_path ./data.parquet --target y_label --auto --max_rounds 5
"""
from __future__ import annotations

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from xgb_core import ModelEvaluator  # noqa: E402
from tuning.dnn_engine import DNNTuningEngine, DNN_PARAM_BOUNDS  # noqa: E402
from tuning.diagnose import diagnose_model, DiagnosisResult  # noqa: E402
from tuning.search_space import build_params_delta  # noqa: E402
from report_artifact import build_full_eval_section  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402
from data_utils import load_and_split_data, render_split_section, DNNImputer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# DNN 默认参数
DEFAULT_DNN_PARAMS = {
    "n_layers": 3,
    "layer_width": 128,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 512,
}

def build_dnn_parser():
    """构建 DNN 调参专用 argparse。"""
    import argparse
    parser = argparse.ArgumentParser(description="DNN 深度学习超参调优")
    # 数据参数
    parser.add_argument("--data_path", "-d", required=True, help="数据文件路径")
    parser.add_argument("--target", "-t", required=True, help="目标变量列名")
    parser.add_argument("--features", help="特征列表，逗号分隔（不传则自动推断）")
    parser.add_argument("--time_col", default="busi_dt", help="时间列名")
    parser.add_argument("--train_filter", help="训练集筛选条件")
    parser.add_argument("--val_filter", help="验证集筛选条件")
    parser.add_argument("--oot_filter", help="OOT 筛选条件")
    parser.add_argument("--oot_ratio", type=float, default=0.20, help="OOT 占比")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Val 占比")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--exclude_cols", default="", help="排除列，逗号分隔")
    # DNN 参数
    parser.add_argument("--n_layers", type=int, default=3, help="隐藏层数")
    parser.add_argument("--layer_width", type=int, default=128, help="首层宽度")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 率")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 正则化")
    parser.add_argument("--batch_size", type=int, default=512, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="完整训练 epochs")
    parser.add_argument("--search_epochs", type=int, default=30, help="搜索期间 epochs")
    # 调优控制
    parser.add_argument("--round", "-r", type=int, default=0, help="当前轮次")
    parser.add_argument("--max_rounds", type=int, default=5, help="最大调优轮数")
    parser.add_argument("--auto", action="store_true", help="自动调优模式")
    parser.add_argument("--metric", default="auc", choices=["auc", "ks"], help="优化指标")
    parser.add_argument("--prev_val_metric", type=float, help="上一轮 val 指标")
    parser.add_argument("--output_dir", help="产物输出目录")
    parser.add_argument("--model_name", help="模型名称")
    parser.add_argument("--report_output", "-o", help="报告输出路径")
    parser.add_argument("--config", help="JSON 配置文件路径")
    return parser

def build_mlp(input_dim: int, params: Dict[str, Any]) -> nn.Sequential:
    """构建递减结构 MLP。"""
    n_layers = int(params.get("n_layers", 3))
    layer_width = int(params.get("layer_width", 128))
    dropout = float(params.get("dropout", 0.3))

    hidden_dims = []
    width = layer_width
    for _ in range(n_layers):
        hidden_dims.append(max(16, width))
        width = max(16, width // 2)

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
    return nn.Sequential(*layers)

def train_dnn_model(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_eval_scaled: np.ndarray,
    y_eval: np.ndarray,
    params: Dict[str, Any],
    epochs: int = 100,
    pos_weight: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float], nn.Sequential, np.ndarray, np.ndarray]:
    """训练 DNN 模型并评估。

    Returns:
        (train_metrics, eval_metrics, model, prob_train, prob_eval)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(params.get("batch_size", 512))
    learning_rate = float(params.get("learning_rate", 0.001))
    weight_decay = float(params.get("weight_decay", 1e-4))

    input_dim = X_train_scaled.shape[1]
    model = build_mlp(input_dim, params).to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.astype(np.float32)),
    )
    eval_ds = TensorDataset(
        torch.FloatTensor(X_eval_scaled),
        torch.FloatTensor(y_eval.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_eval_auc = 0.0
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(1, epochs + 1):
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
            for X_batch, _ in eval_loader:
                logits = model(X_batch.to(device)).squeeze(-1)
                eval_preds.append(torch.sigmoid(logits).cpu().numpy())
        eval_preds = np.concatenate(eval_preds)
        eval_auc = roc_auc_score(y_eval, eval_preds)

        if eval_auc > best_eval_auc:
            best_eval_auc = eval_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 加载最优权重
    if best_state:
        model.load_state_dict(best_state)

    # 最终评估
    model.eval()
    with torch.no_grad():
        train_logits = model(torch.FloatTensor(X_train_scaled).to(device)).squeeze(-1)
        eval_logits = model(torch.FloatTensor(X_eval_scaled).to(device)).squeeze(-1)
        prob_train = torch.sigmoid(train_logits).cpu().numpy()
        prob_eval = torch.sigmoid(eval_logits).cpu().numpy()

    train_metrics = ModelEvaluator.evaluate(y_train, prob_train)
    eval_metrics = ModelEvaluator.evaluate(y_eval, prob_eval)

    return train_metrics, eval_metrics, model, prob_train, prob_eval

def generate_dnn_tuning_report(
    params: Dict[str, Any],
    train_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
    diagnosis: DiagnosisResult,
    round_num: int = 0,
) -> str:
    """生成 DNN 调优报告。"""
    lines = []
    if round_num == 0:
        lines.append("## DNN 基线模型评估")
    else:
        lines.append(f"## Round {round_num} DNN 调优结果")
    lines.append("")
    lines.append(f"**诊断**: {diagnosis.status}")
    lines.append(f"**建议**: {diagnosis.suggestion}")
    lines.append("")

    lines.append("### 当前效果")
    lines.append("")
    lines.append("| 指标 | Train | Val | Gap |")
    lines.append("|------|-------|-----|-----|")
    lines.append(f"| AUC | {train_metrics['auc']:.4f} | {eval_metrics['auc']:.4f} | {diagnosis.gap_auc:.4f} |")
    lines.append(f"| KS | {train_metrics['ks']:.4f} | {eval_metrics['ks']:.4f} | {diagnosis.gap_ks:.4f} |")
    lines.append("")

    lines.append("### 当前参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    for k in ["n_layers", "layer_width", "dropout", "learning_rate", "weight_decay", "batch_size"]:
        lines.append(f"| {k} | {params.get(k)} |")
    lines.append("")

    return "\n".join(lines)

def main(args: Optional[List[str]] = None) -> int:
    parser = build_dnn_parser()
    parsed = parser.parse_args(args)

    # 处理 --config 合并
    if parsed.config and os.path.isfile(parsed.config):
        with open(parsed.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if not hasattr(parsed, k) or getattr(parsed, k) is None:
                setattr(parsed, k, v)

    emitter = ResultEmitter()

    # ── output_dir ──
    output_dir = Path(parsed.output_dir) if parsed.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析特征
    exclude_cols = [c.strip() for c in parsed.exclude_cols.split(",") if c.strip()]
    features = None
    if parsed.features:
        features = [f.strip() for f in parsed.features.split(",")]

    # 加载数据
    split_result = load_and_split_data(
        data_path=parsed.data_path,
        target=parsed.target,
        time_col=parsed.time_col,
        train_filter=parsed.train_filter,
        test_filter=parsed.val_filter,
        oot_filter=parsed.oot_filter,
        val_ratio=parsed.val_ratio,
        oot_ratio=parsed.oot_ratio,
        random_seed=parsed.random_seed,
    )
    train_data = split_result.train
    val_data = split_result.val
    oot_data = split_result.oot
    split_meta = split_result.meta or {}

    # 自动推断特征
    if features is None:
        exclude = set(exclude_cols + [parsed.target, parsed.time_col])
        features = [c for c in train_data.columns if c not in exclude]

    X_train, y_train = train_data[features], train_data[parsed.target].values
    X_val, y_val = val_data[features], val_data[parsed.target].values
    X_oot, y_oot = oot_data[features], oot_data[parsed.target].values

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, OOT: {len(X_oot)}, Features: {len(features)}")

    # 缺失值处理（中位数填充 + 指示列）
    imputer = DNNImputer(features=features)
    X_train_filled, model_features = imputer.fit_transform(X_train)
    X_val_filled, _ = imputer.transform(X_val)
    X_oot_filled, _ = imputer.transform(X_oot)
    miss_overview = imputer.overview()
    logger.info(
        f"缺失处理: 总={miss_overview['total_features']}, 有缺失={miss_overview['features_with_missing']}, "
        f"指示列+={miss_overview['indicator_cols_added']}, max_miss={miss_overview['max_miss_rate']:.2%}"
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled.values).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_filled.values).astype(np.float32)
    X_oot_scaled = scaler.transform(X_oot_filled.values).astype(np.float32)

    # 正样本权重
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = neg_count / max(pos_count, 1)

    # 当前参数
    current_params = {
        "n_layers": parsed.n_layers,
        "layer_width": parsed.layer_width,
        "dropout": parsed.dropout,
        "learning_rate": parsed.learning_rate,
        "weight_decay": parsed.weight_decay,
        "batch_size": parsed.batch_size,
    }

    # ─── AUTO 模式 ─────────────────────────────────────────────────────
    if parsed.auto:
        logger.info(f"启用自动调优模式，最大轮数: {parsed.max_rounds}，指标: {parsed.metric}")

        best_params = current_params.copy()
        best_val_metric = 0.0
        best_oot_metric = 0.0
        prev_val_metric = parsed.prev_val_metric
        no_improvement_count = 0
        tuning_history = []
        tried_directions: List[Dict[str, Any]] = []

        for round_num in range(1, parsed.max_rounds + 1):
            logger.info(f"\n{'='*60}\n开始第 {round_num} 轮 DNN 调优\n{'='*60}")

            # 搜索期间用 search_epochs
            train_metrics, val_metrics, model, _, prob_val = train_dnn_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                current_params, epochs=parsed.search_epochs, pos_weight=pos_weight,
            )

            current_val_metric = val_metrics[parsed.metric]

            # OOT 评估
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.eval()
            with torch.no_grad():
                oot_logits = model(torch.FloatTensor(X_oot_scaled).to(device)).squeeze(-1)
                prob_oot = torch.sigmoid(oot_logits).cpu().numpy()
            oot_metrics = ModelEvaluator.evaluate(y_oot, prob_oot)
            current_oot_metric = oot_metrics[parsed.metric]

            # 诊断
            diagnosis = diagnose_model(train_metrics, val_metrics, prev_val_metric, metric=parsed.metric)

            report = generate_dnn_tuning_report(
                params=current_params,
                train_metrics=train_metrics,
                eval_metrics=val_metrics,
                diagnosis=diagnosis,
                round_num=round_num,
            )
            print("\n" + "=" * 60)
            print(report)
            print("=" * 60 + "\n")

            # 记录历史
            tuning_history.append({
                'round': round_num,
                'status': diagnosis.status,
                'val_auc': val_metrics['auc'],
                'val_ks': val_metrics['ks'],
                'oot_auc': oot_metrics['auc'],
                'oot_ks': oot_metrics['ks'],
                'gap': diagnosis.gap,
                'params': dict(current_params),
            })

            # 最优判断
            is_new_best = current_val_metric > best_val_metric
            if is_new_best:
                best_val_metric = current_val_metric
                best_oot_metric = current_oot_metric
                best_params = current_params.copy()
                no_improvement_count = 0
                logger.info(f"✓ 新最优 val {parsed.metric}: {current_val_metric:.4f}")
            else:
                no_improvement_count += 1
                logger.info(f"✗ 未改进 val {parsed.metric}: {current_val_metric:.4f} (最佳: {best_val_metric:.4f})")

            # 收敛判断
            if prev_val_metric is not None and abs(current_val_metric - prev_val_metric) < 0.001:
                logger.info("收敛：连续改进 < 0.001")
                break
            if no_improvement_count >= 2:
                logger.info("收敛：连续 2 轮未改进")
                break

            # 引擎搜索下一轮参数（传入已填充的 DataFrame，避免重复缺失处理）
            engine = DNNTuningEngine(
                X_train_filled, y_train, X_val_filled, y_val,
                base_params=current_params,
                metric=parsed.metric,
                features=model_features,
                search_epochs=parsed.search_epochs,
                already_imputed=True,
            )
            next_params, _, _, suggestion = engine.run_round(
                diagnosis, tried_directions=tried_directions, n_trials=5,
            )
            logger.info(suggestion)

            params_delta = build_params_delta(current_params, next_params)
            tried_directions.append({
                "status": diagnosis.status,
                "params_delta": params_delta,
                "improved": is_new_best,
            })

            prev_val_metric = current_val_metric
            current_params = next_params

        # 最终用最优参数 + 完整 epochs 训练
        logger.info(f"\n{'='*60}")
        logger.info(f"用最优参数进行完整训练 (epochs={parsed.epochs})")
        logger.info(f"{'='*60}\n")

        final_train_m, final_val_m, best_model, prob_train_final, prob_val_final = train_dnn_model(
            X_train_scaled, y_train, X_val_scaled, y_val,
            best_params, epochs=parsed.epochs, pos_weight=pos_weight,
        )
        best_model.eval()
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            oot_logits = best_model(torch.FloatTensor(X_oot_scaled).to(device)).squeeze(-1)
            prob_oot_final = torch.sigmoid(oot_logits).cpu().numpy()
        oot_eval = ModelEvaluator.evaluate(y_oot, prob_oot_final)

        logger.info(f"DNN 自动调优完成，最佳 val {parsed.metric}: {best_val_metric:.4f}")
        logger.info(f"完整训练后 OOT AUC: {oot_eval['auc']:.4f}, KS: {oot_eval['ks']:.4f}")
        logger.info(f"最佳参数: {json.dumps(best_params, ensure_ascii=False)}")

        # 保存模型
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _model_name = parsed.model_name or f"dnn_tuning_best_{_ts}"
        _model_path = str(models_dir / f"{_model_name}.pt")

        torch.save({
            "model_state": best_model.state_dict(),
            "params": best_params,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "features": features,
            "model_features": model_features,
            "imputer": {
                "medians": imputer.medians_,
                "indicator_cols": imputer.indicator_cols_,
            },
            "input_dim": X_train_scaled.shape[1],
        }, _model_path)
        logger.info(f"✅ 最优 DNN 模型已保存至: {_model_path}")
        emitter.add_model(_model_path, meta_path=None, feature_count=len(features), model_type="dnn_mlp")

        # 保存报告
        if parsed.report_output:
            final_report = f"# DNN 自动调优报告\n\n"
            final_report += f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
            if split_meta:
                final_report += render_split_section(split_meta) + "\n"

            final_report += "## 调优推演记录\n\n"
            final_report += "| 轮次 | 诊断 | Val AUC | Val KS | OOT AUC | OOT KS |\n"
            final_report += "|------|------|---------|--------|---------|--------|\n"
            for h in tuning_history:
                final_report += f"| {h['round']} | {h['status']} | {h['val_auc']:.4f} | {h['val_ks']:.4f} | {h['oot_auc']:.4f} | {h['oot_ks']:.4f} |\n"
            final_report += "\n"

            final_report += "## 最佳参数\n\n"
            final_report += "| 参数 | 值 |\n|------|------|\n"
            for k, v in best_params.items():
                final_report += f"| {k} | {v} |\n"
            final_report += "\n"

            final_report += "## 最终效果（完整训练）\n\n"
            final_report += f"- OOT AUC: {oot_eval['auc']:.4f}\n"
            final_report += f"- OOT KS: {oot_eval['ks']:.4f}\n"

            output_name = os.path.splitext(os.path.basename(parsed.report_output))[0]
            report_path = output_dir / f"{output_name}.md"
            report_path.write_text(final_report, encoding='utf-8')
            logger.info(f"报告已保存至: {report_path}")
            emitter.add_report(report_path)

        emitter.update_state({
            "dnn_tuning": {
                "best_params": best_params,
                "best_metrics": {"oot": {"auc": round(oot_eval['auc'], 4), "ks": round(oot_eval['ks'], 4)}},
                "rounds": len(tuning_history),
            },
        })
        emitter.update_state({
            "model_result": {
                "algorithm": "dnn_mlp",
                "scheme_name": "DNN调优最优",
                "n_features": len(features),
                "metrics": {"oot": {"auc": round(oot_eval['auc'], 4), "ks": round(oot_eval['ks'], 4)}},
                "source": "tuning",
            },
        })
        emitter.set_summary(f"DNN调优完成: OOT AUC={oot_eval['auc']:.4f}")
        emitter.update_metrics({"oot_auc": oot_eval["auc"], "oot_ks": oot_eval["ks"]})
        emitter.write(output_dir / "result.json")
        return 0

    # ─── 交互模式（单轮） ───────────────────────────────────────────────
    train_metrics, val_metrics, model, prob_train, prob_val = train_dnn_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        current_params, epochs=parsed.epochs, pos_weight=pos_weight,
    )

    # OOT 评估
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        oot_logits = model(torch.FloatTensor(X_oot_scaled).to(device)).squeeze(-1)
        prob_oot = torch.sigmoid(oot_logits).cpu().numpy()
    oot_metrics = ModelEvaluator.evaluate(y_oot, prob_oot)

    diagnosis = diagnose_model(train_metrics, val_metrics, parsed.prev_val_metric, metric=parsed.metric)

    report = generate_dnn_tuning_report(
        params=current_params,
        train_metrics=train_metrics,
        eval_metrics=val_metrics,
        diagnosis=diagnosis,
        round_num=parsed.round,
    )
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60 + "\n")

    # 保存模型
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _model_name = parsed.model_name or f"dnn_tuning_r{parsed.round}_{_ts}"
    _model_path = str(models_dir / f"{_model_name}.pt")

    torch.save({
        "model_state": model.state_dict(),
        "params": current_params,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": features,
        "model_features": model_features,
        "imputer": {
            "medians": imputer.medians_,
            "indicator_cols": imputer.indicator_cols_,
        },
        "input_dim": X_train_scaled.shape[1],
    }, _model_path)
    logger.info(f"✅ DNN 模型已保存至: {_model_path}")
    emitter.add_model(_model_path, meta_path=None, feature_count=len(features), model_type="dnn_mlp")

    emitter.update_state({
        "dnn_tuning": {
            "params": current_params,
            "metrics": {"val": val_metrics, "oot": oot_metrics},
            "diagnosis": diagnosis.status,
        },
    })
    emitter.update_state({
        "model_result": {
            "algorithm": "dnn_mlp",
            "scheme_name": f"DNN调优 R{parsed.round}",
            "n_features": len(features),
            "metrics": {"val": val_metrics, "oot": oot_metrics},
            "source": "tuning",
        },
    })
    emitter.set_summary(f"DNN单轮调优: val_{parsed.metric}={val_metrics.get(parsed.metric, 0):.4f}")
    emitter.write(output_dir / "result.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
