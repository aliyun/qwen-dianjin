# -*- coding: utf-8 -*-
"""
LR 评分卡超参调优脚本

基于 LRTuningEngine 实现 WoE+LR 联合调参，支持：
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from xgb_core import ModelEvaluator  # noqa: E402
from tuning.lr_engine import LRTuningEngine, LR_PARAM_BOUNDS  # noqa: E402
from tuning.diagnose import diagnose_model, DiagnosisResult  # noqa: E402
from tuning.search_space import build_params_delta  # noqa: E402
from report_artifact import build_full_eval_section  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402
from data_utils import load_and_split_data, render_split_section  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# LR 默认参数
DEFAULT_LR_PARAMS = {
    "max_n_bins": 8,
    "iv_threshold": 0.02,
    "C": 1.0,
    "regularization": "l2",
}

def build_lr_parser():
    """构建 LR 调参专用 argparse。"""
    import argparse
    parser = argparse.ArgumentParser(description="LR 评分卡超参调优")
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
    # LR 参数
    parser.add_argument("--max_n_bins", type=int, default=8, help="WoE 分箱数")
    parser.add_argument("--iv_threshold", type=float, default=0.02, help="IV 阈值")
    parser.add_argument("--C", type=float, default=1.0, help="正则化强度倒数")
    parser.add_argument("--regularization", default="l2",
                        choices=["l1", "l2", "elasticnet"], help="正则化类型")
    # 调优控制
    parser.add_argument("--round", "-r", type=int, default=0, help="当前轮次")
    parser.add_argument("--max_rounds", type=int, default=5, help="最大调优轮数")
    parser.add_argument("--auto", action="store_true", help="自动调优模式")
    parser.add_argument("--metric", default="auc", choices=["auc", "ks"], help="优化指标")
    parser.add_argument("--prev_val_metric", type=float, help="上一轮 val 指标")
    parser.add_argument("--model_name", help="模型名称")
    parser.add_argument("--report_output", "-o", help="报告输出路径")
    parser.add_argument("--config", help="JSON 配置文件路径")
    return parser

def train_lr_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    features: List[str],
    params: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Any, Any, Any]:
    """训练 WoE+LR 模型并评估。

    Returns:
        (train_metrics, eval_metrics, model, woe_encoder, selected_features)
    """
    from data.woe_transform import WoEEncoder

    max_n_bins = int(params.get("max_n_bins", 8))
    iv_threshold = float(params.get("iv_threshold", 0.02))
    C = float(params.get("C", 1.0))
    regularization = params.get("regularization", "l2")

    # WoE 编码
    woe_encoder = WoEEncoder(
        max_n_bins=max_n_bins,
        iv_threshold=iv_threshold,
        selection_mode="iv_filter",
    )
    woe_encoder.fit(X_train, y_train, features)

    if not woe_encoder.fitted_features:
        raise ValueError(f"所有特征 IV < {iv_threshold}，无法训练。请降低 iv_threshold。")

    X_train_woe = woe_encoder.transform(X_train)
    X_eval_woe = woe_encoder.transform(X_eval)

    # LR 训练
    solver = "saga" if regularization in ("l1", "elasticnet") else "lbfgs"
    lr_params = {
        "penalty": regularization,
        "C": C,
        "solver": solver,
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
    }
    if regularization == "elasticnet":
        lr_params["l1_ratio"] = 0.5

    model = LogisticRegression(**lr_params)
    model.fit(X_train_woe, y_train)

    # 评估
    prob_train = model.predict_proba(X_train_woe)[:, 1]
    prob_eval = model.predict_proba(X_eval_woe)[:, 1]

    train_metrics = ModelEvaluator.evaluate(y_train, prob_train)
    eval_metrics = ModelEvaluator.evaluate(y_eval, prob_eval)

    return train_metrics, eval_metrics, model, woe_encoder, woe_encoder.fitted_features

def generate_lr_tuning_report(
    params: Dict[str, Any],
    train_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
    diagnosis: DiagnosisResult,
    n_features_selected: int,
    round_num: int = 0,
) -> str:
    """生成 LR 调优报告。"""
    lines = []
    if round_num == 0:
        lines.append("## LR 基线模型评估")
    else:
        lines.append(f"## Round {round_num} LR 调优结果")
    lines.append("")
    lines.append(f"**诊断**: {diagnosis.status}")
    lines.append(f"**建议**: {diagnosis.suggestion}")
    lines.append(f"**入模特征数**: {n_features_selected}")
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
    for k in ["max_n_bins", "iv_threshold", "C", "regularization"]:
        lines.append(f"| {k} | {params.get(k)} |")
    lines.append("")

    return "\n".join(lines)

def main(args: Optional[List[str]] = None) -> int:
    parser = build_lr_parser()
    parser.add_argument('--output_dir', default=None, help='产物输出目录')
    parsed = parser.parse_args(args)

    # 处理 --config 合并
    if parsed.config and os.path.isfile(parsed.config):
        with open(parsed.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if not hasattr(parsed, k) or getattr(parsed, k) is None:
                setattr(parsed, k, v)

    # output_dir
    if getattr(parsed, 'output_dir', None):
        output_dir = Path(parsed.output_dir).resolve()
    else:
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / "outputs" / _ts).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="lr-tuning", output_dir=output_dir)

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

    # 当前参数
    current_params = {
        "max_n_bins": parsed.max_n_bins,
        "iv_threshold": parsed.iv_threshold,
        "C": parsed.C,
        "regularization": parsed.regularization,
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
            logger.info(f"\n{'='*60}\n开始第 {round_num} 轮 LR 调优\n{'='*60}")

            # 训练
            try:
                train_metrics, val_metrics, model, woe_enc, sel_feats = train_lr_model(
                    X_train, y_train, X_val, y_val, features, current_params
                )
            except ValueError as e:
                logger.error(f"训练失败: {e}")
                break

            current_val_metric = val_metrics[parsed.metric]

            # OOT 评估（仅用于汇报）
            X_oot_woe = woe_enc.transform(X_oot)
            prob_oot = model.predict_proba(X_oot_woe)[:, 1]
            oot_metrics = ModelEvaluator.evaluate(y_oot, prob_oot)
            current_oot_metric = oot_metrics[parsed.metric]

            # 诊断
            diagnosis = diagnose_model(train_metrics, val_metrics, prev_val_metric, metric=parsed.metric)

            report = generate_lr_tuning_report(
                params=current_params,
                train_metrics=train_metrics,
                eval_metrics=val_metrics,
                diagnosis=diagnosis,
                n_features_selected=len(sel_feats),
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
                'n_features': len(sel_feats),
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

            # 引擎搜索下一轮参数
            engine = LRTuningEngine(
                X_train, y_train, X_val, y_val,
                base_params=current_params,
                metric=parsed.metric,
                features=features,
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

        # 最终结果
        logger.info(f"\n{'='*60}")
        logger.info(f"LR 自动调优完成，最佳 val {parsed.metric}: {best_val_metric:.4f} (OOT: {best_oot_metric:.4f})")
        logger.info(f"最佳参数: {json.dumps(best_params, ensure_ascii=False)}")
        logger.info(f"{'='*60}\n")

        # 用最优参数重新训练
        _, _, best_model, best_woe, best_feats = train_lr_model(
            X_train, y_train, X_val, y_val, features, best_params
        )
        prob_oot_best = best_model.predict_proba(best_woe.transform(X_oot))[:, 1]
        oot_eval = ModelEvaluator.evaluate(y_oot, prob_oot_best)

        # 保存模型
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _model_name = parsed.model_name or f"lr_tuning_best_{_ts}"

        import joblib
        _model_path = str(models_dir / f"{_model_name}.pkl")
        joblib.dump({"model": best_model, "woe_encoder": best_woe, "features": best_feats, "params": best_params}, _model_path)
        logger.info(f"✅ 最优 LR 模型已保存至: {_model_path}")
        emitter.add_model(_model_path, meta_path=None, feature_count=len(best_feats), model_type="lr_scorecard")

        # 保存报告
        if parsed.report_output:
            final_report = f"# LR 评分卡自动调优报告\n\n"
            final_report += f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
            if split_meta:
                final_report += render_split_section(split_meta) + "\n"

            final_report += "## 调优推演记录\n\n"
            final_report += "| 轮次 | 诊断 | Val AUC | Val KS | OOT AUC | OOT KS | 入模特征 |\n"
            final_report += "|------|------|---------|--------|---------|--------|----------|\n"
            for h in tuning_history:
                final_report += f"| {h['round']} | {h['status']} | {h['val_auc']:.4f} | {h['val_ks']:.4f} | {h['oot_auc']:.4f} | {h['oot_ks']:.4f} | {h['n_features']} |\n"
            final_report += "\n"

            final_report += "## 最佳参数\n\n"
            final_report += "| 参数 | 值 |\n|------|------|\n"
            for k, v in best_params.items():
                final_report += f"| {k} | {v} |\n"
            final_report += "\n"

            final_report += "## 最终效果\n\n"
            final_report += f"- OOT AUC: {oot_eval['auc']:.4f}\n"
            final_report += f"- OOT KS: {oot_eval['ks']:.4f}\n"
            final_report += f"- 入模特征数: {len(best_feats)}\n"

            output_name = os.path.splitext(os.path.basename(parsed.report_output))[0]
            report_path = output_dir / f"{output_name}.md"
            report_path.write_text(final_report, encoding='utf-8')
            logger.info(f"报告已保存至: {report_path}")
            emitter.add_report(str(report_path))

        emitter.update_state({
            "lr_tuning": {
                "best_params": best_params,
                "best_metrics": {"oot": {"auc": round(oot_eval['auc'], 4), "ks": round(oot_eval['ks'], 4)}},
                "rounds": len(tuning_history),
                "n_features": len(best_feats),
            },
        })
        emitter.update_state({
            "model_result": {
                "algorithm": "lr_scorecard",
                "scheme_name": "LR调优最优",
                "n_features": len(best_feats),
                "metrics": {"oot": {"auc": round(oot_eval['auc'], 4), "ks": round(oot_eval['ks'], 4)}},
                "source": "tuning",
            },
        })
        emitter.update_metrics({"oot_auc": round(oot_eval['auc'], 4), "oot_ks": round(oot_eval['ks'], 4), "rounds": len(tuning_history)})
        emitter.set_summary(f"lr-tuning auto 完成：OOT AUC={oot_eval['auc']:.4f}，KS={oot_eval['ks']:.4f}")
        result_path = emitter.write(output_dir / "result.json")
        print(f"result: {result_path}")
        return 0

    # ─── 交互模式（单轮） ───────────────────────────────────────────────
    try:
        train_metrics, val_metrics, model, woe_enc, sel_feats = train_lr_model(
            X_train, y_train, X_val, y_val, features, current_params
        )
    except ValueError as e:
        logger.error(f"训练失败: {e}")
        return 1

    # OOT 评估
    X_oot_woe = woe_enc.transform(X_oot)
    prob_oot = model.predict_proba(X_oot_woe)[:, 1]
    oot_metrics = ModelEvaluator.evaluate(y_oot, prob_oot)

    diagnosis = diagnose_model(train_metrics, val_metrics, parsed.prev_val_metric, metric=parsed.metric)

    report = generate_lr_tuning_report(
        params=current_params,
        train_metrics=train_metrics,
        eval_metrics=val_metrics,
        diagnosis=diagnosis,
        n_features_selected=len(sel_feats),
        round_num=parsed.round,
    )
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60 + "\n")

    # 保存模型
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _model_name = parsed.model_name or f"lr_tuning_r{parsed.round}_{_ts}"

    import joblib
    _model_path = str(models_dir / f"{_model_name}.pkl")
    joblib.dump({"model": model, "woe_encoder": woe_enc, "features": sel_feats, "params": current_params}, _model_path)
    logger.info(f"✅ LR 模型已保存至: {_model_path}")
    emitter.add_model(_model_path, meta_path=None, feature_count=len(sel_feats), model_type="lr_scorecard")

    emitter.update_state({
        "lr_tuning": {
            "params": current_params,
            "metrics": {"val": val_metrics, "oot": oot_metrics},
            "diagnosis": diagnosis.status,
            "n_features": len(sel_feats),
        },
    })
    emitter.update_state({
        "model_result": {
            "algorithm": "lr_scorecard",
            "scheme_name": f"LR调优 R{parsed.round}",
            "n_features": len(sel_feats),
            "metrics": {"val": val_metrics, "oot": oot_metrics},
            "source": "tuning",
        },
    })
    emitter.update_metrics({"oot_auc": round(oot_metrics['auc'], 4), "oot_ks": round(oot_metrics['ks'], 4)})
    emitter.set_summary(f"lr-tuning r{parsed.round} 完成：OOT AUC={oot_metrics['auc']:.4f}，KS={oot_metrics['ks']:.4f}")
    result_path = emitter.write(output_dir / "result.json")
    print(f"result: {result_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
