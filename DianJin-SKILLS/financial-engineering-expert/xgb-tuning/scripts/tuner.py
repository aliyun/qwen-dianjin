# -*- coding: utf-8 -*-
"""
XGBoost 超参调优脚本

支持 LLM 诊断驱动的调优，提供：
1. 模型训练与评估
2. 诊断信息输出（供 LLM 分析）
3. 效果对比反馈

Usage:
    # 基线训练
    python tuner.py --data_path ./data.parquet --target y_label --features "f1,f2"
    
    # 指定参数训练
    python tuner.py --data_path ./data.parquet --target y_label --features "f1,f2" \
        --params '{"max_depth": 4, "reg_alpha": 0.5}'
    
    # 与基线对比
    python tuner.py --data_path ./data.parquet --target y_label --features "f1,f2" \
        --params '{"max_depth": 4}' --baseline '{"max_depth": 5}'
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from xgb_core import XGBTrainer, ModelEvaluator  # noqa: E402
from tuning_engine import TuningEngine, DiagnosisResult, diagnose_model, build_params_delta  # noqa: E402
from report_artifact import build_full_eval_section  # noqa: E402
from xgb_cli import build_parser, parse_args  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

from safety_gate import SafetyGate  # noqa: E402
from diagnostic.report import run_diagnostics  # noqa: E402
from model_card import ModelCard, DataFingerprint, FeatureSpec  # noqa: E402

from data_utils import load_and_split_data, build_xgb_eval_set, render_split_section  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 1500,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "verbosity": 0,
    "random_state": 42,
    "early_stopping_rounds": 50,
}

# 注：DiagnosisResult / diagnose_model 已统一移至 shared/tuning_engine.py，通过 import 引入，不在本文件重复定义。

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_oot: pd.DataFrame,
    y_oot: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], XGBTrainer]:
    """训练模型并评估（train/val/oot 三段式，val 用于 early stopping）

    Returns:
        (train_metrics, val_metrics, oot_metrics, trainer)
    """
    trainer = XGBTrainer(params)
    # early stopping 必须用 val，oot 仅用于最终汇报
    trainer.train(X_train, y_train, X_val, y_val)

    p_train = trainer.predict_proba(X_train)
    p_val = trainer.predict_proba(X_val)
    p_oot = trainer.predict_proba(X_oot)

    train_metrics = ModelEvaluator.evaluate(y_train, p_train)
    val_metrics = ModelEvaluator.evaluate(y_val, p_val)
    oot_metrics = ModelEvaluator.evaluate(y_oot, p_oot)

    return train_metrics, val_metrics, oot_metrics, trainer

def format_params_diff(
    old_params: Dict[str, Any],
    new_params: Dict[str, Any],
    tunable_keys: Optional[List[str]] = None,
) -> str:
    """格式化参数变化"""
    if tunable_keys is None:
        tunable_keys = ["max_depth", "learning_rate", "n_estimators", "subsample",
                        "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]
    
    lines = ["| 参数 | 调整前 | 调整后 | 变化 |", "|------|--------|--------|------|"]
    
    for k in tunable_keys:
        old_v = old_params.get(k)
        new_v = new_params.get(k)
        if old_v != new_v:
            if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
                if new_v > old_v:
                    change = f"↑ {new_v/old_v:.1f}x" if old_v != 0 else "↑"
                else:
                    change = f"↓ {old_v/new_v:.1f}x" if new_v != 0 else "↓"
            else:
                change = "变更"
            lines.append(f"| {k} | {old_v} | {new_v} | {change} |")
    
    if len(lines) == 2:
        return "无参数变化"
    
    return "\n".join(lines)

def format_metrics_diff(
    old_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    old_gap: float,
    new_gap: float,
) -> str:
    """格式化效果对比"""
    lines = ["| 指标 | 调整前 | 调整后 | 变化 |", "|------|--------|--------|------|"]
    
    auc_diff = new_metrics["auc"] - old_metrics["auc"]
    ks_diff = new_metrics["ks"] - old_metrics["ks"]
    gap_diff = new_gap - old_gap
    
    auc_mark = "✓" if auc_diff > 0 else ("✗" if auc_diff < -0.005 else "")
    ks_mark = "✓" if ks_diff > 0 else ("✗" if ks_diff < -0.005 else "")
    gap_mark = "✓" if gap_diff < 0 else ("✗" if gap_diff > 0.01 else "")
    
    lines.append(f"| OOT AUC | {old_metrics['auc']:.4f} | {new_metrics['auc']:.4f} | {auc_diff:+.4f} {auc_mark} |")
    lines.append(f"| OOT KS | {old_metrics['ks']:.4f} | {new_metrics['ks']:.4f} | {ks_diff:+.4f} {ks_mark} |")
    lines.append(f"| Gap | {old_gap:.4f} | {new_gap:.4f} | {gap_diff:+.4f} {gap_mark} |")
    
    return "\n".join(lines)

def generate_tuning_report(
    params: Dict[str, Any],
    train_metrics: Dict[str, float],
    oot_metrics: Dict[str, float],
    diagnosis: DiagnosisResult,
    baseline_params: Optional[Dict[str, Any]] = None,
    baseline_oot_metrics: Optional[Dict[str, float]] = None,
    baseline_gap: Optional[float] = None,
    round_num: int = 0,
) -> str:
    """生成调优报告"""
    lines = []
    
    # 标题
    if round_num == 0:
        lines.append("## 基线模型评估")
    else:
        lines.append(f"## Round {round_num} 调优结果")
    
    lines.append("")
    
    # 诊断结果
    if round_num == 0:
        lines.append(f"**诊断**: {diagnosis.status}")
    else:
        lines.append(f"**诊断**: {diagnosis.status}（第 {round_num} 轮）")
    lines.append(f"**建议**: {diagnosis.suggestion}")
    lines.append("")
    
    # 参数变化（如果有基线对比）
    if baseline_params is not None and round_num > 0:
        lines.append("### 参数变化")
        lines.append("")
        lines.append(format_params_diff(baseline_params, params))
        lines.append("")
    
    # 当前效果
    lines.append("### 当前效果")
    lines.append("")
    primary_mark = f"主指标：{diagnosis.primary_metric.upper()}"
    lines.append(f"> 诊断依据 `{primary_mark}`。Train/OOT 均用于汇报，诊断实际依据 train vs val。")
    lines.append("")
    lines.append("| 指标 | Train | OOT | Gap (train-val) |")
    lines.append("|------|-------|-----|-----------------|")
    auc_mark = " ★" if diagnosis.primary_metric == "auc" else ""
    ks_mark = " ★" if diagnosis.primary_metric == "ks" else ""
    lines.append(f"| AUC{auc_mark} | {train_metrics['auc']:.4f} | {oot_metrics['auc']:.4f} | {diagnosis.gap_auc:.4f} |")
    lines.append(f"| KS{ks_mark} | {train_metrics['ks']:.4f} | {oot_metrics['ks']:.4f} | {diagnosis.gap_ks:.4f} |")
    lines.append("")
    
    # 效果对比（如果有基线）
    if baseline_oot_metrics is not None and baseline_gap is not None and round_num > 0:
        lines.append("### 效果对比")
        lines.append("")
        lines.append(format_metrics_diff(baseline_oot_metrics, oot_metrics, baseline_gap, diagnosis.gap))
        lines.append("")
    
    # 当前参数
    lines.append("### 当前参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    for k in ["max_depth", "learning_rate", "n_estimators", "subsample", 
              "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]:
        if k in params:
            lines.append(f"| {k} | {params[k]} |")
    lines.append("")
    
    return "\n".join(lines)

def load_data(
    data_path: str,
    target: str,
    features: List[str],
    time_col: str,
    train_filter: Optional[str],
    val_filter: Optional[str],
    oot_filter: Optional[str],
    val_ratio: float = 0.25,
    oot_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """加载并切分数据为 train/val/oot 三段（统一走 shared/data_utils.load_and_split_data）。

    Returns:
        (X_train, X_val, X_oot, y_train, y_val, y_oot, data_summary, split_meta)
    """
    logger.info("Loading data: {}".format(data_path))

    split_result = load_and_split_data(
        data_path=data_path,
        target=target,
        time_col=time_col,
        train_filter=train_filter,
        test_filter=val_filter,  # 等价 val_filter
        oot_filter=oot_filter,
        val_ratio=val_ratio,
        oot_ratio=oot_ratio,
        random_seed=random_seed,
    )
    train_data = split_result.train
    val_data = split_result.val
    oot_data = split_result.oot
    split_meta = split_result.meta or {}

    # 提取特征和标签
    X_train = train_data[features]
    y_train = train_data[target].values
    X_val = val_data[features]
    y_val = val_data[target].values
    X_oot = oot_data[features]
    y_oot = oot_data[target].values

    # 生成数据摘要（供 LLM 推断基线参数）
    train_pos_rate = y_train.mean() * 100
    val_pos_rate = y_val.mean() * 100 if len(y_val) > 0 else 0.0
    oot_pos_rate = y_oot.mean() * 100

    data_summary = {
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "oot_samples": len(oot_data),
        "n_features": len(features),
        "train_pos_rate": round(train_pos_rate, 2),
        "val_pos_rate": round(val_pos_rate, 2),
        "oot_pos_rate": round(oot_pos_rate, 2),
    }

    logger.info("Train: {}, Val: {}, OOT: {}, Features: {}".format(
        len(train_data), len(val_data), len(oot_data), len(features)))
    logger.info("Train pos_rate: {:.2f}%, Val pos_rate: {:.2f}%, OOT pos_rate: {:.2f}%".format(
        train_pos_rate, val_pos_rate, oot_pos_rate))
    if split_meta:
        logger.info(f"[split-meta] {split_meta}")

    return X_train, X_val, X_oot, y_train, y_val, y_oot, data_summary, split_meta

# 注：_build_eval_charts（KS/ROC/Lift/FI/BCR/校准 6 件套）已下沉至
# shared/report_artifact.build_full_eval_section，本模块直接调用。

# 注：旧的 generate_tuning_strategy 启发式（×2 / ×1.5 / ÷0.5 倍数规则）已统一废弃。
# 下一轮参数由 TuningEngine.run_round() 在 Optuna 约束搜索空间内产出，见 AUTO 循环。

def main(args: Optional[List[str]] = None) -> int:
    """主入口"""
    parser = build_parser("tuning", description="XGBoost 超参调优工具 (portable)")
    parser.add_argument('--warm_start', type=str, default=None,
                        help='WarmStartBundle JSON 字符串或文件路径')
    parser.add_argument('--output_dir', default=None, help='产物输出目录；默认 ./outputs/<ts>')
    parsed = parse_args(parser, args)

    # output_dir
    if getattr(parsed, 'output_dir', None):
        output_dir = Path(parsed.output_dir).resolve()
    else:
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / "outputs" / _ts).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="xgb-tuning", output_dir=output_dir)

    # 解析特征列表
    features = [f.strip() for f in parsed.features.split(",")]

    # 解析参数
    current_params = DEFAULT_PARAMS.copy()
    if parsed.params:
        try:
            custom_params = json.loads(parsed.params)
            current_params.update(custom_params)
        except json.JSONDecodeError:
            logger.error(f"无法解析参数 JSON: {parsed.params}")
            return 1

    baseline_params = None
    if parsed.baseline:
        try:
            baseline_params = DEFAULT_PARAMS.copy()
            baseline_params.update(json.loads(parsed.baseline))
        except json.JSONDecodeError:
            logger.error(f"无法解析基线参数 JSON: {parsed.baseline}")
            return 1

    val_filter_arg = parsed.val_filter

    # 加载数据
    try:
        X_train, X_val, X_oot, y_train, y_val, y_oot, data_summary, split_meta = load_data(
            parsed.data_path,
            parsed.target,
            features,
            parsed.time_col,
            parsed.train_filter,
            val_filter_arg,
            parsed.oot_filter,
            val_ratio=parsed.val_ratio,
            oot_ratio=parsed.oot_ratio,
            random_seed=parsed.random_seed,
        )
    except Exception as e:
        logger.error("Load data failed: {}".format(e))
        return 1

    # 输出数据摘要（供 LLM 推断基线参数）
    print("\n" + "=" * 60)
    print("📊 数据摘要 (Data Summary)")
    print("=" * 60)
    print("训练集样本量: {:,}".format(data_summary['train_samples']))
    print("验证集 val 样本量: {:,}".format(data_summary['val_samples']))
    print("OOT 样本量: {:,}".format(data_summary['oot_samples']))
    print("特征数量: {}".format(data_summary['n_features']))
    print("训练集正样本率: {:.2f}%".format(data_summary['train_pos_rate']))
    print("Val 正样本率: {:.2f}%".format(data_summary['val_pos_rate']))
    print("OOT 正样本率: {:.2f}%".format(data_summary['oot_pos_rate']))
    print("=" * 60 + "\n")

    # SafetyGate 建模前体检
    _train_df = pd.concat([X_train, pd.Series(y_train, index=X_train.index, name=parsed.target)], axis=1)
    _val_df = pd.concat([X_val, pd.Series(y_val, index=X_val.index, name=parsed.target)], axis=1)
    _oot_df = pd.concat([X_oot, pd.Series(y_oot, index=X_oot.index, name=parsed.target)], axis=1)
    gate_report = SafetyGate.check(
        _train_df, _val_df, _oot_df,
        target_col=parsed.target,
        time_col=parsed.time_col,
        features=features,
    )
    if not gate_report.passed:
        for reason in gate_report.blocking_reasons:
            logger.error(f"SafetyGate 阻断: {reason}")
        raise RuntimeError(f"SafetyGate 未通过: {gate_report.blocking_reasons}")
    for w in gate_report.warnings:
        logger.warning(f"SafetyGate 告警: {w}")

    # Task 7-5: 强不均衡场景自动注入 scale_pos_weight
    if data_summary['train_pos_rate'] < 5.0 and 'scale_pos_weight' not in current_params:
        spw = round((100.0 - data_summary['train_pos_rate']) / max(data_summary['train_pos_rate'], 0.01), 1)
        current_params['scale_pos_weight'] = min(spw, 20.0)
        logger.info(f"正样本率 {data_summary['train_pos_rate']:.2f}% < 5%，自动注入 scale_pos_weight={current_params['scale_pos_weight']}")
    
    # 自动调优模式
    if parsed.auto:
        logger.info(f"启用自动调优模式，最大轮数: {parsed.max_rounds}")
        logger.info(f"优化指标: {parsed.metric}")

        # WarmStartBundle 注入（历史先验减少冷启动浪费）
        warm_bundle = None
        if getattr(parsed, 'warm_start', None):
            try:
                ws_raw = parsed.warm_start
                if os.path.isfile(ws_raw):
                    with open(ws_raw, 'r', encoding='utf-8') as f:
                        warm_bundle = json.load(f)
                else:
                    warm_bundle = json.loads(ws_raw)
                logger.info(f"WarmStartBundle 已加载: {len(warm_bundle.get('enqueue_trials', []))} 个先验 trial")
                # 用 baseline_params 覆写默认参数（如果置信度 >= 0.3）
                if warm_bundle.get('confidence', 0) >= 0.3 and warm_bundle.get('baseline_params'):
                    for k, v in warm_bundle['baseline_params'].items():
                        if k in current_params:
                            current_params[k] = v
                    logger.info(f"已用 WarmStartBundle baseline_params 覆写当前参数")
            except Exception as e:
                logger.warning(f"WarmStartBundle 解析失败，忽略: {e}")
                warm_bundle = None

        best_params = current_params.copy()
        best_val_metric = 0.0
        best_oot_metric = 0.0
        best_val_metrics: Dict[str, float] = {}  # 最佳轮次完整 val 指标（含 auc/ks）
        best_oot_metrics: Dict[str, float] = {}  # 最佳轮次完整 oot 指标（含 auc/ks）
        prev_val_metric = parsed.prev_val_metric
        no_improvement_count = 0
        tuning_history = []
        tried_directions: List[Dict[str, Any]] = []  # 记录每轮参数变化方向和效果

        baseline_oot_metrics = None
        baseline_gap = None
        
        for round_num in range(1, parsed.max_rounds + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"开始第 {round_num} 轮自动调优")
            logger.info(f"{'='*60}")
            
            # 训练当前模型
            train_metrics, val_metrics, oot_metrics, trainer = train_and_evaluate(
                X_train, y_train, X_val, y_val, X_oot, y_oot, current_params
            )
            
            # 获取当前指标：调优决策用 val，汇报用 oot
            current_val_metric = val_metrics[parsed.metric]
            current_oot_metric = oot_metrics[parsed.metric]
            
            # 诊断：train vs val（遵循经典三段式，oot 只用于最终汇报）
            diagnosis = diagnose_model(train_metrics, val_metrics, prev_val_metric, metric=parsed.metric)
            
            # 生成并输出报告
            report = generate_tuning_report(
                params=current_params,
                train_metrics=train_metrics,
                oot_metrics=oot_metrics,
                diagnosis=diagnosis,
                baseline_params=baseline_params if round_num == 1 else None,
                baseline_oot_metrics=baseline_oot_metrics if round_num == 1 else None,
                baseline_gap=baseline_gap if round_num == 1 else None,
                round_num=round_num,
            )
            
            print("\n" + "=" * 60)
            print(report)
            print("=" * 60 + "\n")
            
            # 记录历史（以 val 指标作为改进判断，同时保留 oot 供汇报）
            improvement_val = (current_val_metric - prev_val_metric) if prev_val_metric is not None else None
            if current_val_metric > best_val_metric:
                decision_str = "\u2705 新最优"
            elif current_val_metric >= best_val_metric - 0.001:
                decision_str = "— 持平"
            else:
                decision_str = "\u274c 未改进"
            tuning_history.append({
                'round': round_num,
                'status': diagnosis.status,
                'val_auc': val_metrics['auc'],
                'val_ks': val_metrics['ks'],
                'oot_auc': oot_metrics['auc'],
                'oot_ks': oot_metrics['ks'],
                'gap': diagnosis.gap,
                'improvement': improvement_val,
                'decision': decision_str,
                'params': {k: current_params.get(k) for k in ["max_depth", "learning_rate", "reg_alpha", "reg_lambda", "min_child_weight", "subsample"]},
            })
            
            # 检查是否超过最佳（以 val 指标决定最优）
            # 先保存比较结果，避免下方 tried_directions 的 improved 判断在 best 更新后恒为 False
            is_new_best = current_val_metric > best_val_metric
            if is_new_best:
                best_val_metric = current_val_metric
                best_oot_metric = current_oot_metric
                best_val_metrics = dict(val_metrics)
                best_oot_metrics = dict(oot_metrics)
                best_params = current_params.copy()
                no_improvement_count = 0
                logger.info(f"✓ 发现更优模型，val {parsed.metric}: {current_val_metric:.4f}（OOT {parsed.metric}={current_oot_metric:.4f}）")
            else:
                no_improvement_count += 1
                logger.info(f"✗ 未改进，val {parsed.metric}: {current_val_metric:.4f} (最佳 val: {best_val_metric:.4f})")
            
            # 收敛判断（用 val 指标）
            if prev_val_metric is not None:
                improvement = abs(current_val_metric - prev_val_metric)
                if improvement < 0.001:
                    logger.info(f"收敛：连续改进 < 0.001，停止调优")
                    break
            
            if no_improvement_count >= 2:
                logger.info(f"收敛：连续 2 轮未改进，停止调优")
                break
            
            # 通过统一引擎在诊断约束空间内搜索下一轮最优参数（objective 用 val，防止 oot 泄漏）
            engine = TuningEngine(
                X_train, y_train, X_val, y_val,
                base_params=current_params,
                metric=parsed.metric,
            )
            # WarmStartBundle: 第一轮注入历史先验 trial
            n_extra = 0
            if round_num == 1 and warm_bundle and warm_bundle.get('enqueue_trials'):
                try:
                    import optuna
                except ImportError:
                    optuna = None
                if optuna is not None:
                    for et in warm_bundle['enqueue_trials']:
                        n_extra += 1
                    logger.info(f"WarmStartBundle: {n_extra} 个先验 trial 将纳入本轮搜索")
            next_params, round_best_score, _round_trials, round_suggestion = engine.run_round(
                diagnosis,
                tried_directions=tried_directions,
                n_trials=5 + n_extra,
            )
            logger.info(round_suggestion)
            params_delta = build_params_delta(current_params, next_params)
            tried_directions.append({
                "status": diagnosis.status,
                "params_delta": params_delta,
                "improved": is_new_best,
            })

            prev_val_metric = current_val_metric
            current_params = next_params
            
            # 如果是第一轮，保存基线用于对比
            if round_num == 1:
                baseline_params = best_params.copy()
                baseline_oot_metrics = oot_metrics.copy()
                baseline_gap = diagnosis.gap
        
        # 输出最终结果
        logger.info(f"\n{'='*60}")
        logger.info(f"自动调优完成")
        logger.info(f"最佳 val {parsed.metric}: {best_val_metric:.4f}（对应 OOT {parsed.metric}={best_oot_metric:.4f}）")
        logger.info(f"最佳参数: {json.dumps(best_params, indent=2)}")
        logger.info(f"{'='*60}\n")

        # 用最优参数重新训练（仍用 val 做 early stopping），生成完整评估图表
        final_train_m, final_val_m, _, best_trainer = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_oot, y_oot, best_params
        )
        best_prob_train = best_trainer.predict_proba(X_train)
        best_prob_val = best_trainer.predict_proba(X_val)
        best_prob_oot = best_trainer.predict_proba(X_oot)

        # DiagnosticReport（完整维度诊断）
        diag_report = run_diagnostics(
            train_metrics=final_train_m,
            eval_metrics=final_val_m,
            metric=parsed.metric,
            y_true_train=y_train,
            y_prob_train=best_prob_train,
            y_true_val=y_val,
            y_prob_val=best_prob_val,
            y_true_oot=y_oot,
            y_prob_oot=best_prob_oot,
        )
        if diag_report.is_overfit:
            logger.warning(f"DiagnosticReport: 过拟合 severity={diag_report.severity}")
        for w in diag_report.all_warnings:
            logger.warning(f"DiagnosticReport: {w}")

        best_importance = best_trainer.get_feature_importance(sort=True)
        curve_section = "\n## 最优模型评估可视化\n\n" + build_full_eval_section(
            y_oot, best_prob_oot, best_importance, prefix="最优模型 "
        )

        print(curve_section)

        # 保存最优模型
        models_dir = output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _model_name = parsed.model_name or f"xgb_tuning_best_{_ts}"
        _model_path = str(models_dir / f"{_model_name}.json")
        _model_meta_path = str(models_dir / f"{_model_name}_meta.json")
        best_trainer.save_model(_model_path)
        logger.info(f"✅ 最优模型已保存至: {_model_path}")
        emitter.add_model(_model_path, meta_path=_model_meta_path, feature_count=len(features), model_type="xgboost")

        # 生成 ModelCard
        data_fp = DataFingerprint(
            n_samples=data_summary['train_samples'] + data_summary['val_samples'] + data_summary['oot_samples'],
            n_features=data_summary['n_features'],
            pos_rate=data_summary['train_pos_rate'] / 100.0,
        )
        model_card = ModelCard.create(
            model_id=_model_name,
            params=best_params,
            eval_results={
                "val": dict(best_val_metrics) if best_val_metrics else {},
                "oot": dict(best_oot_metrics) if best_oot_metrics else {},
            },
            data_fingerprint=data_fp,
            diagnostic=diag_report.to_dict(),
        )
        _card_path = str(models_dir / f"{_model_name}_card.json")
        with open(_card_path, 'w', encoding='utf-8') as f:
            import json as _json
            f.write(model_card.to_json())
        logger.info(f"✅ ModelCard 已保存至: {_card_path}")

        # 保存最终报告
        if parsed.report_output:
            final_report = f"# XGBoost 自动调优报告\n\n"
            final_report += f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            final_report += "---\n\n"

            # 数据切分概览（统一 render_split_section）
            if split_meta:
                final_report += render_split_section(split_meta) + "\n"

            # 调优配置摘要
            final_report += "## 1. 调优配置\n\n"
            final_report += "| 配置项 | 值 |\n"
            final_report += "|--------|------|\n"
            final_report += f"| 调优模式 | 自动调优 |\n"
            final_report += f"| 优化指标 | {parsed.metric.upper()} |\n"
            final_report += f"| 最大轮数 | {parsed.max_rounds} |\n"
            final_report += f"| 实际轮数 | {round_num} |\n"
            final_report += f"| 特征数 | {len(features)} |\n"
            final_report += f"| 训练集样本 | {data_summary['train_samples']:,} |\n"
            final_report += f"| OOT 样本 | {data_summary['oot_samples']:,} |\n\n"

            # 调优推演记录表
            final_report += "## 2. 调优推演记录\n\n"
            final_report += f"| 轮次 | 诊断状态 | OOT AUC | OOT KS | Gap | 改进 | 决策 |\n"
            final_report += f"|------|----------|---------|--------|-----|------|------|\n"
            # 重新训练每轮以获取历史记录 - 用已保存的变量
            # 注: 此处我们在循环中收集历史记录
            for hist in tuning_history:
                imp_str = f"{hist['improvement']:+.4f}" if hist.get('improvement') is not None else "-"
                decision = hist.get('decision', '')
                final_report += f"| {hist['round']} | {hist['status']} | {hist['oot_auc']:.4f} | {hist['oot_ks']:.4f} | {hist['gap']:.4f} | {imp_str} | {decision} |\n"
            final_report += "\n"

            # 参数变化轨迹
            final_report += "## 3. 参数变化轨迹\n\n"
            track_keys = ["max_depth", "learning_rate", "reg_alpha", "reg_lambda", "min_child_weight", "subsample"]
            header = "| 轮次 | " + " | ".join(track_keys) + " |\n"
            sep = "|------| " + " | ".join(["---"] * len(track_keys)) + " |\n"
            final_report += header + sep
            for hist in tuning_history:
                params_row = hist.get('params', {})
                vals = [str(params_row.get(k, '-')) for k in track_keys]
                final_report += f"| {hist['round']} | " + " | ".join(vals) + " |\n"
            final_report += "\n"

            # 最终效果对比
            final_report += "## 4. 最终效果对比\n\n"
            if tuning_history:
                first = tuning_history[0]
                last = tuning_history[-1]
                final_report += "| 指标 | 基线（Round 1） | 最终 | 变化 |\n"
                final_report += "|------|-----------|------|------|\n"
                auc_diff = last['oot_auc'] - first['oot_auc']
                ks_diff = last['oot_ks'] - first['oot_ks']
                gap_diff = last['gap'] - first['gap']
                final_report += f"| OOT AUC | {first['oot_auc']:.4f} | {last['oot_auc']:.4f} | {auc_diff:+.4f} |\n"
                final_report += f"| OOT KS | {first['oot_ks']:.4f} | {last['oot_ks']:.4f} | {ks_diff:+.4f} |\n"
                final_report += f"| Gap | {first['gap']:.4f} | {last['gap']:.4f} | {gap_diff:+.4f} |\n"

                # 调优前后对比柱状图
                comparison_data = json.dumps({
                    "labels": ["OOT AUC", "OOT KS", "Gap"],
                    "series": [
                        {"name": "基线", "type": "bar", "data": [round(first['oot_auc'], 4), round(first['oot_ks'], 4), round(first['gap'], 4)]},
                        {"name": "调优后", "type": "bar", "data": [round(last['oot_auc'], 4), round(last['oot_ks'], 4), round(last['gap'], 4)]}
                    ]
                }, ensure_ascii=False)
                final_report += '\n<!--ARTIFACT_START--><artifact type="echarts" title="调优前后指标对比">\n'
                final_report += comparison_data
                final_report += '\n</artifact><!--ARTIFACT_END-->\n\n'
            final_report += "\n"

            # 最佳参数
            final_report += "## 5. 最佳参数\n\n"
            final_report += f"- **{parsed.metric.upper()}**: {best_oot_metric:.4f}\n\n"
            final_report += "| 参数 | 值 |\n"
            final_report += "|------|------|\n"
            for k in ["max_depth", "learning_rate", "n_estimators", "subsample",
                      "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]:
                if k in best_params:
                    final_report += f"| {k} | {best_params[k]} |\n"
            final_report += "\n"

            # 最优模型评估可视化
            final_report += "## 6. 最优模型评估可视化\n\n"
            final_report += build_full_eval_section(y_oot, best_prob_oot, best_importance, prefix="最优模型 ")
            final_report += "\n"

            # 调优结论
            final_report += "## 7. 调优结论\n\n"
            if tuning_history:
                total_improvement = best_oot_metric - tuning_history[0][f'oot_{parsed.metric}']
                if total_improvement > 0.005:
                    final_report += f"> 经过 {round_num} 轮调优，{parsed.metric.upper()} 提升 {total_improvement:+.4f}，调优效果显著。"
                elif total_improvement > 0:
                    final_report += f"> 经过 {round_num} 轮调优，{parsed.metric.upper()} 小幅提升 {total_improvement:+.4f}，模型已接近最优。"
                else:
                    final_report += f"> 经过 {round_num} 轮调优，未获得明显改善，建议从特征工程角度优化。"
            final_report += "\n"
            
            output_name = os.path.splitext(os.path.basename(parsed.report_output))[0]
            report_path = output_dir / f"{output_name}.md"
            report_path.write_text(final_report, encoding='utf-8')
            logger.info(f"最终报告已保存至: {report_path}")
            emitter.add_report(str(report_path))

        # 登记 project_state 更新（auto 模式）
        baseline_first = tuning_history[0] if tuning_history else {}
        best_last = tuning_history[-1] if tuning_history else {}
        # best_val_metrics / best_oot_metrics 是最佳轮次的完整指标；若全程未产生更优，则回退到末轮 val/oot
        final_val = best_val_metrics or (val_metrics if 'val_metrics' in dir() else {})
        final_oot = best_oot_metrics or (oot_metrics if 'oot_metrics' in dir() else {})
        baseline_oot_auc = baseline_first.get('oot_auc', final_oot.get('auc', 0))
        baseline_oot_ks = baseline_first.get('oot_ks', final_oot.get('ks', 0))
        emitter.update_state({
            "tuning": {
                "best_params": best_params,
                "best_metrics": {
                    "val": {
                        "auc": round(float(final_val.get('auc', 0)), 4),
                        "ks": round(float(final_val.get('ks', 0)), 4),
                    },
                    "oot": {
                        "auc": round(float(final_oot.get('auc', 0)), 4),
                        "ks": round(float(final_oot.get('ks', 0)), 4),
                    },
                },
                "baseline_metrics": {
                    "oot": {"auc": round(baseline_oot_auc, 4), "ks": round(baseline_oot_ks, 4)},
                } if baseline_first else None,
                "improvement": {
                    "auc_delta": round(float(final_oot.get('auc', 0)) - baseline_oot_auc, 4),
                    "ks_delta": round(float(final_oot.get('ks', 0)) - baseline_oot_ks, 4),
                } if baseline_first else None,
                "rounds": round_num,
            },
        })

        emitter.update_state({
            "model_result": {
                "algorithm": "xgboost",
                "scheme_name": "调优最优",
                "n_features": len(features),
                "metrics": {
                    "val": {
                        "auc": round(float(final_val.get('auc', 0)), 4),
                        "ks": round(float(final_val.get('ks', 0)), 4),
                    },
                    "oot": {
                        "auc": round(float(final_oot.get('auc', 0)), 4),
                        "ks": round(float(final_oot.get('ks', 0)), 4),
                    },
                },
                "source": "tuning",
            },
        })

        emitter.update_metrics({
            "oot_auc": round(float(final_oot.get('auc', 0)), 4),
            "oot_ks": round(float(final_oot.get('ks', 0)), 4),
            "rounds": round_num,
        })
        emitter.set_summary(f"xgb-tuning auto 完成：{round_num} 轮，OOT AUC={final_oot.get('auc',0):.4f}，OOT KS={final_oot.get('ks',0):.4f}")
        result_path = emitter.write(output_dir / "result.json")
        print(f"result: {result_path}")
        return 0
    
    # 交互式模式（原有逻辑，使用经典三段式：诊断/收敛用 val，最终汇报用 oot）
    # 训练基线模型（如果需要对比）
    baseline_oot_metrics = None
    baseline_val_metrics = None
    baseline_gap = None
    if baseline_params is not None:
        logger.info("训练基线模型...")
        baseline_train_metrics, baseline_val_metrics, baseline_oot_metrics, _ = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_oot, y_oot, baseline_params
        )
        # baseline_gap 遵循主指标（与诊断阈值体系对齐）：KS 模式下用 KS Gap，AUC 模式下用 AUC Gap
        baseline_gap = baseline_train_metrics[parsed.metric] - baseline_val_metrics[parsed.metric]
    
    # 训练当前模型
    logger.info("训练当前模型...")
    train_metrics, val_metrics, oot_metrics, trainer = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_oot, y_oot, current_params
    )
    
    # 诊断：train vs val（仅使用 --prev_val_metric）
    prev_metric_for_diag = parsed.prev_val_metric
    diagnosis = diagnose_model(train_metrics, val_metrics, prev_metric_for_diag, metric=parsed.metric)
    
    # 生成报告（调优报告中的 'OOT' 实际展示 oot，val 仅用于诊断/收敛，保持业务习惯的报告语义）
    report = generate_tuning_report(
        params=current_params,
        train_metrics=train_metrics,
        oot_metrics=oot_metrics,
        diagnosis=diagnosis,
        baseline_params=baseline_params,
        baseline_oot_metrics=baseline_oot_metrics,
        baseline_gap=baseline_gap,
        round_num=parsed.round,
    )
    
    # 输出报告
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60 + "\n")
    
    # 生成完整评估图表可视化
    prob_oot = trainer.predict_proba(X_oot)
    round_prefix = f"Round {parsed.round} " if parsed.round > 0 else ""
    curve_section = f"## 模型评估可视化\n\n" + build_full_eval_section(
        y_oot, prob_oot, trainer.get_feature_importance(sort=True), prefix=round_prefix
    )
    print(curve_section)

    # 保存模型
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _round_str = f"r{parsed.round}_" if parsed.round > 0 else "baseline_"
    _model_name = parsed.model_name or f"xgb_tuning_{_round_str}{_ts}"
    _model_path = str(models_dir / f"{_model_name}.json")
    _model_meta_path = str(models_dir / f"{_model_name}_meta.json")
    trainer.save_model(_model_path)
    logger.info(f"✅ 模型已保存至: {_model_path}")
    emitter.add_model(_model_path, meta_path=_model_meta_path, feature_count=len(features), model_type="xgboost")

    # 输出结构化数据（供 Agent 解析）
    result = {
        "round": parsed.round,
        "data_summary": data_summary,  # 数据摘要供 LLM 推断基线参数
        "diagnosis": diagnosis.to_dict(),
        "params": {k: current_params[k] for k in 
                   ["max_depth", "learning_rate", "n_estimators", "subsample",
                    "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]},
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "oot": oot_metrics,
        },
    }
    
    logger.info("诊断结果: {}".format(diagnosis.status))
    logger.info("Val AUC: {:.4f}, KS: {:.4f} | OOT AUC: {:.4f}, KS: {:.4f}, Gap: {:.4f}".format(
        val_metrics['auc'], val_metrics['ks'],
        oot_metrics['auc'], oot_metrics['ks'], diagnosis.gap))
    
    # 保存报告
    if parsed.report_output:
        full_report = f"# XGBoost 调优报告\n\n"
        full_report += f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        if split_meta:
            full_report += render_split_section(split_meta) + "\n"
        full_report += report
        full_report += "\n---\n\n"
        full_report += curve_section
        
        output_name = os.path.splitext(os.path.basename(parsed.report_output))[0]
        report_path = output_dir / f"{output_name}.md"
        report_path.write_text(full_report, encoding='utf-8')
        logger.info(f"报告已保存至: {report_path}")
        emitter.add_report(str(report_path))

    # 登记 project_state 更新（交互模式）
    baseline_auc = baseline_oot_metrics["auc"] if baseline_oot_metrics else oot_metrics["auc"]
    baseline_ks = baseline_oot_metrics["ks"] if baseline_oot_metrics else oot_metrics["ks"]
    val_auc = val_metrics["auc"]
    val_ks = val_metrics["ks"]
    emitter.update_state({
        "tuning": {
            "best_params": {k: current_params[k] for k in
                           ["max_depth", "learning_rate", "n_estimators", "subsample",
                            "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]},
            "best_metrics": {
                "val": {"auc": round(val_auc, 4), "ks": round(val_ks, 4)},
                "oot": {"auc": round(oot_metrics["auc"], 4), "ks": round(oot_metrics["ks"], 4)},
            },
            "baseline_metrics": {
                "oot": {"auc": round(baseline_auc, 4), "ks": round(baseline_ks, 4)},
            },
            "improvement": {
                "auc_delta": round(oot_metrics["auc"] - baseline_auc, 4),
                "ks_delta": round(oot_metrics["ks"] - baseline_ks, 4),
            },
            "rounds": parsed.round,
        },
    })

    emitter.update_state({
        "model_result": {
            "algorithm": "xgboost",
            "scheme_name": f"调优 R{parsed.round}",
            "n_features": len(features),
            "metrics": {
                "val": {"auc": round(val_auc, 4), "ks": round(val_ks, 4)},
                "oot": {"auc": round(oot_metrics["auc"], 4), "ks": round(oot_metrics["ks"], 4)},
            },
            "source": "tuning",
        },
    })

    emitter.update_metrics({
        "oot_auc": round(oot_metrics["auc"], 4),
        "oot_ks": round(oot_metrics["ks"], 4),
        "round": parsed.round,
    })
    emitter.set_summary(f"xgb-tuning r{parsed.round} 完成：OOT AUC={oot_metrics['auc']:.4f}，KS={oot_metrics['ks']:.4f}")
    result_path = emitter.write(output_dir / "result.json")
    print(f"result: {result_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
