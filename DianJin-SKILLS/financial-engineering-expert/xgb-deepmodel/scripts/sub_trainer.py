# -*- coding: utf-8 -*-
"""
DeepModel — 分群子模型训练

对每个分群独立训练 XGBoost 模型，输出各分群指标和子模型路径。

Usage:
    python sub_trainer.py --data_path ./data.parquet --target y_label \
        --segments '{"高风险": "risk_score > 500", "低风险": "risk_score <= 500"}'
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from xgb_core import XGBTrainer, ModelEvaluator  # noqa: E402
from data_utils import load_and_split_data  # noqa: E402
from tuning_engine import TuningEngine, diagnose_model, build_params_delta  # noqa: E402
from xgb_cli import build_parser, parse_args  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("./outputs/deepmodel")

# 数据加载统一使用 shared/data_utils.load_and_split_data

# ── 分群解析 ──────────────────────────────────────────────────

def resolve_segments(
    data: pd.DataFrame, segments_json: Optional[str], segment_col: Optional[str]
) -> Dict[str, str]:
    if segments_json:
        return json.loads(segments_json)
    if segment_col:
        vals = data[segment_col].dropna().unique()
        return {str(v): f"`{segment_col}` == {repr(v)}" for v in sorted(vals)}
    raise ValueError("必须提供 --segments 或 --segment_col")

# ── 单分群训练 ────────────────────────────────────────────────

def _auto_tune_segment(
    trainer: XGBTrainer,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    sample_weight: Optional[np.ndarray],
) -> Tuple[Any, Dict[str, float]]:
    """分群内自动调参：复用 shared/tuning_engine。

    经典三段式：诊断 / early stopping / 调优均用 val，OOT 仅用于最终汇报（本函数不涉及）。

    策略：基线训练后，若 train↔val Gap > 0.05 则用 TuningEngine.run_round 做最多 3 轮
    约束搜索（每轮 5 trials），每轮根据上轮结果重新诊断。返回 (best_model, best_params)。
    """
    # val 为空/None 时无法诊断过拟合也无法调参，直接回落到普通训练
    if y_val is None or len(y_val) == 0:
        trainer.train(X_train, y_train, None, None, sample_weight=sample_weight)
        return trainer.model, trainer.params.copy()

    trainer.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
    best_model = trainer.model
    best_params = trainer.params.copy()
    train_m = ModelEvaluator.evaluate(y_train, best_model.predict_proba(X_train)[:, 1])
    val_m = ModelEvaluator.evaluate(y_val, best_model.predict_proba(X_val)[:, 1])
    best_gap = train_m["auc"] - val_m["auc"]

    if best_gap <= 0.05:
        return best_model, best_params

    tried_directions: List[Dict[str, Any]] = []
    current_params = best_params
    prev_score = val_m["auc"]

    for _round in range(3):
        # sub_trainer 内部使用 AUC 作为主指标（与下方 TuningEngine metric="auc" 对齐）
        diagnosis = diagnose_model(train_m, val_m, metric="auc")
        if diagnosis.gap <= 0.05:
            break
        engine = TuningEngine(
            X_train, y_train, X_val, y_val,
            base_params=current_params,
            metric="auc",
            sample_weight=sample_weight,
        )
        next_params, _round_score, _trials, _suggestion = engine.run_round(
            diagnosis,
            tried_directions=tried_directions,
            n_trials=5,
        )

        # 用本轮最优参数在 train/val 上重新训练，获取完整指标
        t = XGBTrainer(params=next_params)
        t.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
        tr_m = ModelEvaluator.evaluate(y_train, t.model.predict_proba(X_train)[:, 1])
        va_m = ModelEvaluator.evaluate(y_val, t.model.predict_proba(X_val)[:, 1])
        gap = tr_m["auc"] - va_m["auc"]

        params_delta = build_params_delta(current_params, next_params)
        tried_directions.append({
            "status": diagnosis.status,
            "params_delta": params_delta,
            "improved": gap < best_gap,
        })

        if gap < best_gap:
            best_gap = gap
            best_model = t.model
            best_params = next_params
            if gap < 0.05:
                break

        train_m, val_m = tr_m, va_m
        current_params = next_params
        if abs(va_m["auc"] - prev_score) < 0.001:
            break
        prev_score = va_m["auc"]

    return best_model, best_params

def train_segment(
    name: str,
    train: pd.DataFrame, val: pd.DataFrame, oot: pd.DataFrame,
    target: str,
    features: List[str],
    query: str,
    sample_weight_col: Optional[str],
    pos_weight: Optional[float],
    auto_tune: bool,
    output_dir: Path,
) -> Dict:
    seg_train = train.query(query).reset_index(drop=True) if query else train
    seg_val = val.query(query).reset_index(drop=True) if query else val
    seg_oot = oot.query(query).reset_index(drop=True) if query else oot

    logger.info(f"[{name}] 训练:{len(seg_train)} 验证:{len(seg_val)} OOT:{len(seg_oot)}")

    # 过滤可用特征（跳过全空列）
    avail = [f for f in features if f in seg_train.columns and seg_train[f].notna().any()]

    X_train = seg_train[avail]
    y_train = seg_train[target]
    # val/oot 为空时传 None（trainer.train 会自动关闭 early stopping）
    has_val = len(seg_val) > 0
    has_oot = len(seg_oot) > 0
    X_val = seg_val[avail] if has_val else None
    y_val = seg_val[target] if has_val else None
    X_oot = seg_oot[avail] if has_oot else None
    y_oot = seg_oot[target] if has_oot else None

    sample_weight = None
    if sample_weight_col and sample_weight_col in seg_train.columns:
        sample_weight = seg_train[sample_weight_col].values.astype(float)

    params = {}
    if pos_weight is not None:
        params["scale_pos_weight"] = pos_weight

    trainer = XGBTrainer(params=params if params else None)

    oot_metrics: Dict[str, float] = {}
    if auto_tune:
        model, _ = _auto_tune_segment(trainer, X_train, y_train, X_val, y_val, sample_weight)
    else:
        trainer.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
        model = trainer.model

    train_metrics = ModelEvaluator.evaluate(y_train, model.predict_proba(X_train)[:, 1])
    val_metrics = ModelEvaluator.evaluate(y_val, model.predict_proba(X_val)[:, 1]) if has_val else {}
    oot_metrics = ModelEvaluator.evaluate(y_oot, model.predict_proba(X_oot)[:, 1]) if has_oot else {}

    model_path = output_dir / f"{name}.json"
    model.save_model(str(model_path))

    train_auc = train_metrics.get("auc", 0)
    oot_auc = oot_metrics.get("auc", 0)
    gap = round(train_auc - oot_auc, 6)

    return {
        "name": name,
        "query": query,
        "features": avail,
        "model_path": str(model_path),
        "n_train": len(seg_train),
        "n_val": len(seg_val),
        "n_oot": len(seg_oot),
        "train_auc": train_auc,
        "val_auc": val_metrics.get("auc", 0),
        "oot_auc": oot_auc,
        "train_ks": train_metrics.get("ks", 0),
        "val_ks": val_metrics.get("ks", 0),
        "oot_ks": oot_metrics.get("ks", 0),
        "gap": gap,
        "tuned": auto_tune and gap > 0.05,
    }

# ── 报告生成 ──────────────────────────────────────────────────

def _gap_flag(gap: float) -> str:
    return "✅" if gap < 0.05 else "⚠️"

def generate_report(results: List[Dict]) -> str:
    lines = ["# DeepModel — 分群子模型训练报告", ""]

    # 质量门控汇总
    bad_gap = [r for r in results if r["gap"] >= 0.05]
    small_n = [r for r in results if r["n_train"] < 500]
    lines += ["## 质量门控检查", ""]
    if not bad_gap and not small_n:
        lines.append("✅ 所有分群通过质量门控，可进入 Stacking 阶段。")
    else:
        if bad_gap:
            names = "、".join(r["name"] for r in bad_gap)
            lines.append(f"⚠️ OOT Gap ≥ 0.05 的分群：**{names}**（建议合并或重新设计分群条件）")
        if small_n:
            names = "、".join(r["name"] for r in small_n)
            lines.append(f"⚠️ 训练样本 < 500 的分群：**{names}**（建议合并该分群）")
    lines.append("")

    # 指标对比表
    lines += ["## 各分群指标对比", ""]
    lines.append("| 分群 | 训练量 | OOT量 | Train AUC | Val AUC | OOT AUC | Train KS | OOT KS | Gap | 状态 |")
    lines.append("|------|:------:|:-----:|:---------:|:-------:|:-------:|:--------:|:------:|:---:|:----:|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['n_train']:,} | {r['n_oot']:,} "
            f"| {r['train_auc']:.4f} | {r['val_auc']:.4f} | {r['oot_auc']:.4f} "
            f"| {r['train_ks']:.4f} | {r['oot_ks']:.4f} "
            f"| {r['gap']:.4f} | {_gap_flag(r['gap'])} |"
        )
    lines.append("")

    # 分群条件汇总
    lines += ["## 分群条件汇总", ""]
    for r in results:
        lines.append(f"- **{r['name']}**：`{r['query']}`，特征数 {len(r['features'])}，{'已自动调参' if r['tuned'] else '默认参数'}")
    lines.append("")

    lines += [
        "## 下一步",
        "",
        "请确认以上各分群指标，如无异议回复「继续 Stacking」，我将进入融合阶段。",
        "如需调整分群方案，请告知修改意见。",
    ]

    return "\n".join(lines)

# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = build_parser("deepmodel-sub", description="DeepModel 分群子模型训练")
    args = parse_args(parser)
    emitter = ResultEmitter()

    auto_tune = bool(args.auto)
    exclude_cols = [c.strip() for c in (args.exclude_cols or "").split(",") if c.strip()]

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_result = load_and_split_data(
        data_path=args.data_path,
        target=args.target,
        time_col=args.time_col,
        train_filter=args.train_filter,
        test_filter=args.val_filter,
        oot_filter=args.oot_filter,
        exclude_cols=exclude_cols,
        return_features=True,
    )
    train, val, oot = split_result.train, split_result.val, split_result.oot
    all_features = split_result.features or []
    if split_result.meta:
        logger.info(f"[split-meta] {split_result.meta}")

    segments = resolve_segments(train, args.segments, args.segment_col)

    global_features = [f.strip() for f in args.features.split(",")] if args.features else all_features
    features_per_seg = json.loads(args.features_per_segment) if args.features_per_segment else {}
    pos_weight_map = json.loads(args.pos_weight_per_segment) if args.pos_weight_per_segment else {}

    results = []
    for name, query in segments.items():
        seg_features = features_per_seg.get(name, global_features)
        pos_weight = pos_weight_map.get(name)
        result = train_segment(
            name=name,
            train=train, val=val, oot=oot,
            target=args.target,
            features=seg_features,
            query=query,
            sample_weight_col=args.sample_weight_col,
            pos_weight=pos_weight,
            auto_tune=auto_tune,
            output_dir=output_dir,
        )
        results.append(result)
        print(f"[{name}] OOT AUC={result['oot_auc']:.4f} Gap={result['gap']:.4f}", flush=True)

    # 输出报告
    report = generate_report(results)
    print(report, flush=True)

    # STATE_UPDATE
    submodels = [
        {
            "name": r["name"],
            "query": r["query"],
            "features": r["features"],
            "model_path": r["model_path"],
            "oot_auc": r["oot_auc"],
            "gap": r["gap"],
            "n_train": r["n_train"],
        }
        for r in results
    ]
    segment_metrics = {r["name"]: {"oot_auc": r["oot_auc"], "oot_ks": r["oot_ks"], "gap": r["gap"]} for r in results}

    state = {
        "deepmodel_phase": "submodels_done",
        "submodels": submodels,
        "segment_metrics": segment_metrics,
        "output_dir": str(output_dir),
        "segments_json": args.segments,
        "segment_col": args.segment_col,
        "target": args.target,
        "time_col": args.time_col,
        "train_filter": args.train_filter,
        "val_filter": args.val_filter,
        "oot_filter": args.oot_filter,
        "exclude_cols": args.exclude_cols,
    }
    emitter.update_state(state)

    for r in results:
        emitter.add_model(
            r["model_path"],
            segment=r["name"],
            feature_count=len(r["features"]),
            model_type="xgboost",
        )

    emitter.set_summary(f"分群子模型训练完成: {len(results)} 个分群")
    emitter.write(output_dir / "result.json")
    return 0

if __name__ == "__main__":
    main()
