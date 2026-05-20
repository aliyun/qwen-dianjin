# -*- coding: utf-8 -*-
"""
DeepModel — 集成对比报告

对比三种方案：单模型基线 / 最优子模型 / Stacking 集成，给出最终建议。

Usage:
    python comparator.py --data_path ./data.parquet --target y_label \
        --stack_model_path /path/stack_meta.json \
        --submodel_paths '["/path/高风险.json", "/path/低风险.json"]' \
        --segments '{"高风险": "risk_score > 500", "低风险": "risk_score <= 500"}'
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from xgb_core import XGBTrainer, ModelEvaluator  # noqa: E402
from data_utils import load_and_split_data  # noqa: E402
from xgb_cli import build_parser, parse_args  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

# Task 0-5 基础设施层
from safety_gate import SafetyGate  # noqa: E402

# model-comparison 的 trainers 子包不在默认 PYTHONPATH 中，动态插入以支持多算法基线复用
_MC_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "model-comparison" / "scripts"
)
if _MC_SCRIPTS_DIR.is_dir() and str(_MC_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_MC_SCRIPTS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("./outputs/deepmodel")

# 数据加载统一使用 shared/data_utils.load_and_split_data

# ── 模型加载兼容工具 ──────────────────────────────────────────

def _restore_sklearn_attrs(m: xgb.XGBClassifier, path: str) -> xgb.XGBClassifier:
    """XGBoost 2.x load_model() 不恢复 sklearn 属性，手动补回。"""
    if hasattr(m, "n_classes_") and m.n_classes_ is not None:
        return m

    try:
        with open(path) as f:
            model_data = json.load(f)
        learner = model_data.get("learner", {})
        lmp = learner.get("learner_model_param", {})
        num_class = int(lmp.get("num_class", "0"))
        n_classes = max(num_class, 2)
        obj_name = learner.get("objective", {}).get("name", "binary:logistic")
        if "multi" in obj_name or "softprob" in obj_name:
            n_classes = max(n_classes, 2)
        m.n_classes_ = n_classes
        fn_raw = learner.get("feature_names", "")
        if fn_raw:
            fn_list = json.loads(fn_raw) if isinstance(fn_raw, str) else fn_raw
            m._feature_names_in_ = np.array(fn_list)
    except Exception as e:
        logger.warning(f"无法从模型文件恢复 sklearn 属性: {e}")
        if not hasattr(m, "n_classes_") or m.n_classes_ is None:
            m.n_classes_ = 2
    return m

# ── 模型预测工具 ──────────────────────────────────────────────

def load_model(path: str) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(path)
    _restore_sklearn_attrs(m, path)
    return m

def predict(model: xgb.XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
    if feature_names is not None:
        avail = [f for f in feature_names if f in df.columns]
        X = df[avail]
    else:
        avail = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]
        X = df[avail]
    return model.predict_proba(X)[:, 1]

def submodel_ensemble_predict(
    df: pd.DataFrame,
    submodels: List[Dict],
    strategy: str = "routing",
) -> np.ndarray:
    """
    子模型集成预测。
    routing: 每个样本由其对应分群的子模型预测（互斥分群）。
    """
    preds = np.zeros(len(df))
    covered = np.zeros(len(df), dtype=bool)

    for sm in submodels:
        query = sm.get("query")
        if query:
            try:
                seg_idx = df.query(query).index.tolist()
            except Exception:
                seg_idx = []
        else:
            seg_idx = list(range(len(df)))

        if not seg_idx:
            continue

        seg_df = df.loc[seg_idx]
        seg_preds = predict(sm["model"], seg_df)
        for k, orig_idx in enumerate(seg_idx):
            preds[orig_idx] = seg_preds[k]
            covered[orig_idx] = True

    # 未覆盖样本用第一个子模型兜底
    if not covered.all() and submodels:
        uncovered = np.where(~covered)[0]
        fb_preds = predict(submodels[0]["model"], df.iloc[uncovered])
        preds[uncovered] = fb_preds

    return preds

# ── 单模型基线训练 ────────────────────────────────────────────

def train_baseline(
    train: pd.DataFrame, val: pd.DataFrame, target: str, output_dir: Path
) -> Tuple[xgb.XGBClassifier, List[str]]:
    exclude = {target}
    features = [c for c in train.columns if c not in exclude and train[c].dtype in ["int64", "float64"]]
    trainer = XGBTrainer()
    trainer.train(train[features], train[target], val[features], val[target])
    model = trainer.model
    baseline_path = output_dir / "baseline_single.json"
    model.save_model(str(baseline_path))
    logger.info(f"基线单模型已保存: {baseline_path}")
    return model, features

# ── 多算法额外基线（复用 model-comparison trainers）────────────────

def _parse_extra_algos(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    cand = [a.strip().lower() for a in raw.split(",") if a.strip()]
    # xgb 已由原生基线覆盖，额外基线只限 lr / dnn
    return [a for a in cand if a in ("lr", "dnn")]

def train_extra_baselines(
    algos: List[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    oot: pd.DataFrame,
    target: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """用同切分跑额外算法基线，返回 {algo: metrics_set}。

    某个算法失败仅 warning 后跳过，不阻断主流程。
    """
    if not algos:
        return {}

    try:
        from trainers import REGISTRY  # type: ignore
    except Exception as e:
        logger.warning(
            "[extra-baseline] 无法加载 model-comparison trainers，跳过多算法基线: %s", e,
        )
        return {}

    features = [
        c for c in train.columns
        if c != target and train[c].dtype in ["int64", "float64"]
    ]

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for algo in algos:
        if algo not in REGISTRY:
            logger.warning("[extra-baseline] 未知算法 %s，跳过", algo)
            continue
        try:
            logger.info("[extra-baseline][%s] 训练额外单模型基线", algo.upper())
            Trainer = REGISTRY[algo]
            res = Trainer().train(train, val, oot, features, target, {})
            if res is None:
                continue
            out[algo] = {
                "train": {"auc": float(res.train.auc), "ks": float(res.train.ks)},
                "val": {"auc": float(res.val.auc), "ks": float(res.val.ks)},
                "oot": {"auc": float(res.oot.auc), "ks": float(res.oot.ks)},
            }
            logger.info(
                "[extra-baseline][%s] OOT AUC=%.4f KS=%.4f",
                algo.upper(), res.oot.auc, res.oot.ks,
            )
        except Exception as e:
            logger.warning("[extra-baseline][%s] 训练失败，已跳过: %s", algo.upper(), e)
            traceback.print_exc()
    return out

# ── 报告生成 ──────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.4f}"

def _delta(a: float, b: float) -> str:
    d = a - b
    return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

def generate_report(
    baseline_m: Dict[str, Dict[str, float]],
    best_sub_m: Dict[str, Dict[str, float]],
    best_sub_name: str,
    stack_m: Optional[Dict[str, Dict[str, float]]],
    segment_comparison: List[Dict],
    has_stack: bool = True,
    extra_baselines: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Tuple[str, str]:
    """返回 (report_text, final_recommendation)"""
    extra_baselines = extra_baselines or {}
    _ALGO_LABEL = {"lr": "LR 基线", "dnn": "DNN 基线"}
    lines = ["# DeepModel — 最终对比报告", ""]

    # 汇总对比（KS 优先）
    lines += ["## 方案对比汇总", ""]
    rows: List[Tuple[str, Dict[str, Dict[str, float]]]] = [
        ("单模型基线（XGB）", baseline_m),
    ]
    for algo in ("lr", "dnn"):
        if algo in extra_baselines:
            rows.append((_ALGO_LABEL[algo], extra_baselines[algo]))
    rows.append((f"最优子模型（{best_sub_name}）", best_sub_m))
    if has_stack:
        rows.append(("Stacking 集成", stack_m))

    lines.append("| 方案 | Val KS | Val AUC | OOT KS | OOT AUC | KS Gap |")
    lines.append("|------|:------:|:-------:|:------:|:-------:|:------:|")
    for name, m in rows:
        tr_ks = m["train"].get("ks", 0)
        val_ks = m["val"].get("ks", 0)
        val_auc = m["val"].get("auc", 0)
        oot_ks = m["oot"].get("ks", 0)
        oot_auc = m["oot"].get("auc", 0)
        ks_gap = tr_ks - oot_ks
        lines.append(
            f"| {name} | {_fmt(val_ks)} | {_fmt(val_auc)} | {_fmt(oot_ks)} | {_fmt(oot_auc)} | {ks_gap:.4f} |"
        )
    if not has_stack:
        lines.append("| Stacking 集成 | — | — | — | — | — |")
    lines.append("")

    # 与基线的增量（KS 优先）
    baseline_oot_ks = baseline_m["oot"].get("ks", 0)
    best_sub_oot_ks = best_sub_m["oot"].get("ks", 0)
    stack_oot_ks = stack_m["oot"].get("ks", 0) if has_stack else 0

    lines += ["## 增量分析（vs 单模型 XGB 基线）", ""]
    lines.append(f"- 最优子模型 OOT KS 增量：{_delta(best_sub_oot_ks, baseline_oot_ks)}")
    if has_stack:
        lines.append(f"- Stacking 集成 OOT KS 增量：{_delta(stack_oot_ks, baseline_oot_ks)}")
    else:
        lines.append("- Stacking 集成：未执行（Stacking 训练失败，跳过集成对比）")
    for algo, m in extra_baselines.items():
        lines.append(
            f"- {_ALGO_LABEL.get(algo, algo.upper())} OOT KS 增量："
            f"{_delta(m['oot'].get('ks', 0), baseline_oot_ks)}"
        )
    lines.append("")

    # 各分群对比
    if segment_comparison:
        lines += ["## 各分群在不同方案下的 OOT KS", ""]
        if has_stack:
            lines.append("| 分群 | 单模型基线 | 子模型（专属） | Stacking 集成 |")
            lines.append("|------|:----------:|:--------------:|:-------------:|")
            for sc in segment_comparison:
                lines.append(
                    f"| {sc['name']} | {_fmt(sc['baseline_oot_ks'])} "
                    f"| {_fmt(sc['submodel_oot_ks'])} | {_fmt(sc['stack_oot_ks'])} |"
                )
        else:
            lines.append("| 分群 | 单模型基线 | 子模型（专属） |")
            lines.append("|------|:----------:|:--------------:|")
            for sc in segment_comparison:
                lines.append(
                    f"| {sc['name']} | {_fmt(sc['baseline_oot_ks'])} "
                    f"| {_fmt(sc['submodel_oot_ks'])} |"
                )
        lines.append("")

    # 最终推荐（KS 优先判决；额外基线仅作对照，不改变判决逻辑）
    lines += ["## 最终推荐", ""]
    if has_stack and stack_oot_ks > baseline_oot_ks + 0.005:
        rec = f"✅ **推荐使用 Stacking 集成**：OOT KS 较基线提升 {_delta(stack_oot_ks, baseline_oot_ks)}，分群集成策略有效。"
        recommendation = "stacking"
    elif best_sub_oot_ks > baseline_oot_ks + 0.005:
        rec = (
            f"⚠️ **推荐使用最优子模型（{best_sub_name}）**：Stacking 增益不足，"
            f"但该分群子模型比全局基线 KS 提升 {_delta(best_sub_oot_ks, baseline_oot_ks)}。"
            f"考虑将该分群单独建模，其余分群使用基线。"
        )
        recommendation = "best_submodel"
    else:
        rec = (
            "❌ **建议保留单模型基线**：分群集成未带来明显 KS 提升（OOT KS 差异 < 0.005）。"
            "当前分群方案可能不够合理，建议重新审视分群条件或特征设计。"
        )
        recommendation = "baseline"

    lines.append(rec)

    # 额外基线注释（提示用户换算法 vs 分群集成 的 ROI）
    if extra_baselines:
        lines.append("")
        lines.append("### 多算法基线观察")
        any_extra_beats = any(
            m["oot"].get("ks", 0) > baseline_oot_ks + 0.005 for m in extra_baselines.values()
        )
        if any_extra_beats:
            lines.append(
                "- 有单算法基线 OOT KS 超过 XGB 基线 0.005+，建议切换至 Agent 模式调用 "
                "`xgb-tuning → lr-tuning → dnn-tuning → model-comparison --use_tuned` 做公平多算法对比，"
                "确认是否换算法比分群集成更值得投入。"
            )
        else:
            lines.append(
                "- 额外单算法基线均未明显优于 XGB 基线，本层面无明显换算法收益。"
            )

    return "\n".join(lines), recommendation

# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = build_parser("deepmodel-compare", description="DeepModel 集成对比报告")
    args = parse_args(parser)
    emitter = ResultEmitter()

    submodel_paths = json.loads(args.submodel_paths) if isinstance(args.submodel_paths, str) else args.submodel_paths

    if args.segments:
        segments = json.loads(args.segments) if isinstance(args.segments, str) else args.segments
    elif args.segment_col:
        path = Path(args.data_path)
        tmp = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        vals = tmp[args.segment_col].dropna().unique()
        segments = {str(v): f"`{args.segment_col}` == {repr(v)}" for v in sorted(vals)}
    else:
        segments = {f"segment_{i}": None for i in range(len(submodel_paths))}

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_result = load_and_split_data(
        data_path=args.data_path,
        target=args.target,
        time_col=args.time_col,
        train_filter=args.train_filter,
        test_filter=args.val_filter,
        oot_filter=args.oot_filter,
    )
    train, val, oot = split_result.train, split_result.val, split_result.oot
    if split_result.meta:
        logger.info(f"[split-meta] {split_result.meta}")

    # SafetyGate 建模前体检
    gate_report = SafetyGate.check(
        train, val, oot,
        target_col=args.target,
        time_col=args.time_col,
    )
    if not gate_report.passed:
        for reason in gate_report.blocking_reasons:
            logger.error(f"SafetyGate 阻断: {reason}")
        raise RuntimeError(f"SafetyGate 未通过: {gate_report.blocking_reasons}")
    for w in gate_report.warnings:
        logger.warning(f"SafetyGate 告警: {w}")

    # 加载子模型
    names = list(segments.keys())
    submodels = []
    for i, p in enumerate(submodel_paths):
        submodels.append({
            "name": names[i] if i < len(names) else f"segment_{i}",
            "query": list(segments.values())[i] if i < len(segments) else None,
            "model": load_model(p),
        })

    # 加载 stacking 模型（可选）
    stack_model = None
    if args.stack_model_path:
        stack_model = load_model(args.stack_model_path)
    else:
        logger.warning("未提供 --stack_model_path，将仅对比基线 vs 子模型（Stacking 降级模式）")

    # 训练/加载基线
    if args.baseline_model_path:
        baseline_model = load_model(args.baseline_model_path)
        bl_features = baseline_model.feature_names_in_.tolist() if hasattr(baseline_model, "feature_names_in_") else None
    else:
        logger.info("未提供基线模型路径，自动训练全量单模型基线...")
        baseline_model, bl_features = train_baseline(train, val, args.target, output_dir)

    y_train = train[args.target].values
    y_val = val[args.target].values
    y_oot = oot[args.target].values

    # 各方案预测
    bl_train_p = predict(baseline_model, train)
    bl_val_p = predict(baseline_model, val)
    bl_oot_p = predict(baseline_model, oot)

    # Stacking 预测（仅在 stacking 模型可用时）
    def stack_predict(df: pd.DataFrame) -> Optional[np.ndarray]:
        if stack_model is None:
            return None
        sub_preds = np.column_stack([predict(sm["model"], df) for sm in submodels])
        return stack_model.predict_proba(sub_preds)[:, 1]

    has_stack = stack_model is not None
    st_val_p = stack_predict(val) if has_stack else None
    st_oot_p = stack_predict(oot) if has_stack else None

    # 最优子模型（routing 策略）
    best_sub_val_p = submodel_ensemble_predict(val, submodels)
    best_sub_oot_p = submodel_ensemble_predict(oot, submodels)

    # 计算各方案指标
    def metrics_set(y_t, p_t, y_v, p_v, y_o, p_o) -> Dict[str, Dict]:
        return {
            "train": ModelEvaluator.evaluate(y_t, p_t),
            "val": ModelEvaluator.evaluate(y_v, p_v),
            "oot": ModelEvaluator.evaluate(y_o, p_o),
        }

    baseline_m = metrics_set(y_train, bl_train_p, y_val, bl_val_p, y_oot, bl_oot_p)
    if has_stack:
        stack_m = metrics_set(y_train, stack_predict(train), y_val, st_val_p, y_oot, st_oot_p)
    else:
        stack_m = {"train": {"auc": 0}, "val": {"auc": 0, "ks": 0}, "oot": {"auc": 0, "ks": 0}}
    best_sub_m = metrics_set(
        y_train, submodel_ensemble_predict(train, submodels),
        y_val, best_sub_val_p,
        y_oot, best_sub_oot_p,
    )

    # 各分群对比
    segment_comparison = []
    for sm in submodels:
        query = sm["query"]
        if query:
            try:
                seg_oot = oot.query(query).reset_index(drop=True)
            except Exception:
                seg_oot = oot
        else:
            seg_oot = oot
        if len(seg_oot) < 30:
            continue

        y_seg = seg_oot[args.target].values
        try:
            bl_ks = ModelEvaluator.evaluate(y_seg, predict(baseline_model, seg_oot))["ks"]
            sub_ks = ModelEvaluator.evaluate(y_seg, predict(sm["model"], seg_oot))["ks"]
            st_ks = ModelEvaluator.evaluate(y_seg, stack_predict(seg_oot))["ks"] if has_stack else 0.0
            segment_comparison.append({
                "name": sm["name"],
                "baseline_oot_ks": bl_ks,
                "submodel_oot_ks": sub_ks,
                "stack_oot_ks": st_ks,
            })
        except Exception as e:
            logger.warning(f"[{sm['name']}] 分群对比失败: {e}")

    # 最优子模型名称（改为 KS）
    best_sub_name = max(
        submodels,
        key=lambda sm: ModelEvaluator.evaluate(
            oot[args.target].values,
            predict(sm["model"], oot)
        )["ks"] if len(oot) > 0 else 0
    )["name"] if submodels else "—"

    # 额外多算法基线（可选）
    extra_algos = _parse_extra_algos(getattr(args, "extra_baseline_algos", ""))
    if extra_algos:
        logger.info("[extra-baseline] 将训练额外单模型基线: %s", extra_algos)
    extra_baselines = train_extra_baselines(
        extra_algos, train, val, oot, args.target,
    )

    report, recommendation = generate_report(
        baseline_m, best_sub_m, best_sub_name, stack_m, segment_comparison,
        has_stack=has_stack,
        extra_baselines=extra_baselines,
    )
    print(report, flush=True)

    state = {
        "deepmodel_phase": "done",
        "final_recommendation": recommendation,
        "ensemble_gain": {
            "stacking_vs_baseline_ks": round(stack_m["oot"]["ks"] - baseline_m["oot"]["ks"], 6) if has_stack else None,
            "best_sub_vs_baseline_ks": round(best_sub_m["oot"]["ks"] - baseline_m["oot"]["ks"], 6),
        },
        "metrics": {
            "baseline": baseline_m["oot"],
            "best_submodel": best_sub_m["oot"],
            "stacking": stack_m["oot"] if has_stack else None,
            "extra_baselines": {a: m["oot"] for a, m in extra_baselines.items()},
        },
        "safety_gate": gate_report.to_dict(),
    }
    emitter.update_state(state)
    emitter.set_summary(f"DeepModel对比完成: 推荐={state.get('recommendation', 'N/A')}")
    emitter.write(output_dir / "result.json")
    return 0

if __name__ == "__main__":
    main()
