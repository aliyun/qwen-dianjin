# -*- coding: utf-8 -*-
"""
DeepModel — XGBoost Stacking 融合

从 project_state 读取各分群子模型，生成 OOF 预测分，训练 meta-learner。

Usage:
    python stacker.py --data_path ./data.parquet --target y_label \
        --submodel_paths '["/path/高风险.json", "/path/低风险.json"]' \
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
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from xgb_core import XGBTrainer, ModelEvaluator  # noqa: E402
from data_utils import load_and_split_data  # noqa: E402
from xgb_cli import build_parser, parse_args  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("./outputs/deepmodel")

# Meta-learner 固定保守参数，防止过拟合
META_PARAMS = {
    "max_depth": 2,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "random_state": 42,
}

# sklearn-compatible XGBClassifier.get_params() 返回的元参数/内部参数，
# 在创建新 fold 模型时需要排除，避免干扰原生 xgb 训练。
_SKLEARN_META_KEYS = {
    'missing', 'n_jobs', 'nthread', 'silent',
    'random_state', 'verbosity', 'early_stopping_rounds',
    'eval_metric', 'objective', 'booster', 'importance_type',
    'device', 'enable_categorical', 'max_cat_to_onehot',
    'gpu_id', 'predictor', 'tree_method', 'scale_pos_weight',
    'base_score', 'callbacks', 'use_label_encoder',
}

# 数据加载统一使用 shared/data_utils.load_and_split_data

# ── 模型加载兼容工具 ──────────────────────────────────────────

def _restore_sklearn_attrs(m: xgb.XGBClassifier, path: str) -> xgb.XGBClassifier:
    """XGBoost 2.x load_model() 不恢复 sklearn 属性，手动补回。

    从模型 JSON 中读取 objective / num_class / feature_names，
    设置 n_classes_ 和 _feature_names_in_ 以支持 predict_proba()。
    """
    if hasattr(m, "n_classes_") and m.n_classes_ is not None:
        return m  # 已有属性，无需修复

    try:
        with open(path) as f:
            model_data = json.load(f)
        learner = model_data.get("learner", {})

        # 解析 num_class: 二分类时为 0，多分类时 > 1
        lmp = learner.get("learner_model_param", {})
        num_class = int(lmp.get("num_class", "0"))
        n_classes = max(num_class, 2)  # 0 表示二分类 → 2

        # 解析 objective 辅助判断
        obj_name = learner.get("objective", {}).get("name", "binary:logistic")
        if "multi" in obj_name or "softprob" in obj_name:
            n_classes = max(n_classes, 2)

        # n_classes_ 存在 __dict__ 中，可直接赋值
        m.n_classes_ = n_classes

        # feature_names_in_ 是 property，需通过 _feature_names_in_ 赋值
        fn_raw = learner.get("feature_names", "")
        if fn_raw:
            fn_list = json.loads(fn_raw) if isinstance(fn_raw, str) else fn_raw
            m._feature_names_in_ = np.array(fn_list)
    except Exception as e:
        logger.warning(f"无法从模型文件恢复 sklearn 属性: {e}")
        # 兜底：二分类默认值
        if not hasattr(m, "n_classes_") or m.n_classes_ is None:
            m.n_classes_ = 2

    return m

# ── 子模型加载 ────────────────────────────────────────────────

def load_submodels(submodel_paths: List[str], segments: Dict[str, str]) -> List[Dict]:
    """加载子模型，将路径与分群条件对齐。"""
    names = list(segments.keys())
    models = []
    for i, path in enumerate(submodel_paths):
        m = xgb.XGBClassifier()
        m.load_model(path)
        _restore_sklearn_attrs(m, path)
        models.append({
            "name": names[i] if i < len(names) else f"segment_{i}",
            "query": list(segments.values())[i] if i < len(segments) else None,
            "model": m,
            "path": path,
        })
    return models

# ── OOF 预测生成 ──────────────────────────────────────────────

def generate_oof_predictions(
    train: pd.DataFrame,
    target: str,
    submodels: List[Dict],
    cv_folds: int,
) -> pd.DataFrame:
    """
    在训练集上用 k-fold 生成 OOF 预测分，避免标签泄漏。
    每个子模型的 OOF 预测在其对应分群内独立做 k-fold。
    不属于任何分群的样本用全局均值填充。
    """
    n = len(train)
    oof_matrix = np.full((n, len(submodels)), np.nan)

    for j, sm in enumerate(submodels):
        query = sm["query"]
        model = sm["model"]
        feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None

        if query:
            seg_idx = train.query(query).index.tolist()
        else:
            seg_idx = list(range(n))

        if not seg_idx:
            logger.warning(f"[{sm['name']}] 训练集中无匹配样本，跳过 OOF 生成")
            continue

        seg_data = train.loc[seg_idx]
        y_seg = seg_data[target].values

        if feature_names is not None:
            avail = [f for f in feature_names if f in seg_data.columns]
        else:
            avail = [c for c in seg_data.columns if c != target and seg_data[c].dtype in ["int64", "float64"]]

        X_seg = seg_data[avail].values

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(seg_idx))

        # 从已加载的子模型提取原始训练参数（而非 meta-learner 的保守参数），
        # 确保 OOF 预测分能准确代表子模型的预测行为。
        sub_params = model.get_params()
        fold_params = {
            k: v for k, v in sub_params.items()
            if k not in _SKLEARN_META_KEYS and v is not None
        }
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_seg, y_seg)):
            fold_model = xgb.XGBClassifier(**fold_params)
            fold_model.fit(
                X_seg[tr_idx], y_seg[tr_idx],
                eval_set=[(X_seg[va_idx], y_seg[va_idx])],
                verbose=False,
            )
            oof_preds[va_idx] = fold_model.predict_proba(X_seg[va_idx])[:, 1]

        for k, orig_idx in enumerate(seg_idx):
            oof_matrix[orig_idx, j] = oof_preds[k]

    # 用列均值填充 NaN（不属于任何分群的样本）
    col_means = np.nanmean(oof_matrix, axis=0)
    for j in range(oof_matrix.shape[1]):
        mask = np.isnan(oof_matrix[:, j])
        oof_matrix[mask, j] = col_means[j]

    cols = [f"oof_{sm['name']}" for sm in submodels]
    return pd.DataFrame(oof_matrix, columns=cols)

def get_submodel_preds(df: pd.DataFrame, submodels: List[Dict]) -> np.ndarray:
    """在给定数据集上获取各子模型预测分，组成特征矩阵。"""
    preds = []
    for sm in submodels:
        model = sm["model"]
        feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
        if feature_names is not None:
            avail = [f for f in feature_names if f in df.columns]
            X = df[avail]
        else:
            avail = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]
            X = df[avail]
        p = model.predict_proba(X)[:, 1]
        preds.append(p)
    return np.column_stack(preds)

# ── Meta-learner 训练 ─────────────────────────────────────────

def train_meta_learner(
    X_meta_train: np.ndarray, y_train: np.ndarray,
    X_meta_val: np.ndarray, y_val: np.ndarray,
) -> xgb.XGBClassifier:
    meta = xgb.XGBClassifier(**META_PARAMS)
    meta.fit(
        X_meta_train, y_train,
        eval_set=[(X_meta_val, y_val)],
        verbose=False,
    )
    return meta

# ── 报告生成 ──────────────────────────────────────────────────

def generate_report(
    submodels: List[Dict],
    oof_aucs: List[float],
    val_metrics: Dict[str, float],
    oot_metrics: Dict[str, float],
    best_submodel_oot: float,
) -> str:
    lines = ["# DeepModel — Stacking 融合报告", ""]

    # OOF AUC 对比
    lines += ["## OOF 预测分质量（各子模型）", ""]
    lines.append("| 子模型 | OOF AUC |")
    lines.append("|--------|:-------:|")
    for sm, auc in zip(submodels, oof_aucs):
        lines.append(f"| {sm['name']} | {auc:.4f} |")
    lines.append("")

    # Stacking 指标
    lines += ["## Stacking Meta-learner 指标", ""]
    lines.append("| 数据集 | AUC | KS |")
    lines.append("|--------|:---:|:--:|")
    for split_name, m in [("验证集", val_metrics), ("OOT", oot_metrics)]:
        lines.append(f"| {split_name} | {m.get('auc', 0):.4f} | {m.get('ks', 0):.4f} |")
    lines.append("")

    stack_oot = oot_metrics.get("auc", 0)
    gain = stack_oot - best_submodel_oot
    if gain > 0.005:
        lines.append(f"✅ Stacking OOT AUC 较最优子模型提升 **+{gain:.4f}**，集成有效。")
    elif gain > -0.01:
        lines.append(f"⚠️ Stacking OOT AUC 与最优子模型持平（差异 {gain:+.4f}），集成增益有限。")
    else:
        lines.append(f"❌ Stacking OOT AUC 较最优子模型下降 {gain:.4f}，建议检查分群设计。")

    lines += [
        "",
        "## 下一步",
        "",
        "Stacking 融合完成，正在生成与单模型基线的对比报告...",
    ]

    return "\n".join(lines)

# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = build_parser("deepmodel-stack", description="DeepModel Stacking 融合")
    args = parse_args(parser)

    submodel_paths = json.loads(args.submodel_paths) if isinstance(args.submodel_paths, str) else args.submodel_paths

    # 分群条件解析
    if args.segments:
        segments = json.loads(args.segments) if isinstance(args.segments, str) else args.segments
    elif args.segment_col:
        # 推断分群：读文件取唯一值
        path = Path(args.data_path)
        tmp = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        vals = tmp[args.segment_col].dropna().unique()
        segments = {str(v): f"`{args.segment_col}` == {repr(v)}" for v in sorted(vals)}
    else:
        segments = {f"segment_{i}": None for i in range(len(submodel_paths))}

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="xgb-deepmodel", output_dir=output_dir)

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

    submodels = load_submodels(submodel_paths, segments)
    logger.info(f"已加载 {len(submodels)} 个子模型")

    # 生成 OOF 预测
    logger.info("生成 OOF 预测分...")
    oof_df = generate_oof_predictions(train, args.target, submodels, args.cv_folds)
    y_train = train[args.target].values

    # 计算各子模型 OOF AUC
    oof_aucs = []
    for col in oof_df.columns:
        try:
            oof_aucs.append(round(roc_auc_score(y_train, oof_df[col].values), 4))
        except Exception:
            oof_aucs.append(0.0)

    # 在 val/oot 上获取子模型预测
    X_meta_val = get_submodel_preds(val, submodels)
    X_meta_oot = get_submodel_preds(oot, submodels)
    y_val = val[args.target].values
    y_oot = oot[args.target].values

    # 训练 meta-learner
    logger.info("训练 meta-learner...")
    meta_model = train_meta_learner(oof_df.values, y_train, X_meta_val, y_val)

    val_metrics = ModelEvaluator.evaluate(y_val, meta_model.predict_proba(X_meta_val)[:, 1])
    oot_metrics = ModelEvaluator.evaluate(y_oot, meta_model.predict_proba(X_meta_oot)[:, 1])

    # 最优子模型 OOT AUC（用于对比）
    best_submodel_oot = max(
        roc_auc_score(y_oot, sm["model"].predict_proba(
            oot[[f for f in (sm["model"].feature_names_in_ if hasattr(sm["model"], "feature_names_in_") else []) if f in oot.columns]]
        )[:, 1])
        for sm in submodels
        if len(oot) > 0
    ) if oot_metrics else 0.0

    meta_path = output_dir / "stack_meta.json"
    meta_model.save_model(str(meta_path))

    report = generate_report(submodels, oof_aucs, val_metrics, oot_metrics, best_submodel_oot)
    print(report, flush=True)

    state = {
        "deepmodel_phase": "stacking_done",
        "stack_model_path": str(meta_path),
        "stack_metrics": {
            "val_auc": val_metrics.get("auc", 0),
            "val_ks": val_metrics.get("ks", 0),
            "oot_auc": oot_metrics.get("auc", 0),
            "oot_ks": oot_metrics.get("ks", 0),
        },
        "oof_aucs": dict(zip([sm["name"] for sm in submodels], oof_aucs)),
        "best_submodel_oot_auc": round(best_submodel_oot, 6),
    }
    emitter.update_state(state)
    emitter.add_file(str(meta_path), role="meta", meta={"scheme_name": "stack_meta"})
    emitter.set_summary(f"Stacking完成: OOT AUC={oot_metrics.get('auc', 0):.4f}")
    emitter.write(output_dir / "result.json")
    return 0

if __name__ == "__main__":
    main()
