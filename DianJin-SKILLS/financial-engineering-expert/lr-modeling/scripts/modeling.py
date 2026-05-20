# -*- coding: utf-8 -*-
"""
LR 评分卡建模主入口脚本

基于 WoE 编码 + Logistic Regression 训练标准评分卡模型。
复用平台已有的三段式数据切分、评估体系、稳定性分析。

Usage:
    python modeling.py --data_path ./data.parquet --target y_label

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

from data_utils import load_and_split_data, render_split_section  # noqa: E402
from data.woe_transform import WoEEncoder  # noqa: E402
from lr_cli import build_parser, parse_args  # noqa: E402
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
# 评分卡转换
# ============================================================

class ScoreCardConverter:
    """将 LR 模型 + WoE 转换为标准评分卡。

    评分卡公式：
        Score = base_score - factor * ln(odds)
        factor = pdo / ln(2)

    每个特征每个分箱的分数贡献：
        score_i = -(coef_i * woe_ij + intercept/n) * factor
    """

    def __init__(
        self,
        base_score: int = 600,
        pdo: int = 50,
        base_odds: float = 50.0,
    ) -> None:
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def convert(
        self,
        lr_model: LogisticRegression,
        woe_encoder: WoEEncoder,
    ) -> pd.DataFrame:
        """生成评分卡表。

        Returns:
            DataFrame with columns [feature, bin, woe, coef, score]
        """
        coefs = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]
        n_features = len(woe_encoder.fitted_features)

        rows = []
        for i, feat in enumerate(woe_encoder.fitted_features):
            coef = coefs[i]
            optb = woe_encoder._binners[feat]
            table_df = optb.binning_table.build()

            # 排除 Totals 行
            for _, row in table_df.iterrows():
                bin_label = str(row["Bin"])
                if bin_label in ("Totals", "Total"):
                    continue
                woe_raw = row["WoE"]
                woe_val = 0.0 if (pd.isna(woe_raw) or woe_raw == "" or woe_raw is None) else float(woe_raw)
                # 分数 = -(系数 × WoE + 截距均摊) × factor
                score_contribution = -(coef * woe_val + intercept / n_features) * self.factor
                rows.append({
                    "feature": feat,
                    "bin": bin_label,
                    "woe": round(woe_val, 6),
                    "coef": round(coef, 6),
                    "score": round(score_contribution, 2),
                })

        return pd.DataFrame(rows)

    def predict_score(
        self,
        lr_model: LogisticRegression,
        X_woe: pd.DataFrame,
    ) -> np.ndarray:
        """预测评分卡分数（越高越好/风险越低）。"""
        # log_odds = X @ coef + intercept
        log_odds = X_woe.values @ lr_model.coef_[0] + lr_model.intercept_[0]
        scores = self.offset - self.factor * log_odds
        return scores

# ============================================================
# 模型训练
# ============================================================

def train_lr_model(
    X_train_woe: pd.DataFrame,
    y_train: np.ndarray,
    X_val_woe: pd.DataFrame,
    y_val: np.ndarray,
    regularization: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """训练 LR 模型。

    Args:
        X_train_woe: WoE 编码后的训练集
        y_train: 训练集标签
        X_val_woe: WoE 编码后的验证集
        y_val: 验证集标签
        regularization: 正则化类型
        C: 正则化强度
        max_iter: 最大迭代次数

    Returns:
        (lr_model, training_info)
    """
    # 选择 solver
    if regularization == "l1":
        solver = "saga"
        penalty = "l1"
    elif regularization == "elasticnet":
        solver = "saga"
        penalty = "elasticnet"
    else:
        solver = "lbfgs"
        penalty = "l2"

    lr_params = {
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": 42,
        "n_jobs": -1,
    }
    if penalty == "elasticnet":
        lr_params["l1_ratio"] = 0.5

    model = LogisticRegression(**lr_params)
    model.fit(X_train_woe, y_train)

    # 训练信息
    train_prob = model.predict_proba(X_train_woe)[:, 1]
    val_prob = model.predict_proba(X_val_woe)[:, 1]
    train_auc = roc_auc_score(y_train, train_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    info = {
        "n_features": X_train_woe.shape[1],
        "regularization": regularization,
        "C": C,
        "train_auc": round(train_auc, 6),
        "val_auc": round(val_auc, 6),
        "converged": model.n_iter_[0] < max_iter,
        "n_iter": int(model.n_iter_[0]),
    }

    logger.info(
        f"LR 训练完成: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, "
        f"迭代 {model.n_iter_[0]} 轮, {'收敛' if info['converged'] else '⚠️ 未收敛'}"
    )

    return model, info

# ============================================================
# 系数分析
# ============================================================

def analyze_coefficients(
    model: LogisticRegression,
    feature_names: List[str],
    iv_table: Dict[str, float],
) -> pd.DataFrame:
    """分析 LR 系数，检查符号一致性。

    Returns:
        DataFrame with columns [feature, coef, abs_coef, iv, sign_consistent]
    """
    coefs = model.coef_[0]
    rows = []
    for i, feat in enumerate(feature_names):
        coef = coefs[i]
        iv = iv_table.get(feat, 0)
        # WoE 编码下，正系数意味着 WoE 越大（好客户占比越多）→ 越不容易违约
        # 正常情况下 LR 系数应为正（WoE 正方向 = 好客户方向）
        sign_consistent = coef > 0
        rows.append({
            "feature": feat,
            "coef": round(coef, 6),
            "abs_coef": round(abs(coef), 6),
            "iv": round(iv, 6),
            "sign_consistent": sign_consistent,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df

# ============================================================
# 报告生成
# ============================================================

def generate_report(
    data_path: str,
    target: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    time_col: str,
    woe_encoder: WoEEncoder,
    lr_model: LogisticRegression,
    training_info: Dict[str, Any],
    coef_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    eval_results: Dict[str, Dict[str, float]],
    stability_report: Optional[Dict[str, Any]],
    split_meta: Optional[Dict[str, Any]] = None,
    args: Any = None,
) -> str:
    """生成 LR 评分卡建模报告。"""
    lines = []

    # 报告头部
    lines.append("# LR 评分卡建模报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 数据文件：{data_path}")
    lines.append(f"> 目标变量：{target}")
    lines.append(f"> 算法：Logistic Regression + WoE 编码")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 执行摘要
    oot_metrics = eval_results.get("oot", {})
    val_metrics = eval_results.get("val", {})
    train_metrics = eval_results.get("train", {})
    gap_auc = train_metrics.get("auc", 0) - oot_metrics.get("auc", 0)
    gap_ks = train_metrics.get("ks", 0) - oot_metrics.get("ks", 0)

    lines.append("## 执行摘要")
    lines.append("")
    lines.append(f"- **入模特征**：{len(woe_encoder.fitted_features)} 个（WoE 编码后）")
    lines.append(f"- **OOT 效果**：AUC={oot_metrics.get('auc', 0):.4f}，KS={oot_metrics.get('ks', 0):.4f}")
    lines.append(f"- **泛化性**：AUC Gap={gap_auc:.4f}（{'良好' if gap_auc < 0.03 else '需关注' if gap_auc < 0.05 else '⚠️ 过拟合风险'}），"
                 f"KS Gap={gap_ks:.4f}（{'良好' if gap_ks < 0.05 else '需关注' if gap_ks < 0.10 else '⚠️ 过拟合风险'}）")
    lines.append(f"- **正则化**：{args.regularization if args else 'l2'}, C={args.C if args else 1.0}")
    # 符号一致性检查
    inconsistent = coef_df[~coef_df["sign_consistent"]]
    if len(inconsistent) > 0:
        lines.append(f"- **⚠️ 系数方向异常**：{len(inconsistent)} 个特征系数为负（需关注业务含义）")
    else:
        lines.append("- **系数方向**：全部一致（✓）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 数据概览
    lines.append("## 1. 数据概览")
    lines.append("")
    if split_meta:
        split_md = render_split_section(split_meta)
        if split_md:
            lines.append(split_md)
            lines.append("")
    total_samples = len(train_data) + len(val_data) + len(oot_data)
    lines.append(f"| 数据集 | 样本量 | 正样本量 | 正样本率 |")
    lines.append("|--------|--------|----------|----------|")
    for name, df in [("Train", train_data), ("Val", val_data), ("OOT", oot_data)]:
        n = len(df)
        pos = int(df[target].sum())
        rate = df[target].mean()
        lines.append(f"| {name} | {n:,} | {pos:,} | {rate:.4%} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. WoE 分箱摘要
    lines.append("## 2. WoE 分箱与特征筛选")
    lines.append("")
    iv_summary = woe_encoder.get_iv_summary()
    lines.append(f"共 {len(iv_summary)} 个特征通过 IV 筛选（阈值={args.iv_threshold if args else 0.02}）：")
    lines.append("")
    lines.append("| 特征 | IV | IV等级 | 分箱数 |")
    lines.append("|------|-----|--------|--------|")
    for _, row in iv_summary.iterrows():
        lines.append(f"| {row['feature']} | {row['iv']:.4f} | {row['iv_level']} | {row['n_bins']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. LR 系数表
    lines.append("## 3. LR 系数分析")
    lines.append("")
    lines.append(f"- 正则化：{training_info['regularization']}，C={args.C if args else 1.0}")
    lines.append(f"- 收敛状态：{'✓ 收敛' if training_info['converged'] else '⚠️ 未收敛'}（{training_info['n_iter']} 轮）")
    lines.append("")
    lines.append("| 特征 | 系数 | |系数| | IV | 方向一致 |")
    lines.append("|------|------|--------|-----|----------|")
    for _, row in coef_df.iterrows():
        consistent = "✓" if row["sign_consistent"] else "⚠️"
        lines.append(f"| {row['feature']} | {row['coef']:.4f} | {row['abs_coef']:.4f} | {row['iv']:.4f} | {consistent} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. 评估指标
    lines.append("## 4. 模型评估")
    lines.append("")
    lines.append("### 4.1 评估指标")
    lines.append("")
    lines.append("| 指标 | Train | Val | OOT |")
    lines.append("|------|-------|-----|-----|")
    for metric in ["auc", "ks", "gini"]:
        tr = train_metrics.get(metric, 0)
        va = val_metrics.get(metric, 0)
        oot = oot_metrics.get(metric, 0)
        lines.append(f"| {metric.upper()} | {tr:.4f} | {va:.4f} | {oot:.4f} |")
    lines.append("")

    # 5. 评分卡转换表（取前 30 行避免过长）
    lines.append("## 5. 评分卡转换表")
    lines.append("")
    lines.append(f"> 基础分={args.base_score if args else 600}，PDO={args.pdo if args else 50}")
    lines.append("")
    display_df = scorecard_df.head(50)
    lines.append(display_df.to_markdown(index=False))
    lines.append("")

    # 6. 稳定性分析（完整段落：月度 AUC/KS 表 + 趋势 + echarts + 月度 PSI 表）
    lines.append(render_monthly_stability_section(
        stability_report,
        heading="6. 稳定性分析",
        heading_level=2,
    ))

    return "\n".join(lines)

# ============================================================
# 主函数
# ============================================================

def main():
    # ── 参数解析 ──
    parser = build_parser()
    args = parse_args(parser)

    # ── output_dir ──
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="lr-modeling", output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("LR 评分卡建模开始")
    logger.info("=" * 60)

    # ── 数据加载与切分（复用三段式） ──
    logger.info(f"加载数据: {args.data_path}")
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

    logger.info(f"数据切分完成: train={len(train_data)}, val={len(val_data)}, oot={len(oot_data)}")

    # ── 确定入模特征 ──
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
        logger.info(f"使用指定特征: {len(features)} 个")
    else:
        features = all_features
        logger.info(f"使用自动推断特征: {len(features)} 个（将由 IV 筛选）")

    # ── SafetyGate 检查 ──
    gate = SafetyGate()
    gate_result = gate.check(train_data, val_data, oot_data, target_col=args.target, features=features)
    if not gate_result.passed:
        logger.warning(f"SafetyGate 未通过: {'; '.join(gate_result.blocking_reasons)}")
        # 不中断，仅警告

    # ── WoE 编码 ──
    logger.info("执行 WoE 分箱编码...")
    woe_encoder = WoEEncoder(
        max_n_bins=args.max_n_bins,
        min_bin_size=args.min_bin_size,
        iv_threshold=args.iv_threshold,
        selection_mode="iv_filter",  # 自动按 IV 筛选
    )

    y_train = train_data[args.target].values
    y_val = val_data[args.target].values
    y_oot = oot_data[args.target].values

    woe_encoder.fit(train_data, y_train, features)

    if not woe_encoder.fitted_features:
        logger.error("没有特征通过 WoE 编码，建模中止")
        print(f"[ERROR] 所有特征 IV < {args.iv_threshold}，无法建模", file=sys.stderr)
        sys.exit(1)

    X_train_woe = woe_encoder.transform(train_data)
    X_val_woe = woe_encoder.transform(val_data)
    X_oot_woe = woe_encoder.transform(oot_data)

    logger.info(f"WoE 编码完成: {len(woe_encoder.fitted_features)} 个特征入模")

    # ── LR 训练 ──
    logger.info("训练 Logistic Regression...")
    lr_model, training_info = train_lr_model(
        X_train_woe, y_train,
        X_val_woe, y_val,
        regularization=args.regularization,
        C=args.C,
        max_iter=args.max_iter,
    )

    # ── 预测与评估（复用 ModelEvaluator） ──
    prob_train = lr_model.predict_proba(X_train_woe)[:, 1]
    prob_val = lr_model.predict_proba(X_val_woe)[:, 1]
    prob_oot = lr_model.predict_proba(X_oot_woe)[:, 1]

    eval_train = ModelEvaluator.evaluate(y_train, prob_train)
    eval_val = ModelEvaluator.evaluate(y_val, prob_val)
    eval_oot = ModelEvaluator.evaluate(y_oot, prob_oot)

    eval_results = {"train": eval_train, "val": eval_val, "oot": eval_oot}

    logger.info(f"评估: Train AUC={eval_train['auc']:.4f}, Val AUC={eval_val['auc']:.4f}, OOT AUC={eval_oot['auc']:.4f}")
    logger.info(f"评估: Train KS={eval_train['ks']:.4f}, Val KS={eval_val['ks']:.4f}, OOT KS={eval_oot['ks']:.4f}")

    # ── 系数分析 ──
    coef_df = analyze_coefficients(lr_model, woe_encoder.fitted_features, woe_encoder.iv_table)

    # ── 评分卡转换 ──
    converter = ScoreCardConverter(
        base_score=args.base_score,
        pdo=args.pdo,
        base_odds=args.base_odds,
    )
    scorecard_df = converter.convert(lr_model, woe_encoder)

    # ── 稳定性分析 ──
    stability_report = None
    if args.time_col and args.time_col in oot_data.columns:
        try:
            analyzer = ModelStabilityAnalyzer()
            all_data = pd.concat([train_data, val_data, oot_data], ignore_index=True)
            all_data["_lr_score"] = np.concatenate([prob_train, prob_val, prob_oot])
            months = sorted(all_data[args.time_col].astype(str).str[:6].unique())
            if len(months) >= 2:
                baseline_months = months[:max(1, len(months) // 2)]
                stability_report = analyzer.full_stability_report(
                    all_data, args.time_col, args.target, "_lr_score", baseline_months
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
        time_col=args.time_col,
        woe_encoder=woe_encoder,
        lr_model=lr_model,
        training_info=training_info,
        coef_df=coef_df,
        scorecard_df=scorecard_df,
        eval_results=eval_results,
        stability_report=stability_report,
        split_meta=split_meta,
        args=args,
    )

    # 报告渲染：追加评估图表与明细表（与 XGB 报告结构对齐）
    # 特征重要性：LR 以 |coef| 作为重要性（输入已 WoE 标准化，可直接比较）
    lr_importance = {
        feat: float(abs(c))
        for feat, c in zip(woe_encoder.fitted_features, lr_model.coef_[0])
    }

    report_content += "\n\n## 7. 评估图表\n\n"
    chart_renderers = [
        ("模型分数 IV 分箱", lambda: render_score_iv_table_section(
            y_oot, prob_oot, heading="7.1 模型分数 IV 最优分箱（OOT）", heading_level=3)),
        ("KS 曲线", lambda: render_ks_curve(y_oot, prob_oot, prefix="OOT ")),
        ("ROC 曲线", lambda: render_roc_curve(y_oot, prob_oot, prefix="OOT ")),
        ("Lift 明细表", lambda: render_lift_detail_table_section(
            y_oot, prob_oot, heading="7.2 Lift 明细表（OOT）", heading_level=3)),
        ("Lift Chart", lambda: render_lift_chart(y_oot, prob_oot, prefix="OOT ")),
        ("Lift Chart（IV最优分箱）", lambda: render_lift_chart_optbinning(y_oot, prob_oot, prefix="OOT ")),
        ("特征重要性", lambda: render_feature_importance(
            lr_importance, top_n=15, prefix="LR ")),
        ("坏客户捕获率 明细", lambda: render_bcr_detail_table_section(
            y_oot, prob_oot, heading="7.3 坏客户捕获率（OOT）", heading_level=3)),
        ("Bad Capture Rate", lambda: render_bad_capture_rate(y_oot, prob_oot, prefix="OOT ")),
    ]
    for chart_name, renderer in chart_renderers:
        try:
            report_content += renderer() + "\n"
        except Exception as e:
            logger.warning(f"图表 {chart_name} 渲染失败: {e}")
            report_content += f"> ⚠️ 图表 **{chart_name}** 渲染失败: {e}\n\n"

    # ── 保存产物 ──
    # 报告
    model_name = args.model_name or f"lr_scorecard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_path = output_dir / f"{model_name}_report.md"
    report_path.write_text(report_content, encoding='utf-8')
    emitter.add_report(report_path)

    # 模型文件
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    model_bundle = {
        "lr_model": lr_model,
        "woe_encoder": woe_encoder,
        "features": woe_encoder.fitted_features,
        "training_info": training_info,
        "eval_results": eval_results,
        "scorecard_params": {
            "base_score": args.base_score,
            "pdo": args.pdo,
            "base_odds": args.base_odds,
        },
    }
    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(model_bundle, model_path)
    emitter.add_model(model_path, model_type="lr_scorecard")

    # 评分卡 JSON
    scorecard_path = models_dir / f"{model_name}_scorecard.json"
    scorecard_dict = {
        "base_score": args.base_score,
        "pdo": args.pdo,
        "features": woe_encoder.fitted_features,
        "card": scorecard_df.to_dict(orient="records"),
    }
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard_dict, f, ensure_ascii=False, indent=2)
    emitter.add_file(scorecard_path, role="asset", meta={"type": "scorecard"})

    # ── State 更新 ──
    emitter.update_state({
        "model_result": {
            "algorithm": "lr_scorecard",
            "scheme_name": f"WoE+LR ({len(woe_encoder.fitted_features)}特征)",
            "features": woe_encoder.fitted_features,
            "n_features": len(woe_encoder.fitted_features),
            "metrics": eval_results,
            "gap": round(eval_train["ks"] - eval_oot["ks"], 4),
            "gap_auc": round(eval_train["auc"] - eval_oot["auc"], 4),
            "psi": stability_report.get("overall_score_psi") if stability_report else None,
        },
    })

    # ── Artifact（供前端渲染） ──
    emitter.add_artifact("metrics-table", "LR 评分卡评估指标", {
        "metrics": eval_results,
        "model_type": "lr_scorecard",
    })

    # ── 输出 ──
    emitter.set_summary(f"LR评分卡建模完成: OOT AUC={eval_oot['auc']:.4f}, KS={eval_oot['ks']:.4f}")
    emitter.update_metrics({"oot_auc": eval_oot["auc"], "oot_ks": eval_oot["ks"], "features_count": len(woe_encoder.fitted_features)})
    emitter.write(output_dir / "result.json")

    logger.info("=" * 60)
    logger.info(f"LR 评分卡建模完成 ✓")
    logger.info(f"  模型: {model_path}")
    logger.info(f"  报告: {report_path}")
    logger.info(f"  评分卡: {scorecard_path}")
    logger.info(f"  result.json: {output_dir / 'result.json'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
