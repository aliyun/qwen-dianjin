# -*- coding: utf-8 -*-
"""
XGBoost 建模主入口脚本

支持多特征方案对比、稳定性分析，生成结构化建模报告。
参数调优请使用独立的 xgb-tuning 技能。

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

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))
# 同目录的 feature_selector 也要可导入
sys.path.insert(0, str(_SCRIPT_DIR))

from xgb_core import XGBTrainer, ModelEvaluator, ModelStabilityAnalyzer  # noqa: E402
from feature_selector import FeatureSelector  # noqa: E402

from data_utils import load_and_split_data, build_xgb_eval_set, render_split_section  # noqa: E402
# 6 件套评估图表（KS/ROC/Lift/BCR/校准）统一走 shared/report_artifact
from report_artifact import (
    render_ks_curve,
    render_roc_curve,
    render_lift_chart,
    render_lift_chart_optbinning,
    render_bad_capture_rate,
)
from xgb_cli import build_parser, parse_args
from result_emitter import ResultEmitter

# Task 0-5 基础设施层
from safety_gate import SafetyGate

def _safe_chart(name: str, renderer, lines: list):
    """安全渲染单张图表，失败时追加警告而不中断报告生成。"""
    try:
        lines.append(renderer())
    except Exception as e:
        logger.warning(f"图表 {name} 渲染失败: {e}")
        lines.append(f"> ⚠️ 图表 **{name}** 渲染失败: {e}")
    lines.append("")
from diagnostic.report import run_diagnostics
from model_card import ModelCard, DataFingerprint, FeatureSpec, CutPoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 特征集合准备
# ============================================================

def prepare_feature_sets(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    train_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    features_arg: Optional[str],
    feature_sets_arg: Optional[str],
    feature_scheme: Optional[str],
    auto_select: bool,
    baseline_filter: Optional[str],
    comparison_filter: Optional[str],
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]]]:
    """准备特征集合，返回 (特征方案, 分析结果摘要, 方案详情)"""
    # 优先级 1: 指定单套特征
    if features_arg:
        feat_list = [f.strip() for f in features_arg.split(',') if f.strip()]
        logger.info(f"使用指定特征: {len(feat_list)} 个")
        return {'指定特征': feat_list}, None, None

    # 优先级 2: JSON 多方案
    if feature_sets_arg:
        if feature_sets_arg.endswith('.json'):
            with open(feature_sets_arg, 'r', encoding='utf-8') as f:
                feature_sets = json.load(f)
        else:
            feature_sets = json.loads(feature_sets_arg)
        logger.info(f"使用 JSON 多方案: {list(feature_sets.keys())}")
        return feature_sets, None, None

    # 优先级 3: 指定单套自动筛选方案（--feature_scheme full/decorr/high_iv/stable）
    if feature_scheme:
        logger.info(f"执行指定方案自动筛选: {feature_scheme}")
        selector = FeatureSelector()
        selector.analyze(
            data=data,
            features=features,
            target=target,
            baseline_data=train_data,
            comparison_data=oot_data,
        )
        scheme_method = {
            'full': selector.get_full_scheme,
            'decorr': selector.get_decorr_scheme,
            'high_iv': selector.get_high_iv_scheme,
            'stable': selector.get_stable_scheme,
        }.get(feature_scheme)
        if not scheme_method:
            raise ValueError(f"未知的 feature_scheme: {feature_scheme!r}，可选: full/decorr/high_iv/stable")
        feat_list, info = scheme_method()
        display_name = info['strategy']
        feature_sets = {display_name: feat_list}
        feature_sets_info = {display_name: info}
        logger.info(f"  {display_name}: {len(feat_list)} 个特征")
        analysis_summary = selector.get_analysis_summary()
        return feature_sets, analysis_summary, feature_sets_info

    # 优先级 4: 自动特征筛选（全量对比）
    if auto_select:
        logger.info("执行自动特征筛选...")
        selector = FeatureSelector()

        # 准备 PSI 数据
        baseline_data = train_data
        comparison_data = oot_data

        selector.analyze(
            data=data,
            features=features,
            target=target,
            baseline_data=baseline_data,
            comparison_data=comparison_data,
        )

        schemes = selector.get_all_schemes()
        feature_sets = {}
        feature_sets_info = {}
        for name, (feat_list, info) in schemes.items():
            if feat_list:
                display_name = info['strategy']
                feature_sets[display_name] = feat_list
                feature_sets_info[display_name] = info
                logger.info(f"  {display_name}: {len(feat_list)} 个特征")

        analysis_summary = selector.get_analysis_summary()
        return feature_sets, analysis_summary, feature_sets_info

    # 默认: 全部特征
    logger.info(f"使用全部特征: {len(features)} 个")
    return {'全部特征': features}, None, None

# ============================================================
# 模型训练与评估
# ============================================================

def train_and_evaluate(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    oot_data: pd.DataFrame,
    target: str,
    feature_sets: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """对多个特征方案训练并评估（train/val/oot 三段式，val 用于 early stopping，oot 仅用于汇报）"""
    all_results = {}

    for name, features in feature_sets.items():
        if not features:
            continue

        logger.info(f"训练方案: {name} ({len(features)} 特征)")

        X_train = train_data[features]
        y_train = train_data[target]
        X_val = val_data[features]
        y_val = val_data[target]
        X_oot = oot_data[features]
        y_oot = oot_data[target]

        # 防止 LLM 误把 oot 喂进 eval_set 导致信息泄漏
        eval_set = build_xgb_eval_set(
            train=train_data, val=val_data,
            features=features, target=target,
        )

        trainer = XGBTrainer()
        # eval_set[1] = (X_val, y_val)，仅用 val 做 early stopping
        trainer.train(X_train, y_train, eval_set[1][0], eval_set[1][1])

        prob_train = trainer.predict_proba(X_train)
        prob_val = trainer.predict_proba(X_val)
        prob_oot = trainer.predict_proba(X_oot)

        eval_train = ModelEvaluator.evaluate(y_train, prob_train)
        eval_val = ModelEvaluator.evaluate(y_val, prob_val)
        eval_oot = ModelEvaluator.evaluate(y_oot, prob_oot)

        # 模型分数 IV 最优分箱（OOT）
        score_iv_table = ModelEvaluator.calculate_score_iv_table(y_oot, prob_oot)

        all_results[name] = {
            'features': features,
            'train': eval_train,
            'val': eval_val,
            'oot': eval_oot,
            'importance': trainer.get_feature_importance(),
            'trainer': trainer,
            'prob_oot': prob_oot,
            'score_iv_table': score_iv_table,
        }

        logger.info(f"  Train AUC={eval_train['auc']:.4f}, Val AUC={eval_val['auc']:.4f}, OOT AUC={eval_oot['auc']:.4f} | Train KS={eval_train['ks']:.4f}, Val KS={eval_val['ks']:.4f}, OOT KS={eval_oot['ks']:.4f}")

    return all_results

def select_best_scheme(results: Dict[str, Dict[str, Any]], max_gap: float = 0.08) -> str:
    """选择最优方案：OOT KS 最高且 KS Gap 可接受（KS 优先）"""
    candidates = []
    for name, res in results.items():
        oot_ks = res['oot']['ks']
        ks_gap = res['train']['ks'] - res['oot']['ks']
        candidates.append((name, oot_ks, ks_gap))

    # 先筛选 KS Gap 可接受的，再按 OOT KS 排序
    acceptable = [(n, k, g) for n, k, g in candidates if g < max_gap]
    if acceptable:
        acceptable.sort(key=lambda x: x[1], reverse=True)
        return acceptable[0][0]

    # 如果都不可接受，取 KS Gap 最小的
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0]

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
    all_results: Dict[str, Dict[str, Any]],
    best_name: str,
    stability_report: Optional[Dict[str, Any]],
    analysis_summary: Optional[Dict[str, Any]] = None,
    feature_sets_info: Optional[Dict[str, Dict[str, Any]]] = None,
    split_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """生成 Markdown 报告"""
    lines = []

    # 报告头部
    lines.append("# XGBoost 建模报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 数据文件：{data_path}")
    lines.append(f"> 目标变量：{target}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 执行摘要（放在最前，方便业务方快速扫描）
    best_res_summary = all_results[best_name]
    summary_oot = best_res_summary['oot']
    summary_gap = best_res_summary['train']['auc'] - summary_oot['auc']
    summary_psi_str = f"{stability_report['overall_score_psi']:.4f}" if stability_report else "未计算"
    summary_psi_status = ""
    if stability_report:
        psi_val = stability_report['overall_score_psi']
        summary_psi_status = "稳定" if psi_val < 0.1 else ("轻微漂移" if psi_val < 0.25 else "⚠️ 显著漂移")

    lines.append("## 执行摘要")
    lines.append("")
    lines.append(f"- **最优方案**：「{best_name}」（{len(best_res_summary['features'])} 个特征），"
                 f"OOT AUC={summary_oot['auc']:.4f}，OOT KS={summary_oot['ks']:.4f}")
    lines.append(f"- **泛化性**：Train-OOT Gap={summary_gap:.4f}，"
                 f"{'无过拟合风险' if summary_gap < 0.05 else '⚠️ 存在过拟合风险'}")
    if stability_report:
        lines.append(f"- **稳定性**：分数 PSI={summary_psi_str}，{summary_psi_status}")
    # 找出最需要关注的特征
    top_feat = list(best_res_summary['importance'].keys())[0] if best_res_summary.get('importance') else ""
    if top_feat:
        top_imp = best_res_summary['importance'][top_feat]
        lines.append(f"- **关注点**：Top 特征 `{top_feat}` 贡献 {top_imp:.1%} 重要性，需关注其数据源稳定性")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 数据概览
    lines.append("## 1. 数据概览")
    lines.append("")
    # 插入统一切分说明（train/val/oot 三段式 + 切分策略 + OOT 边界）
    if split_meta:
        split_md = render_split_section(split_meta)
        if split_md:
            lines.append(split_md)
            lines.append("")
    total_samples = len(train_data) + len(val_data) + len(oot_data)
    total_pos = int(train_data[target].sum()) + int(val_data[target].sum()) + int(oot_data[target].sum())
    total_neg = total_samples - total_pos
    pos_neg_ratio = total_pos / max(total_neg, 1)
    lines.append(f"- **总样本量**: {total_samples:,}，正负样本比 1:{1/pos_neg_ratio:.1f}")
    lines.append("")
    # 时间范围
    if time_col and time_col in train_data.columns:
        try:
            train_range = f"{train_data[time_col].min()} ~ {train_data[time_col].max()}"
            val_range = f"{val_data[time_col].min()} ~ {val_data[time_col].max()}"
            oot_range = f"{oot_data[time_col].min()} ~ {oot_data[time_col].max()}"
            lines.append(f"- **Train 时间范围**: {train_range}")
            lines.append(f"- **Val 时间范围**: {val_range}")
            lines.append(f"- **OOT 时间范围**: {oot_range}")
            lines.append("")
        except Exception:
            pass
    lines.append("| 数据集 | 样本量 | 正样本量 | 正样本率 |")
    lines.append("|--------|--------|----------|----------|")
    for name, df in [('Train', train_data), ('Val', val_data), ('OOT', oot_data)]:
        n = len(df)
        pos = int(df[target].sum())
        rate = df[target].mean()
        lines.append(f"| {name} | {n:,} | {pos:,} | {rate:.4%} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. 特征方案对比
    lines.append("## 2. 特征方案对比")
    lines.append("")
    lines.append("### 2.1 方案效果对比")
    lines.append("")
    lines.append("| 方案 | 特征数 | Train AUC | Val AUC | OOT AUC | Train KS | Val KS | OOT KS | AUC Gap | KS Gap |")
    lines.append("|------|--------|-----------|---------|---------|----------|--------|--------|---------|--------|")
    for name, res in all_results.items():
        n_feat = len(res['features'])
        tr_auc = res['train']['auc']
        va_auc = res['val']['auc']
        oot_auc = res['oot']['auc']
        tr_ks = res['train']['ks']
        va_ks = res['val']['ks']
        oot_ks = res['oot']['ks']
        auc_gap = tr_auc - oot_auc
        ks_gap = tr_ks - oot_ks
        marker = " ✓" if name == best_name else ""
        lines.append(f"| {name}{marker} | {n_feat} | {tr_auc:.4f} | {va_auc:.4f} | {oot_auc:.4f} | {tr_ks:.4f} | {va_ks:.4f} | {oot_ks:.4f} | {auc_gap:.4f} | {ks_gap:.4f} |")
    lines.append("")
    lines.append(f"**最优方案**: {best_name}")
    lines.append("")

    # 多方案对比图表（多系列）
    if len(all_results) > 1:
        scheme_names = list(all_results.keys())
        # AUC 对比图
        auc_series = []
        for metric_key, metric_label in [('train', 'Train AUC'), ('val', 'Val AUC'), ('oot', 'OOT AUC')]:
            auc_series.append({
                'name': metric_label,
                'values': [round(all_results[n][metric_key]['auc'], 4) for n in scheme_names]
            })
        multi_chart_json = json.dumps({'labels': scheme_names, 'series': auc_series}, ensure_ascii=False)
        lines.append(f'<!--ARTIFACT_START--><artifact type="chart" chart-type="bar" title="方案 AUC 对比">')
        lines.append(multi_chart_json)
        lines.append('</artifact><!--ARTIFACT_END-->')
        lines.append("")
        # KS 对比图
        ks_series = []
        for metric_key, metric_label in [('train', 'Train KS'), ('val', 'Val KS'), ('oot', 'OOT KS')]:
            ks_series.append({
                'name': metric_label,
                'values': [round(all_results[n][metric_key]['ks'], 4) for n in scheme_names]
            })
        ks_chart_json = json.dumps({'labels': scheme_names, 'series': ks_series}, ensure_ascii=False)
        lines.append(f'<!--ARTIFACT_START--><artifact type="chart" chart-type="bar" title="方案 KS 对比">')
        lines.append(ks_chart_json)
        lines.append('</artifact><!--ARTIFACT_END-->')
        lines.append("")
    # 方案文字分析
    if len(all_results) > 1:
        lines.append("**方案分析**:")
        lines.append("")
        best_oot_auc = all_results[best_name]['oot']['auc']
        best_oot_ks = all_results[best_name]['oot']['ks']
        for name, res in all_results.items():
            n_feat = len(res['features'])
            oot_auc = res['oot']['auc']
            oot_ks = res['oot']['ks']
            auc_gap = res['train']['auc'] - res['oot']['auc']
            ks_gap = res['train']['ks'] - res['oot']['ks']
            auc_diff = oot_auc - best_oot_auc
            ks_diff = oot_ks - best_oot_ks
            parts = [f"{n_feat} 个特征"]
            if auc_gap >= 0.05:
                parts.append("存在过拟合风险（AUC Gap={:.4f}）".format(auc_gap))
            elif auc_gap < 0.03:
                parts.append("泛化性好（AUC Gap={:.4f}）".format(auc_gap))
            if ks_gap >= 0.10:
                parts.append("KS 过拟合风险（KS Gap={:.4f}）".format(ks_gap))
            elif ks_gap < 0.05:
                parts.append("KS 泛化性好（KS Gap={:.4f}）".format(ks_gap))
            if name == best_name:
                parts.append(f"综合表现最优（OOT AUC={oot_auc:.4f}, OOT KS={oot_ks:.4f}）")
            else:
                close_parts = []
                if abs(auc_diff) < 0.005:
                    close_parts.append("AUC 差异仅 {:.4f}".format(abs(auc_diff)))
                if abs(ks_diff) < 0.005:
                    close_parts.append("KS 差异仅 {:.4f}".format(abs(ks_diff)))
                if close_parts:
                    parts.append("与最优方案" + "，".join(close_parts) + "，可作为备选")
            lines.append(f"- **{name}**: {'，'.join(parts)}")
        lines.append("")

    # 2.2 方案特征明细
    if analysis_summary:
        lines.append("### 2.2 方案特征明细")
        lines.append("")
        # PSI 基线标注
        lines.append("> **PSI 基线**：Train 数据集")
        lines.append("")
        
        iv_results = analysis_summary.get('iv_results', {})
        psi_results = analysis_summary.get('psi_results', {})
        stats_results = analysis_summary.get('stats_results', {})

        for scheme_name, res in all_results.items():
            feat_list = res['features']
            importance = res.get('importance', {})
            
            lines.append(f"#### {scheme_name}（{len(feat_list)} 个特征）")
            lines.append("")
            lines.append("| 特征名 | XGB重要性 | IV | IV等级 | PSI | PSI等级 | 缺失率 | 综合评价 |")
            lines.append("|--------|-----------|-----|--------|-----|---------|--------|----------|")
            
            # 按 XGB 重要性降序排列
            feat_with_imp = [(f, importance.get(f, 0)) for f in feat_list]
            feat_with_imp.sort(key=lambda x: x[1], reverse=True)
            
            for feat, imp in feat_with_imp:
                iv_val = iv_results.get(feat, 0)
                psi_val = psi_results.get(feat, 0)
                miss_rate = stats_results.get(feat, {}).get('miss_rate', 0)
                
                # IV 等级判定
                if iv_val >= 0.3:
                    iv_level = "极强"
                elif iv_val >= 0.1:
                    iv_level = "强"
                elif iv_val >= 0.02:
                    iv_level = "中等"
                else:
                    iv_level = "弱"
                
                # PSI 等级判定
                if psi_val >= 0.25:
                    psi_level = "显著漂移"
                elif psi_val >= 0.1:
                    psi_level = "轻微漂移"
                else:
                    psi_level = "稳定"
                
                # 综合评价
                if iv_val >= 0.1 and psi_val < 0.1:
                    eval_label = "高预测+稳定"
                elif iv_val >= 0.02 and psi_val < 0.1:
                    eval_label = "中预测+稳定"
                elif iv_val >= 0.1 and psi_val >= 0.1:
                    eval_label = "高预测+需关注"
                else:
                    eval_label = "-"
                
                lines.append(f"| {feat} | {imp:.4f} | {iv_val:.4f} | {iv_level} | {psi_val:.4f} | {psi_level} | {miss_rate:.2%} | {eval_label} |")
            lines.append("")
        
        # IV/PSI 等级说明
        lines.append("> **IV等级**：<0.02 弱, 0.02-0.1 中等, 0.1-0.3 强, ≥0.3 极强")
        lines.append("> **PSI等级**：<0.1 稳定, 0.1-0.25 轻微漂移, ≥0.25 显著漂移")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 3. 最优方案详情
    best_res = all_results[best_name]
    lines.append("## 3. 最优方案详情")
    lines.append("")
    lines.append("### 3.1 评估指标")
    lines.append("")
    lines.append("| 指标 | Train | Val | OOT |")
    lines.append("|------|-------|-----|-----|")
    for metric in ['auc', 'ks', 'gini']:
        tr = best_res['train'][metric]
        va = best_res['val'][metric]
        oot = best_res['oot'][metric]
        lines.append(f"| {metric.upper()} | {tr:.4f} | {va:.4f} | {oot:.4f} |")
    lines.append("")

    # 模型分数 IV 最优分箱表
    score_iv_table = best_res.get('score_iv_table')
    if score_iv_table is not None and not score_iv_table.empty:
        lines.append("### 3.2 模型分数 IV 最优分箱（OOT）")
        lines.append("")
        lines.append("> 基于 OptimalBinning 对模型预测概率进行最优分箱，评估分数区分能力")
        lines.append("")
        lines.append(score_iv_table.to_markdown(index=False))
        lines.append("")

        # 解读
        total_iv = score_iv_table[score_iv_table['区间'] == 'Total']['IV'].values
        if len(total_iv) > 0:
            iv_val = float(total_iv[0])
            if iv_val >= 0.5:
                iv_level = "很强"
            elif iv_val >= 0.3:
                iv_level = "较强"
            elif iv_val >= 0.1:
                iv_level = "中等"
            elif iv_val >= 0.02:
                iv_level = "较差"
            else:
                iv_level = "无预测能力"
            lines.append(f"> **模型分数 IV = {iv_val:.4f}**，预测能力 **{iv_level}**")
        lines.append("")
    else:
        lines.append("### 3.2 模型分数 IV 最优分箱（OOT）")
        lines.append("")
        lines.append("> 未计算（optbinning 未安装或计算失败）")
        lines.append("")

    # ROC 曲线（OOT）
    _safe_chart("ROC 曲线", lambda: render_roc_curve(oot_data[target], best_res['prob_oot'], prefix="OOT "), lines)

    # KS 曲线（OOT）
    _safe_chart("KS 曲线", lambda: render_ks_curve(oot_data[target], best_res['prob_oot'], prefix="OOT "), lines)

    # Lift 表
    lines.append("### 3.3 Lift 表（OOT）")
    lines.append("")
    lift_table = ModelEvaluator.calculate_lift_table(oot_data[target], best_res['prob_oot'])
    
    # 计算累计列
    total_pos = lift_table['positive'].sum()
    lift_table['cum_positive'] = lift_table['positive'].cumsum()
    lift_table['cum_positive_rate'] = lift_table['cum_positive'] / total_pos
    lift_table['cum_count'] = lift_table['count'].cumsum()
    lift_table['cum_lift'] = (lift_table['cum_positive'] / lift_table['cum_count']) / (total_pos / lift_table['count'].sum())
    
    lines.append("| 分箱 | 样本数 | 正样本数 | 正样本率 | 累计正样本率 | Lift | 累计Lift |")
    lines.append("|------|--------|----------|----------|--------------|------|----------|")
    for idx, row in lift_table.iterrows():
        lines.append(f"| {idx} | {int(row['count'])} | {int(row['positive'])} | {row['positive_rate']:.4%} | {row['cum_positive_rate']:.2%} | {row['lift']:.2f} | {row['cum_lift']:.2f} |")
    lines.append("")

    # Lift Chart Artifact（等频分箱）
    _safe_chart("Lift Chart", lambda: render_lift_chart(oot_data[target], best_res['prob_oot'], prefix="OOT "), lines)

    # Lift Chart Artifact（IV 最优分箱）
    _safe_chart("Lift Chart（IV最优分箱）", lambda: render_lift_chart_optbinning(oot_data[target], best_res['prob_oot'], prefix="OOT "), lines)

    # Lift 解读
    top1_lift = lift_table.iloc[0]['lift']
    top1_pos_rate = lift_table.iloc[0]['positive_rate']
    overall_pos_rate = total_pos / total_samples if total_samples > 0 else 0
    top2_cum_lift = lift_table.iloc[1]['cum_lift'] if len(lift_table) > 1 else top1_lift
    lines.append(f"**Lift 解读**:")
    lines.append("")
    lines.append(f"- Top 10% 人群中正样本率为 {top1_pos_rate:.2%}，是整体的 **{top1_lift:.1f} 倍**")
    lines.append(f"- Top 20% 人群累计捕获效率为随机的 **{top2_cum_lift:.1f} 倍**")
    if top1_lift >= 3:
        lines.append("- 模型区分度良好，头部人群高度富集")
    elif top1_lift >= 2:
        lines.append("- 模型具备一定区分能力，可用于粗筛")
    else:
        lines.append("- 模型区分度较弱，建议优化特征或调参")
    lines.append("")

    # 特征重要性
    lines.append("### 3.4 特征重要性 Top 10")
    lines.append("")
    importance = best_res['importance']
    if analysis_summary:
        iv_results_local = analysis_summary.get('iv_results', {})
        psi_results_local = analysis_summary.get('psi_results', {})
        stats_results_local = analysis_summary.get('stats_results', {})
        lines.append("| 排名 | 特征名 | XGB重要性 | IV | PSI | 缺失率 |")
        lines.append("|------|--------|-----------|-----|-----|--------|")
        for rank, (feat, imp) in enumerate(list(importance.items())[:10], 1):
            iv_val = iv_results_local.get(feat, 0)
            psi_val = psi_results_local.get(feat, 0)
            miss_rate = stats_results_local.get(feat, {}).get('miss_rate', 0)
            lines.append(f"| {rank} | {feat} | {imp:.4f} | {iv_val:.4f} | {psi_val:.4f} | {miss_rate:.2%} |")
    else:
        lines.append("| 排名 | 特征名 | 重要性 |")
        lines.append("|------|--------|--------|")
        for rank, (feat, imp) in enumerate(list(importance.items())[:10], 1):
            lines.append(f"| {rank} | {feat} | {imp:.4f} |")
    lines.append("")
    # 特征集中度分析
    imp_values = list(importance.values())
    if imp_values:
        total_imp = sum(imp_values)
        if total_imp > 0:
            top3_imp = sum(imp_values[:3]) / total_imp
            top5_imp = sum(imp_values[:min(5, len(imp_values))]) / total_imp
            lines.append(f"**特征集中度**: Top 3 特征贡献 {top3_imp:.1%} 重要性，Top 5 贡献 {top5_imp:.1%}")
            lines.append("")
            if top3_imp > 0.7:
                lines.append("> 特征集中度较高，模型对少数特征依赖严重，建议关注这些特征的数据源稳定性")
            elif top3_imp > 0.5:
                lines.append("> 特征集中度适中，前几个特征贡献主要预测力")
            else:
                lines.append("> 特征贡献较为均衡，模型鲁棒性较好")
            lines.append("")
    # 3.4 坏客户捕获率（Bad Capture Rate @ K%）
    lines.append("### 3.5 坏客户捕获率（OOT）")
    lines.append("")
    lines.append("> 按得分从高到低排序，拒绝前 K% 人群能捕获多少比例的坏客户")
    lines.append("")
    bcr_data = ModelEvaluator.calculate_bad_capture_rate(oot_data[target], best_res['prob_oot'])
    lines.append("| 拒绝比例 | 坏客户捕获率 | 说明 |")
    lines.append("|---------|-------------|------|")
    for pct, rate in zip(bcr_data['topk_pcts'], bcr_data['bad_capture_rates']):
        comment = ""
        if rate >= 0.5:
            comment = "捕获过半坏客户"
        elif rate >= 0.3:
            comment = "良好"
        lines.append(f"| 拒绝 {pct}% | {rate:.2%} | {comment} |")
    lines.append("")

    # BCR 折线图（统一走 shared/report_artifact）
    _safe_chart("Bad Capture Rate", lambda: render_bad_capture_rate(oot_data[target], best_res['prob_oot'], prefix="OOT "), lines)

    lines.append("---")
    lines.append("")

    # 4. 稳定性分析
    lines.append("## 4. 稳定性分析")
    lines.append("")
    if stability_report:
        perf = stability_report['performance_by_month']
        lines.append("### 4.1 按月 AUC/KS")
        lines.append("")

        baseline_auc = perf.head(3)['auc'].mean() if len(perf) >= 3 else perf['auc'].mean()

        # 判断 CI 列是否有效（样本足够时才计算）
        has_ci = 'auc_ci_lower' in perf.columns and perf['auc_ci_lower'].notna().any()

        if has_ci:
            lines.append("| 月份 | 样本数 | 正样本率 | AUC | 95% CI | KS | 较基线变化 | 状态 |")
            lines.append("|------|--------|----------|-----|--------|-----|------------|------|")
        else:
            lines.append("| 月份 | 样本数 | 正样本率 | AUC | KS | 较基线变化 | 状态 |")
            lines.append("|------|--------|----------|-----|-----|------------|------|")

        for idx, row in perf.iterrows():
            auc_diff = row['auc'] - baseline_auc
            status = "✅" if abs(auc_diff) < 0.02 else ("⚠️" if auc_diff < -0.03 else "✅")
            change_str = "基线" if idx < 3 else f"{auc_diff:+.4f}"

            if has_ci and not (row.get('auc_ci_lower') != row.get('auc_ci_lower')):  # nan check
                ci_str = f"[{row['auc_ci_lower']:.4f}, {row['auc_ci_upper']:.4f}]"
                lines.append(f"| {row['month']} | {row['samples']} | {row['positive_rate']:.4%} | {row['auc']:.4f} | {ci_str} | {row['ks']:.4f} | {change_str} | {status} |")
            else:
                lines.append(f"| {row['month']} | {row['samples']} | {row['positive_rate']:.4%} | {row['auc']:.4f} | {row['ks']:.4f} | {change_str} | {status} |")

        lines.append("")
        lines.append(f"> AUC 标准差: {stability_report['auc_std']:.4f}，KS 标准差: {stability_report['ks_std']:.4f}")
        lines.append("")

        # Mann-Kendall 趋势检验结果
        auc_trend = stability_report.get('auc_trend', {})
        ks_trend = stability_report.get('ks_trend', {})
        lines.append("**趋势检验（Mann-Kendall）**:")
        lines.append("")
        trend_map = {
            "increasing": "↑ 显著上升（p={:.3f}，tau={:.3f}）",
            "decreasing": "↓ **显著下降**（p={:.3f}，tau={:.3f}）⚠️ 建议关注",
            "no trend": "→ 无显著趋势（p={:.3f}，tau={:.3f}）",
            "insufficient data": "数据不足（至少需要 4 个月）",
            "unknown": "未计算",
        }
        auc_trend_str = trend_map.get(auc_trend.get('trend', 'unknown'), '未知')
        if '{' in auc_trend_str:
            auc_trend_str = auc_trend_str.format(auc_trend.get('p_value', 0), auc_trend.get('tau', 0))
        lines.append(f"- AUC 趋势：{auc_trend_str}")

        ks_trend_str = trend_map.get(ks_trend.get('trend', 'unknown'), '未知')
        if '{' in ks_trend_str:
            ks_trend_str = ks_trend_str.format(ks_trend.get('p_value', 0), ks_trend.get('tau', 0))
        lines.append(f"- KS 趋势：{ks_trend_str}")
        lines.append("")

        # 月度 AUC 趋势图（带 CI）—— echarts artifact
        months_list = perf['month'].tolist()
        auc_list = [round(v, 4) for v in perf['auc'].tolist()]
        auc_chart_series = [{"name": "AUC", "type": "line", "data": auc_list, "smooth": True}]

        if has_ci:
            ci_lower = perf['auc_ci_lower'].fillna(method='ffill').tolist()
            ci_upper = perf['auc_ci_upper'].fillna(method='ffill').tolist()
            auc_chart_series.append({
                "name": "95% CI 上界",
                "type": "line",
                "data": [round(v, 4) for v in ci_upper],
                "lineStyle": {"opacity": 0},
                "areaStyle": {"opacity": 0.15, "color": "#5b9bd5"},
                "symbol": "none",
            })
            auc_chart_series.append({
                "name": "95% CI 下界",
                "type": "line",
                "data": [round(v, 4) for v in ci_lower],
                "lineStyle": {"opacity": 0},
                "areaStyle": {"opacity": 0},
                "symbol": "none",
            })

        auc_chart_option = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["AUC"] + (["95% CI 上界", "95% CI 下界"] if has_ci else [])},
            "xAxis": {"type": "category", "data": months_list},
            "yAxis": {"type": "value", "name": "AUC"},
            "series": auc_chart_series,
        }
        auc_chart_json = json.dumps(auc_chart_option, ensure_ascii=False)
        lines.append(f'<!--ARTIFACT_START--><artifact type="echarts" title="月度 AUC 趋势（含 95% CI）">')
        lines.append(auc_chart_json)
        lines.append('</artifact><!--ARTIFACT_END-->')
        lines.append("")

        psi_monthly = stability_report['score_psi_by_month']
        lines.append("### 4.2 分数 PSI")
        lines.append("")
        # PSI 基线标注
        train_month_str = ', '.join(stability_report.get('baseline_months', ['Train前3月'])[:3])
        lines.append(f"> **PSI 基线**：{train_month_str}")
        lines.append("")
        lines.append("| 月份 | 样本数 | 平均分数 | PSI | 漂移等级 |")
        lines.append("|------|--------|----------|-----|----------|")
        for _, row in psi_monthly.iterrows():
            psi_val = row['psi']
            if psi_val >= 0.25:
                drift_level = "⚠️ 显著漂移"
            elif psi_val >= 0.1:
                drift_level = "⚠️ 轻微漂移"
            else:
                drift_level = "✅ 稳定"
            lines.append(f"| {row['month']} | {row['samples']} | {row['mean_score']:.4f} | {row['psi']:.4f} | {drift_level} |")
        lines.append("")
        lines.append(f"> 整体分数 PSI: {stability_report['overall_score_psi']:.4f}")
        lines.append("")
        
        # PSI 解读
        overall_psi = stability_report['overall_score_psi']
        if overall_psi < 0.1:
            lines.append("**PSI 解读**: 整体 PSI < 0.1，分数分布稳定")
        elif overall_psi < 0.25:
            lines.append("**PSI 解读**: 整体 PSI 处于临界范围，建议持续监控")
        else:
            lines.append("**PSI 解读**: 整体 PSI > 0.25，分数分布存在显著漂移，建议重新评估模型")
    else:
        lines.append("> 未执行稳定性分析")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. 超参调优
    lines.append("## 5. 超参调优")
    lines.append("")
    lines.append("> 本报告为基线模型，若需进行参数调优，请使用 **xgb-tuning** 技能。")
    lines.append("")
    lines.append("调参建议：")
    lines.append("")
    best_gap = best_res['train']['auc'] - best_res['oot']['auc']
    if best_gap >= 0.05:
        lines.append("- 当前模型存在**过拟合风险** (Gap={:.4f})，建议调优方向：增大正则化、减小树深度".format(best_gap))
    elif best_gap >= 0.03:
        lines.append("- 当前模型 Gap={:.4f}，可进行精细调优".format(best_gap))
    else:
        lines.append("- 当前模型 Gap={:.4f}，泛化性良好，可采用默认参数".format(best_gap))
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. 最终模型总结
    lines.append("## 6. 最终模型总结")
    lines.append("")

    final_features = best_res['features']

    # 6.1 方案选择分析
    lines.append("### 6.1 方案选择分析")
    lines.append("")
    lines.append(f"**选中方案**：{best_name}")
    lines.append("")

    # 特征维度分析
    lines.append("**特征维度分析**：")
    best_n_feat = len(final_features)
    best_gap = best_res['train']['auc'] - best_res['oot']['auc']
    best_oot_auc = best_res['oot']['auc']

    # 找到次优方案对比
    other_schemes = [(n, r) for n, r in all_results.items() if n != best_name]
    if other_schemes:
        # 按 OOT AUC 排序，取最高的作为对比
        other_schemes.sort(key=lambda x: x[1]['oot']['auc'], reverse=True)
        compare_name, compare_res = other_schemes[0]
        compare_n_feat = len(compare_res['features'])
        compare_oot_auc = compare_res['oot']['auc']
        compare_gap = compare_res['train']['auc'] - compare_res['oot']['auc']
        auc_diff = best_oot_auc - compare_oot_auc
        gap_diff = compare_gap - best_gap

        if best_n_feat < compare_n_feat:
            lines.append(f"- 入选理由：{best_n_feat} 个特征 vs {compare_name} {compare_n_feat} 个，OOT AUC 差异 {auc_diff:+.4f}，Gap 缩小 {gap_diff:.4f}")
        else:
            lines.append(f"- 入选理由：OOT AUC {best_oot_auc:.4f} 为最优")

    lines.append(f"- 泛化性：Train-OOT Gap={best_gap:.4f}{'，无明显过拟合' if best_gap < 0.05 else '，存在一定过拟合风险'}")
    lines.append("")

    # 综合结论
    lines.append("**综合结论**：")
    conclusion_parts = [f"「{best_name}」方案"]
    if best_n_feat <= 5:
        conclusion_parts.append(f"以精简特征集（{best_n_feat} 个）")
    if best_gap < 0.05:
        conclusion_parts.append("泛化性良好")
    conclusion_parts.append(f"OOT AUC 达 {best_oot_auc:.4f}")
    lines.append(f"> {'，'.join(conclusion_parts)}，推荐作为基线模型。")
    lines.append("")

    lines.append("### 6.2 最终特征列表")
    lines.append("")
    lines.append(f"共 {len(final_features)} 个特征：")
    lines.append("")
    lines.append("```")
    lines.append(", ".join(final_features))
    lines.append("```")
    lines.append("")

    lines.append("### 6.3 最终表现")
    lines.append("")
    final_oot = best_res['oot']
    lines.append(f"- OOT AUC: **{final_oot['auc']:.4f}**")
    lines.append(f"- OOT KS: **{final_oot['ks']:.4f}**")
    if stability_report:
        lines.append(f"- 分数 PSI: **{stability_report['overall_score_psi']:.4f}**")
    lines.append("")
    
    # 6.4 风险提示
    lines.append("### 6.4 风险提示")
    lines.append("")
    lines.append("| 风险项 | 说明 | 建议 |")
    lines.append("|--------|------|------|")
    
    risk_items = []
    
    # 检查 Gap
    if best_gap >= 0.05:
        risk_items.append(("过拟合风险", f"Train-OOT Gap={best_gap:.4f} 偏大", "增大正则化或减少特征"))
    
    # 检查特征集中度
    if importance:
        top_imp = list(importance.values())[0] if importance else 0
        if top_imp > 0.7:
            top_feat = list(importance.keys())[0]
            risk_items.append(("特征集中度高", f"{top_feat} 贡献 {top_imp:.0%} 重要性", "关注该特征源稳定性"))
    
    # 检查稳定性
    if stability_report:
        if stability_report['overall_score_psi'] >= 0.1:
            risk_items.append(("分数漂移", f"整体 PSI={stability_report['overall_score_psi']:.4f}", "持续监控，必要时重训"))
        
        # 检查近期效果下降
        perf = stability_report['performance_by_month']
        if len(perf) >= 3:
            recent_auc = perf.tail(2)['auc'].mean()
            early_auc = perf.head(3)['auc'].mean()
            if recent_auc - early_auc < -0.02:
                risk_items.append(("近期效果下降", f"近期 AUC 较早期下降 {abs(recent_auc - early_auc):.4f}", "关注样本分布变化"))
    
    if risk_items:
        for risk, desc, suggestion in risk_items:
            lines.append(f"| {risk} | {desc} | {suggestion} |")
    else:
        lines.append("| 无明显风险 | 模型表现稳定 | 正常监控即可 |")
    lines.append("")
    
    # 6.5 部署建议
    lines.append("### 6.5 部署建议")
    lines.append("")
    lines.append(f"- **监控频率**: 周度监控 AUC/KS，月度监控 PSI")
    lines.append(f"- **重训触发**: AUC 下降 > 0.02 或 PSI > 0.15")
    lines.append(f"- **特征依赖**: 确保 {', '.join(final_features[:3])}{'...' if len(final_features) > 3 else ''} 数据源稳定")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 7. 模型总体评价
    lines.append("## 7. 模型总体评价")
    lines.append("")
    final_auc = best_res['oot']['auc']
    final_ks = best_res['oot']['ks']
    final_gap = best_res['train']['auc'] - final_auc
    eval_parts = []
    # AUC 评价
    if final_auc >= 0.75:
        eval_parts.append("模型区分度优秀（OOT AUC={:.4f}）".format(final_auc))
    elif final_auc >= 0.65:
        eval_parts.append("模型区分度良好（OOT AUC={:.4f}）".format(final_auc))
    elif final_auc >= 0.55:
        eval_parts.append("模型具备基础区分能力（OOT AUC={:.4f}），仍有优化空间".format(final_auc))
    else:
        eval_parts.append("模型区分度较弱（OOT AUC={:.4f}），建议重新审视特征工程".format(final_auc))
    # 泛化性评价
    if final_gap < 0.03:
        eval_parts.append("泛化性良好")
    elif final_gap < 0.05:
        eval_parts.append("泛化性可接受")
    else:
        eval_parts.append("存在过拟合风险，需持续关注")
    # 稳定性评价
    if stability_report:
        overall_psi = stability_report.get('overall_score_psi', 0)
        if overall_psi < 0.1:
            eval_parts.append("分数分布稳定")
        elif overall_psi < 0.25:
            eval_parts.append("分数分布存在轻微漂移")
        else:
            eval_parts.append("分数分布存在显著漂移，建议关注")
    lines.append(f"> {'；'.join(eval_parts)}。综合评估，{'建议作为基线模型上线部署' if final_auc >= 0.65 and final_gap < 0.05 else '建议进一步优化后再考虑部署'}。")
    lines.append("")

    return "\n".join(lines)

# ============================================================
# 主流程
# ============================================================

def main(args: List[str] | None = None) -> int:
    parser = build_parser("modeling", description="XGBoost 建模工具 (portable)")
    # 补充 portable 独有的 --output_dir
    parser.add_argument("--output_dir", default=None, help="产物输出目录；默认 ./outputs/<ts>")
    parsed = parse_args(parser, args)

    # 准备 output_dir
    if parsed.output_dir:
        output_dir = Path(parsed.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / "outputs" / ts).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="xgb-modeling", output_dir=output_dir)

    exclude_cols = [c.strip() for c in parsed.exclude_cols.split(',') if c.strip()]
    exclude_cols.append(parsed.time_col)

    val_filter_arg = parsed.val_filter

    # 报告输出名称
    if parsed.report_output:
        output_name = os.path.splitext(os.path.basename(parsed.report_output))[0]
    else:
        output_name = 'xgb_modeling_report'

    try:
        # 0. SafetyGate 建模前体检（需要先加载数据）
        # 1+2. 加载与切分数据（统一走 shared/data_utils.load_and_split_data）
        split_result = load_and_split_data(
            data_path=parsed.data_path,
            target=parsed.target,
            time_col=parsed.time_col,
            train_filter=parsed.train_filter,
            test_filter=val_filter_arg,  # 等价 val_filter
            oot_filter=parsed.oot_filter,
            oot_ratio=parsed.oot_ratio,
            val_ratio=parsed.val_ratio,
            random_seed=parsed.random_seed,
            return_features=True,
            exclude_cols=exclude_cols,
        )
        train_data = split_result.train
        val_data = split_result.val
        oot_data = split_result.oot
        features = split_result.features or []
        split_meta = split_result.meta
        if split_meta:
            logger.info(f"[split-meta] {split_meta}")

        # SafetyGate 建模前体检
        gate_report = SafetyGate.check(
            train_data, val_data, oot_data,
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

        # 供 prepare_feature_sets / 稳定性分析使用的全量数据（train+val+oot 拼接）
        data = pd.concat([train_data, val_data, oot_data], ignore_index=True)

        # 3. 准备特征集合
        feature_sets, analysis_summary, feature_sets_info = prepare_feature_sets(
            data=data,
            features=features,
            target=parsed.target,
            train_data=train_data,
            oot_data=oot_data,
            features_arg=parsed.features,
            feature_sets_arg=parsed.feature_sets,
            feature_scheme=parsed.feature_scheme,
            auto_select=parsed.auto_select,
            baseline_filter=parsed.baseline_filter,
            comparison_filter=parsed.comparison_filter,
        )

        # 4. 多方案训练评估
        all_results = train_and_evaluate(
            train_data, val_data, oot_data,
            parsed.target, feature_sets
        )

        # 5. 选择最优方案
        best_name = select_best_scheme(all_results)
        logger.info(f"最优方案: {best_name}")

        # 6. 稳定性分析
        best_trainer = all_results[best_name]['trainer']
        best_features = all_results[best_name]['features']

        scored_data = data.copy()
        scored_data['pred_score'] = best_trainer.predict_proba(data[best_features])

        train_months = sorted(train_data[parsed.time_col].astype(str).str[:6].unique().tolist())

        stability = ModelStabilityAnalyzer()
        stability_report = stability.full_stability_report(
            scored_data, parsed.time_col, parsed.target, 'pred_score', train_months
        )

        # 6.5 训练后诊断（DiagnosticReport）
        best_res = all_results[best_name]
        diag_report = run_diagnostics(
            train_metrics=best_res['train'],
            eval_metrics=best_res['val'],
            metric="ks",
            y_true_train=train_data[parsed.target].values,
            y_prob_train=best_trainer.predict_proba(train_data[best_features]),
            y_true_val=val_data[parsed.target].values,
            y_prob_val=best_trainer.predict_proba(val_data[best_features]),
            y_true_oot=oot_data[parsed.target].values,
            y_prob_oot=best_res['prob_oot'],
        )
        if diag_report.is_overfit:
            logger.warning(f"DiagnosticReport: 过拟合 severity={diag_report.severity}")
        for w in diag_report.all_warnings:
            logger.warning(f"DiagnosticReport: {w}")

        # 7. 生成报告
        report = generate_report(
            data_path=parsed.data_path,
            target=parsed.target,
            train_data=train_data,
            val_data=val_data,
            oot_data=oot_data,
            time_col=parsed.time_col,
            all_results=all_results,
            best_name=best_name,
            stability_report=stability_report,
            analysis_summary=analysis_summary,
            feature_sets_info=feature_sets_info,
            split_meta=split_meta,
        )

        # 报告落盘到 output_dir
        report_path = output_dir / f"{output_name}.md"
        report_path.write_text(report, encoding='utf-8')
        logger.info(f"✅ 报告已保存至: {report_path}")
        emitter.add_report(str(report_path))

        # 登记 project_state 更新
        best_res = all_results[best_name]
        emitter.update_state({
            "model_result": {
                "algorithm": "xgboost",
                "scheme_name": best_name,
                "features": best_res['features'],
                "n_features": len(best_res['features']),
                "metrics": {
                    "train": best_res['train'],
                    "val": best_res['val'],
                    "oot": best_res['oot'],
                },
                "gap": round(best_res['train']['ks'] - best_res['oot']['ks'], 4),
                "gap_auc": round(best_res['train']['auc'] - best_res['oot']['auc'], 4),
                "psi": stability_report.get('overall_score_psi') if stability_report else None,
                "safety_gate": gate_report.to_dict(),
                "diagnostic": diag_report.to_dict(),
            },
        })

        # 8. 保存模型（无条件，统一由 emitter 登记）
        if parsed.model_name:
            model_name = parsed.model_name
        else:
            model_name = 'xgb_model_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

        models_dir = output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = str(models_dir / '{}.json'.format(model_name))
        model_meta_path = str(models_dir / '{}_meta.json'.format(model_name))

        best_trainer.save_model(model_save_path)
        logger.info(f"✅ 模型已保存至: {model_save_path}")

        emitter.add_model(
            model_save_path,
            meta_path=model_meta_path,
            feature_count=len(best_features),
            scheme_name=best_name,
            model_type="xgboost",
        )

        # 9. 生成 ModelCard
        data_fp = DataFingerprint.from_dataframe(
            train_data, target_col=parsed.target,
            time_col=parsed.time_col, feature_cols=best_features,
        )
        feature_specs = [
            FeatureSpec(
                name=f,
                shap_importance=best_res['importance'].get(f),
            )
            for f in best_features
        ]
        model_card = ModelCard.create(
            model_id=model_name,
            params={},  # 基线默认参数
            eval_results={
                "train": best_res['train'],
                "val": best_res['val'],
                "oot": best_res['oot'],
            },
            data_fingerprint=data_fp,
            diagnostic=diag_report.to_dict(),
            features=feature_specs,
        )
        # 保存 ModelCard JSON
        card_path = str(models_dir / f"{model_name}_card.json")
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(model_card.to_json())
        logger.info(f"✅ ModelCard 已保存至: {card_path}")

        print(f"\n=== 模型保存成功 ===")
        print(f"模型文件: {model_save_path}")
        print(f"模型名称: {model_name}")
        print(f"特征数量: {len(best_features)}")
        print(f"特征列表: {', '.join(best_features)}")

        # 补充 metrics 和 summary
        emitter.update_metrics({
            "oot_auc": round(best_res['oot']['auc'], 4),
            "oot_ks": round(best_res['oot']['ks'], 4),
            "train_auc": round(best_res['train']['auc'], 4),
            "val_auc": round(best_res['val']['auc'], 4),
            "auc_gap": round(best_res['train']['auc'] - best_res['oot']['auc'], 4),
            "ks_gap": round(best_res['train']['ks'] - best_res['oot']['ks'], 4),
            "n_features": len(best_features),
            "overall_score_psi": stability_report.get('overall_score_psi') if stability_report else None,
        })
        emitter.set_summary(
            f"xgb-modeling 完成：最优方案「{best_name}」，"
            f"OOT AUC={best_res['oot']['auc']:.4f}，OOT KS={best_res['oot']['ks']:.4f}，"
            f"特征数={len(best_features)}，"
            f"Gap={best_res['train']['auc'] - best_res['oot']['auc']:.4f}"
        )
        result_path = emitter.write(output_dir / "result.json")
        print(f"result: {result_path}")
        return 0

    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        try:
            emitter.set_status("failed")
            emitter.set_summary(f"执行失败：{type(e).__name__}: {e}")
            fail_path = emitter.write(output_dir / "result.json")
            print(f"FAILED | {type(e).__name__}: {e}")
            print(f"result: {fail_path}")
        except Exception:
            pass
        return 1

if __name__ == '__main__':
    sys.exit(main())
