# -*- coding: utf-8 -*-

"""
特征分析主编排脚本
统一入口，支持命令行参数，调度各分析器并生成报告

Usage:
    python analyzer.py --data_path ./data.parquet --target y_label [options]

"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from fa_core.statistics import StatisticsCalculator  # noqa: E402
from fa_core.iv import IVCalculator  # noqa: E402
from fa_core.psi import PSICalculator  # noqa: E402
from fa_core.correlation import CorrelationCalculator  # noqa: E402
from data_utils import read_dataset, infer_features  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

# 统一日志配置：stderr，与 xgb-tuning / xgb-modeling 保持一致
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ============================================================
# 辅助函数
# ============================================================

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """将 DataFrame 转换为 Markdown 表格（不依赖 tabulate）"""
    cols = df.columns.tolist()
    lines = []
    # Header
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    # Rows
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)

def iv_interpret(iv: float) -> str:
    """IV 值解释"""
    if iv < 0.02:
        return "无预测能力"
    if iv < 0.1:
        return "较差"
    if iv < 0.3:
        return "中等"
    if iv < 0.5:
        return "较强"
    return "很强"

def psi_interpret(psi: float) -> str:
    """PSI 值解释"""
    if psi < 0.1:
        return "稳定"
    if psi < 0.25:
        return "轻微漂移"
    return "⚠️ 显著漂移"

def greedy_select(candidate_features: List[str], iv_results: Dict[str, float],
                  corr_matrix: pd.DataFrame, corr_threshold: float = 0.7) -> List[str]:
    """按 IV 从高到低贪心去共线性"""
    sorted_feats = sorted(candidate_features, key=lambda f: iv_results.get(f, 0), reverse=True)
    selected = []
    for feat in sorted_feats:
        conflict = any(
            abs(corr_matrix.loc[feat, sel]) >= corr_threshold
            for sel in selected
            if feat in corr_matrix.index and sel in corr_matrix.columns
        )
        if not conflict:
            selected.append(feat)
    return selected

def greedy_select_top_n(candidate_features: List[str], iv_results: Dict[str, float],
                        corr_matrix: pd.DataFrame, corr_threshold: float = 0.7, n: int = 10) -> List[str]:
    """贪心选 Top N，选满 N 个停止"""
    sorted_feats = sorted(candidate_features, key=lambda f: iv_results.get(f, 0), reverse=True)
    selected = []
    for feat in sorted_feats:
        if len(selected) >= n:
            break
        conflict = any(
            abs(corr_matrix.loc[feat, sel]) >= corr_threshold
            for sel in selected
            if feat in corr_matrix.index and sel in corr_matrix.columns
        )
        if not conflict:
            selected.append(feat)
    return selected

# ============================================================
# 数据加载
# ============================================================

def load_data(data_path: str, target: str, exclude_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """加载数据并确定特征列表（统一走 shared.data_utils）"""
    data = read_dataset(data_path, label_filter=None)  # feature-analysis 不强制过滤 0/1
    if target not in data.columns:
        raise ValueError(f"目标变量不存在: {target}")
    features = infer_features(data, target=target, exclude_cols=exclude_cols, numeric_only=False)

    print("数据加载完成：{} 行，{} 个特征".format(len(data), len(features)))
    print("目标变量正样本率：{:.4f}".format(data[target].mean()))

    return data, features

# ============================================================
# 报告生成
# ============================================================

def render_scheme_table(scheme_features: List[str], iv_results: Dict[str, float],
                        psi_results: Dict[str, float], stats_results: Dict[str, Dict]) -> str:
    """生成方案特征表"""
    rows = ["| 序号 | 特征名 | IV值 | PSI值 | 缺失率 |",
            "|------|--------|------|-------|--------|"]
    for i, f in enumerate(scheme_features, 1):
        iv = iv_results.get(f, 0)
        psi = psi_results.get(f, '-')
        mr = stats_results.get(f, {}).get('miss_rate', '-')
        psi_str = "{:.4f}".format(psi) if isinstance(psi, float) else str(psi)
        mr_str = "{:.2%}".format(mr) if isinstance(mr, float) else str(mr)
        rows.append("| {} | {} | {:.4f} | {} | {} |".format(i, f, iv, psi_str, mr_str))
    return "\n".join(rows)

def build_report(data: pd.DataFrame, features: List[str], target: str, data_path: str,
                 stats_results: Dict, iv_results: Dict, psi_results: Dict,
                 corr_matrix: pd.DataFrame, high_corr_pairs: List[Dict],
                 schemes: Dict[str, List[str]], iv_calc: IVCalculator,
                 top_n: Optional[int] = None, specified_features: Optional[List[str]] = None) -> str:
    """构建完整报告"""
    lines = []

    # 报告头部
    lines.append("# 建模特征分析报告")
    lines.append("")
    lines.append("> 生成时间：{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append("> 数据文件：{}".format(data_path))
    lines.append("> 样本量：{}".format(len(data)))
    lines.append("> 特征数：{}".format(len(features)))
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 数据概览
    lines.append("## 1. 数据概览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append("| 总样本量 | {} |".format(len(data)))
    lines.append("| 特征数量 | {} |".format(len(features)))
    lines.append("| 目标变量 | {} |".format(target))
    lines.append("| 正样本率 | {:.4f} |".format(data[target].mean()))
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. 基础统计分析
    lines.append("## 2. 基础统计分析")
    lines.append("")
    lines.append("### 2.1 统计指标汇总")
    lines.append("")
    lines.append("| 特征名 | 缺失率 | 均值 | 标准差 | 最小值 | 最大值 | 偏度 | 峰度 |")
    lines.append("|--------|--------|------|--------|--------|--------|------|------|")

    for f in features:
        s = stats_results.get(f, {})
        mr = s.get('miss_rate', '-')
        mr_str = "{:.2%}".format(mr) if isinstance(mr, float) else str(mr)
        mean_str = "{:.4f}".format(s.get('mean', 0)) if s.get('mean') is not None else '-'
        std_str = "{:.4f}".format(s.get('std', 0)) if s.get('std') is not None else '-'
        min_str = "{:.4f}".format(s.get('min', 0)) if s.get('min') is not None else '-'
        max_str = "{:.4f}".format(s.get('max', 0)) if s.get('max') is not None else '-'
        skew_str = "{:.4f}".format(s.get('skewness', 0)) if s.get('skewness') is not None else '-'
        kurt_str = "{:.4f}".format(s.get('kurtosis', 0)) if s.get('kurtosis') is not None else '-'
        # 异常标注
        flags = []
        skew_val = s.get('skewness')
        kurt_val = s.get('kurtosis')
        if skew_val is not None and abs(skew_val) > 3:
            flags.append("高度右偏" if skew_val > 0 else "高度左偏")
        if kurt_val is not None and kurt_val > 10:
            flags.append("厚尾")
        if isinstance(mr, float) and mr > 0.3:
            flags.append("高缺失")
        flag_str = " ⚠️" + ",".join(flags) if flags else ""
        lines.append("| {}{} | {} | {} | {} | {} | {} | {} | {} |".format(
            f, flag_str, mr_str, mean_str, std_str, min_str, max_str, skew_str, kurt_str))

    # 数据质量问题
    high_missing = [f for f in features if stats_results.get(f, {}).get('miss_rate', 0) > 0.3]
    high_mode = [f for f in features if stats_results.get(f, {}).get('mode_ratio', 0) > 0.9]
    # 异常特征汇总
    high_skew = [f for f in features if abs(stats_results.get(f, {}).get('skewness', 0) or 0) > 3]
    high_kurt = [f for f in features if (stats_results.get(f, {}).get('kurtosis', 0) or 0) > 10]

    lines.append("")
    lines.append("### 2.2 数据质量问题")
    lines.append("")
    lines.append("- **缺失率 > 30%** 的特征（{} 个）：{}".format(
        len(high_missing), ', '.join(high_missing) if high_missing else '无'))
    lines.append("- **众数占比 > 90%** 的特征（{} 个）：{}".format(
        len(high_mode), ', '.join(high_mode) if high_mode else '无'))
    if high_skew:
        lines.append("- **高度偏斜** 的特征（|skew|>3，{} 个）：{}".format(
            len(high_skew), ', '.join(high_skew)))
    if high_kurt:
        lines.append("- **厚尾分布** 的特征（kurtosis>10，{} 个）：{}".format(
            len(high_kurt), ', '.join(high_kurt)))
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. IV值分析
    lines.append("## 3. IV值分析（预测能力）")
    lines.append("")
    lines.append("### 3.1 IV值排名（Top 20）")
    lines.append("")
    lines.append("| 排名 | 特征名 | IV值 | 预测能力 |")
    lines.append("|------|--------|------|----------|")

    sorted_iv = sorted(iv_results.items(), key=lambda x: x[1], reverse=True)[:20]
    for rank, (f, iv) in enumerate(sorted_iv, 1):
        lines.append("| {} | {} | {:.4f} | {} |".format(rank, f, iv, iv_interpret(iv)))

    # IV Top 10 特征重要性图表
    iv_top10 = sorted(iv_results.items(), key=lambda x: x[1], reverse=True)[:10]
    iv_chart_data = json.dumps(
        [{"name": f, "value": round(v, 4)} for f, v in iv_top10],
        ensure_ascii=False
    )
    lines.append('')
    lines.append('<!--ARTIFACT_START--><artifact type="feature-importance" title="IV Top 10 特征排名">')
    lines.append(iv_chart_data)
    lines.append('</artifact><!--ARTIFACT_END-->')
    lines.append('')

    # IV 核心发现总结
    strong_iv = [f for f, v in iv_results.items() if v >= 0.1]
    medium_iv = [f for f, v in iv_results.items() if 0.02 <= v < 0.1]
    weak_iv = [f for f, v in iv_results.items() if v < 0.02]
    lines.append("")
    lines.append("**核心发现**: 共 {} 个特征具备中等以上预测力（IV≥0.1），{} 个弱预测力（IV 0.02~0.1），{} 个无预测能力（IV<0.02）".format(
        len(strong_iv), len(medium_iv), len(weak_iv)))
    if strong_iv:
        lines.append("")
        lines.append("预测力最强的特征集中在：{}".format(', '.join(strong_iv[:5])))

    # IV 分布统计
    iv_vals = list(iv_results.values())
    iv_buckets = [
        ("无预测能力 (< 0.02)", sum(1 for v in iv_vals if v < 0.02)),
        ("较差 (0.02-0.1)", sum(1 for v in iv_vals if 0.02 <= v < 0.1)),
        ("中等 (0.1-0.3)", sum(1 for v in iv_vals if 0.1 <= v < 0.3)),
        ("较强 (0.3-0.5)", sum(1 for v in iv_vals if 0.3 <= v < 0.5)),
        ("很强 (≥ 0.5)", sum(1 for v in iv_vals if v >= 0.5)),
    ]

    lines.append("")
    lines.append("### 3.2 IV分布统计")
    lines.append("")
    lines.append("| IV区间 | 特征数量 | 占比 |")
    lines.append("|--------|----------|------|")

    total_feats = len(iv_vals)
    for label, cnt in iv_buckets:
        lines.append("| {} | {} | {:.1%} |".format(label, cnt, cnt / total_feats if total_feats > 0 else 0))

    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. PSI稳定性分析
    lines.append("## 4. PSI稳定性分析")
    lines.append("")

    if psi_results:
        lines.append("### 4.1 PSI排名（稳定性由差到好）")
        lines.append("")
        lines.append("| 排名 | 特征名 | PSI值 | 稳定性评估 |")
        lines.append("|------|--------|-------|------------|")

        sorted_psi = sorted(psi_results.items(), key=lambda x: x[1], reverse=True)
        for rank, (f, psi) in enumerate(sorted_psi, 1):
            lines.append("| {} | {} | {:.4f} | {} |".format(rank, f, psi, psi_interpret(psi)))

        high_psi = [f for f, v in psi_results.items() if v >= 0.25]
        medium_psi = [f for f, v in psi_results.items() if 0.1 <= v < 0.25]
        lines.append("")
        lines.append("### 4.2 稳定性诊断")
        lines.append("")
        if high_psi:
            lines.append("**显著漂移特征（PSI ≥ 0.25）**：")
            lines.append("")
            for f in high_psi:
                lines.append("- **{}**（PSI={:.4f}）⚠️ 显著漂移，建议谨慎使用或排查数据源变化".format(f, psi_results[f]))
        if medium_psi:
            lines.append("")
            lines.append("**轻微漂移特征（PSI 0.1~0.25）**：")
            lines.append("")
            for f in medium_psi:
                lines.append("- {}  （PSI={:.4f}），需持续监控".format(f, psi_results[f]))
        if not high_psi and not medium_psi:
            lines.append("所有特征 PSI 均 < 0.1，稳定性良好。")

        # PSI 趋势图表
        sorted_psi_chart = sorted(psi_results.items(), key=lambda x: x[1], reverse=True)[:15]
        psi_chart_data = json.dumps({
            "labels": [p[0] for p in sorted_psi_chart],
            "values": [round(p[1], 4) for p in sorted_psi_chart]
        }, ensure_ascii=False)
        lines.append('')
        lines.append('<!--ARTIFACT_START--><artifact type="chart" chart-type="bar" title="PSI Top 15 特征稳定性排名">')
        lines.append(psi_chart_data)
        lines.append('</artifact><!--ARTIFACT_END-->')
        lines.append('')
    else:
        lines.append("> 未执行 PSI 分析（无时间维度数据或未提供切分条件）")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. 相关性分析
    lines.append("## 5. 相关性分析")
    lines.append("")
    lines.append("### 5.1 高相关特征对（|r| ≥ 0.7）")
    lines.append("")

    if high_corr_pairs:
        lines.append("| 特征1 | 特征2 | 相关系数 |")
        lines.append("|-------|-------|----------|")
        # 按相关系数降序排列
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        for pair in sorted_pairs:
            lines.append("| {} | {} | {:.4f} |".format(
                pair['feature1'], pair['feature2'], pair['correlation']))

        # 相关性热力图（Top 15 特征）
        top_iv_feats = sorted(iv_results, key=iv_results.get, reverse=True)[:15]
        top_iv_feats_in_matrix = [f for f in top_iv_feats if f in corr_matrix.index and f in corr_matrix.columns]
        if len(top_iv_feats_in_matrix) >= 3:
            sub_matrix = corr_matrix.loc[top_iv_feats_in_matrix, top_iv_feats_in_matrix]
            corr_labels = top_iv_feats_in_matrix
            corr_matrix_data = [[round(float(sub_matrix.iloc[i, j]), 4) for j in range(len(corr_labels))] for i in range(len(corr_labels))]
            heatmap_data = json.dumps({
                "labels": corr_labels,
                "matrix": corr_matrix_data
            }, ensure_ascii=False)
            lines.append('')
            lines.append('<!--ARTIFACT_START--><artifact type="correlation-heatmap" title="Top 15 特征相关性热力图">')
            lines.append(heatmap_data)
            lines.append('</artifact><!--ARTIFACT_END-->')
            lines.append('')

        lines.append("")
        lines.append("### 5.2 共线性建议")
        lines.append("")
        lines.append("基于高相关特征对，建议保留 IV 值较高的特征：")
        lines.append("")
        for pair in high_corr_pairs:
            f1, f2 = pair['feature1'], pair['feature2']
            iv1, iv2 = iv_results.get(f1, 0), iv_results.get(f2, 0)
            if iv1 >= iv2:
                keep, keep_iv, drop, drop_iv = f1, iv1, f2, iv2
            else:
                keep, keep_iv, drop, drop_iv = f2, iv2, f1, iv1
            lines.append("- 保留 **{}**（IV={:.4f}），移除 **{}**（IV={:.4f}）".format(
                keep, keep_iv, drop, drop_iv))
    else:
        lines.append("无高相关特征对（阈值 |r| ≥ 0.7）。")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. 综合建议
    qualified_pool = schemes.get('qualified_pool', [])
    lines.append("## 6. 综合建议")
    lines.append("")
    lines.append("### 6.1 推荐保留特征")
    lines.append("")
    lines.append("满足以下条件（基础合格池，共 {} 个特征）：".format(len(qualified_pool)))
    lines.append("- IV ≥ 0.02（有预测能力）")
    lines.append("- PSI < 0.25（稳定）")
    lines.append("- 缺失率 < 50%")
    lines.append("")
    lines.append("### 6.2 建议移除特征")
    lines.append("")
    lines.append("| 特征名 | 移除原因 |")
    lines.append("|--------|----------|")

    for f in features:
        reasons = []
        iv_val = iv_results.get(f, 0)
        if iv_val < 0.02:
            reasons.append("IV={:.4f}（无预测能力）".format(iv_val))
        if psi_results and psi_results.get(f, 0) >= 0.25:
            reasons.append("PSI={:.4f}（显著漂移）".format(psi_results.get(f, 0)))
        mr = stats_results.get(f, {}).get('miss_rate', 0)
        if mr >= 0.5:
            reasons.append("缺失率={:.2%}".format(mr))
        if reasons:
            lines.append("| {} | {} |".format(f, '; '.join(reasons)))

    lines.append("")
    lines.append("---")
    lines.append("")

    # 7. 建模特征集合方案
    scheme1 = schemes.get('scheme1', [])
    scheme2 = schemes.get('scheme2', [])
    scheme3 = schemes.get('scheme3', [])
    scheme4 = schemes.get('scheme4', [])

    lines.append("## 7. 建模特征集合方案")
    lines.append("")
    lines.append("### 7.1 方案一：全量入模方案")
    lines.append("")
    lines.append("> 适用场景：树模型（XGBoost/LightGBM）等对共线性不敏感的算法")
    lines.append("> 特征数量：{}".format(len(scheme1)))
    lines.append("")
    lines.append(render_scheme_table(scheme1, iv_results, psi_results, stats_results))

    lines.append("")
    lines.append("### 7.2 方案二：去共线性标准方案")
    lines.append("")
    lines.append("> 适用场景：逻辑回归、评分卡等对共线性敏感的模型")
    lines.append("> 去重阈值：|r| ≥ 0.7")
    lines.append("> 特征数量：{}".format(len(scheme2)))
    lines.append("")
    lines.append(render_scheme_table(scheme2, iv_results, psi_results, stats_results))

    lines.append("")
    lines.append("### 7.3 方案三：高预测力精选方案")
    lines.append("")
    lines.append("> 适用场景：特征数量受限、可解释性要求高")
    lines.append("> 筛选条件：IV ≥ 0.1 + |r| ≥ 0.7 去重")
    lines.append("> 特征数量：{}".format(len(scheme3)))
    lines.append("")
    lines.append(render_scheme_table(scheme3, iv_results, psi_results, stats_results))

    lines.append("")
    lines.append("### 7.4 方案四：稳定性优先方案")
    lines.append("")
    lines.append("> 适用场景：线上长期部署、高稳定性要求")
    lines.append("> 筛选条件：PSI < 0.1 + |r| ≥ 0.7 去重")
    lines.append("> 特征数量：{}".format(len(scheme4)))
    lines.append("")
    lines.append(render_scheme_table(scheme4, iv_results, psi_results, stats_results))

    # 方案对比总览
    lines.append("")
    lines.append("### 7.5 方案对比总览")
    lines.append("")
    lines.append("| 方案 | 特征数 | 平均IV | 适用场景 |")
    lines.append("|------|--------|--------|----------|")

    avg_iv_1 = sum(iv_results.get(f, 0) for f in scheme1) / max(len(scheme1), 1)
    avg_iv_2 = sum(iv_results.get(f, 0) for f in scheme2) / max(len(scheme2), 1)
    avg_iv_3 = sum(iv_results.get(f, 0) for f in scheme3) / max(len(scheme3), 1)
    avg_iv_4 = sum(iv_results.get(f, 0) for f in scheme4) / max(len(scheme4), 1)

    lines.append("| 全量入模 | {} | {:.4f} | 树模型 |".format(len(scheme1), avg_iv_1))
    lines.append("| 去共线性标准 | {} | {:.4f} | 逻辑回归/评分卡 |".format(len(scheme2), avg_iv_2))
    lines.append("| 高预测力精选 | {} | {:.4f} | 特征受限/可解释性 |".format(len(scheme3), avg_iv_3))
    lines.append("| 稳定性优先 | {} | {:.4f} | 线上部署/高稳定 |".format(len(scheme4), avg_iv_4))

    # 推荐方案标注
    lines.append("")
    lines.append("### 7.6 推荐方案")
    lines.append("")
    # 自动推荐逻辑
    recommend_name = ""
    recommend_reason = ""
    if len(scheme3) >= 5 and avg_iv_3 >= 0.1:
        recommend_name = "方案三（高预测力精选）"
        recommend_reason = "特征精简且平均预测力强，兼顾效果与可解释性"
    elif len(scheme4) >= 5 and psi_results:
        recommend_name = "方案四（稳定性优先）"
        recommend_reason = "特征稳定性良好，适合线上长期部署"
    elif len(scheme2) >= 3:
        recommend_name = "方案二（去共线性标准）"
        recommend_reason = "平衡了特征数量和共线性风险，适用性广"
    else:
        recommend_name = "方案一（全量入模）"
        recommend_reason = "合格特征数量有限，建议全部使用"
    lines.append("> **推荐方案**: {}。理由：{}。".format(recommend_name, recommend_reason))

    lines.append("")
    lines.append("---")
    lines.append("")

    # 单变量深度分析（可选）
    if top_n or specified_features:
        lines.append("## 8. 单变量深度分析")
        lines.append("")

        if top_n:
            top_n_features = sorted(iv_results, key=iv_results.get, reverse=True)[:top_n]
            lines.append("### 8.1 Top {} 特征分箱明细".format(top_n))
            lines.append("")

            for rank, feat in enumerate(top_n_features, 1):
                iv_val = iv_results.get(feat, 0)
                psi_val = psi_results.get(feat, None)
                mr = stats_results.get(feat, {}).get('miss_rate', None)
                iv_table = iv_calc.iv_tables.get(feat)

                psi_str = "{:.4f}（{}）".format(psi_val, psi_interpret(psi_val)) if psi_val else "未计算"
                mr_str = "{:.2%}".format(mr) if mr is not None else "未知"

                lines.append("#### [{rank}] {feat}".format(rank=rank, feat=feat))
                lines.append("")
                lines.append("- IV: {:.4f}（{}）".format(iv_val, iv_interpret(iv_val)))
                lines.append("- PSI: {}".format(psi_str))
                lines.append("- 缺失率: {}".format(mr_str))
                lines.append("")

                if iv_table is not None:
                    lines.append("**分箱明细表：**")
                    lines.append("")
                    lines.append(dataframe_to_markdown(iv_table))
                    lines.append("")

        if specified_features:
            lines.append("### 8.2 指定特征分箱明细")
            lines.append("")

            for feat in specified_features:
                if feat not in iv_results:
                    lines.append("#### {}（未找到）".format(feat))
                    lines.append("")
                    continue

                iv_val = iv_results.get(feat, 0)
                psi_val = psi_results.get(feat, None)
                mr = stats_results.get(feat, {}).get('miss_rate', None)
                iv_table = iv_calc.iv_tables.get(feat)

                # 风险标签
                tags = []
                if iv_val < 0.02:
                    tags.append("🔴 IV<0.02")
                if psi_val is not None and psi_val >= 0.25:
                    tags.append("🟡 PSI≥0.25")
                if mr is not None and mr >= 0.5:
                    tags.append("🟠 缺失率≥50%")
                risk_str = " | ".join(tags) if tags else "✅ 无风险"

                psi_str = "{:.4f}（{}）".format(psi_val, psi_interpret(psi_val)) if psi_val else "未计算"
                mr_str = "{:.2%}".format(mr) if mr is not None else "未知"

                lines.append("#### {}".format(feat))
                lines.append("")
                lines.append("- IV: {:.4f}（{}）".format(iv_val, iv_interpret(iv_val)))
                lines.append("- PSI: {}".format(psi_str))
                lines.append("- 缺失率: {}".format(mr_str))
                lines.append("- 风险评估: {}".format(risk_str))
                lines.append("")

                if iv_table is not None:
                    lines.append("**分箱明细表：**")
                    lines.append("")
                    lines.append(dataframe_to_markdown(iv_table))
                    lines.append("")

        lines.append("---")
        lines.append("")

    # 附录
    lines.append("## 附录：分析参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    lines.append("| IV最大分箱数 | 8 |")
    lines.append("| IV最小分箱比例 | 5% |")
    lines.append("| PSI分箱数 | 10 |")
    lines.append("| 相关性阈值 | 0.7 |")
    lines.append("| 相关性方法 | Pearson |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 分析结论
    lines.append("## 分析结论")
    lines.append("")
    conclusion_parts = []
    # 数据质量
    n_quality_issues = len(high_missing) + len(high_mode)
    if n_quality_issues == 0:
        conclusion_parts.append("数据质量良好，无明显缺失或常量问题")
    else:
        conclusion_parts.append("存在 {} 个质量问题特征（高缺失或高众数占比），建模前需处理".format(n_quality_issues))
    # 预测力
    strong_count = sum(1 for v in iv_results.values() if v >= 0.1)
    if strong_count >= 5:
        conclusion_parts.append("预测力充足，共 {} 个强预测特征（IV≥0.1）".format(strong_count))
    elif strong_count >= 2:
        conclusion_parts.append("预测力中等，{} 个强预测特征".format(strong_count))
    else:
        conclusion_parts.append("预测力较弱，仅 {} 个强预测特征，建议补充特征".format(strong_count))
    # 稳定性
    if psi_results:
        unstable_count = sum(1 for v in psi_results.values() if v >= 0.25)
        if unstable_count == 0:
            conclusion_parts.append("特征稳定性良好")
        else:
            conclusion_parts.append("{} 个特征存在显著漂移，建议谨慎使用".format(unstable_count))
    # 共线性
    if high_corr_pairs:
        conclusion_parts.append("发现 {} 对高相关特征，需根据模型类型决定是否去重".format(len(high_corr_pairs)))
    lines.append("。".join(conclusion_parts) + "。")
    lines.append("")
    lines.append("建议入模策略：优先使用{}，合格特征共 {} 个。".format(
        recommend_name, len(qualified_pool)))

    return "\n".join(lines)

# ============================================================
# 主流程
# ============================================================

def run_analysis(data_path: str, target: str, exclude_cols: List[str],
                 baseline_filter: Optional[str], comparison_filter: Optional[str],
                 top_n: Optional[int], specified_features: Optional[List[str]],
                 output_name: str, output_dir: Path) -> None:
    """执行完整分析流程
    
    Args:
        output_name: 报告输出名称
        output_dir: 产物输出目录
    """

    # 1. 加载数据
    print("\n" + "=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    data, features = load_data(data_path, target, exclude_cols)

    # 2. 基础统计分析
    print("\n" + "=" * 60)
    print("Step 2: 基础统计分析")
    print("=" * 60)
    stats_calc = StatisticsCalculator()
    stats_results = stats_calc.calculate_basic_statistics(data, features)
    print("完成 {} 个特征的统计分析".format(len(stats_results)))

    # 3. IV 值分析
    print("\n" + "=" * 60)
    print("Step 3: IV值分析")
    print("=" * 60)
    iv_calc = IVCalculator(max_n_bins=8, min_bin_size=0.05)
    iv_results = iv_calc.calculate_iv(data, target, features)
    print("完成 {} 个特征的IV计算".format(len(iv_results)))

    # 4. PSI 稳定性分析
    print("\n" + "=" * 60)
    print("Step 4: PSI稳定性分析")
    print("=" * 60)
    psi_results = {}
    psi_error_msg = None
    if baseline_filter and comparison_filter:
        psi_calc = PSICalculator(n_bins=10)
        try:
            baseline = data.query(baseline_filter)
            comparison = data.query(comparison_filter)
            print("基准集: {} 行，对比集: {} 行".format(len(baseline), len(comparison)))
            psi_results, _ = psi_calc.calculate(baseline, comparison, features)
            print("完成 {} 个特征的PSI计算".format(len(psi_results)))
        except Exception as e:
            # logger.exception 会自动附带完整 traceback，比 print(traceback.format_exc()) 更干净
            logger.exception("PSI 计算失败（baseline_filter=%r, comparison_filter=%r）", baseline_filter, comparison_filter)
            psi_error_msg = "{}: {}".format(type(e).__name__, e)
            psi_results = {}
    else:
        split_idx = int(len(data) * 0.7)
        if split_idx >= 100 and len(data) - split_idx >= 100:
            psi_calc = PSICalculator(n_bins=10)
            try:
                baseline = data.iloc[:split_idx]
                comparison = data.iloc[split_idx:]
                print("未指定PSI切分条件，使用 70/30 自动切分计算PSI")
                print("基准集: {} 行，对比集: {} 行".format(len(baseline), len(comparison)))
                psi_results, _ = psi_calc.calculate(baseline, comparison, features)
                print("完成 {} 个特征的PSI计算".format(len(psi_results)))
            except Exception as e:
                logger.exception("PSI 自动 70/30 计算失败")
                psi_error_msg = "{}: {}".format(type(e).__name__, e)
                psi_results = {}
        else:
            print("数据量不足，跳过PSI分析")

    # 5. 相关性分析
    print("\n" + "=" * 60)
    print("Step 5: 相关性分析")
    print("=" * 60)
    corr_calc = CorrelationCalculator()
    corr_summary = corr_calc.get_correlation_summary(data, features, method='pearson', threshold=0.7)
    corr_matrix = corr_summary['correlation_matrix']
    high_corr_pairs = corr_summary['high_correlation_pairs']
    print("发现 {} 对高相关特征（|r| >= 0.7）".format(len(high_corr_pairs)))

    # 6. 构建建模方案
    print("\n" + "=" * 60)
    print("Step 6: 构建建模特征方案")
    print("=" * 60)

    # 基础合格池
    qualified_pool = [
        f for f in features
        if iv_results.get(f, 0) >= 0.02
        and (not psi_results or psi_results.get(f, 0) < 0.25)
        and stats_results.get(f, {}).get('miss_rate', 1) < 0.5
    ]
    print("基础合格池: {} 个特征".format(len(qualified_pool)))

    # 构建各方案
    scheme1 = qualified_pool
    scheme2 = greedy_select(qualified_pool, iv_results, corr_matrix, 0.7)
    high_iv_pool = [f for f in qualified_pool if iv_results.get(f, 0) >= 0.1]
    scheme3 = greedy_select(high_iv_pool, iv_results, corr_matrix, 0.7)
    stable_pool = [f for f in qualified_pool if psi_results.get(f, 0) < 0.1] if psi_results else qualified_pool
    scheme4 = greedy_select(stable_pool, iv_results, corr_matrix, 0.7)

    schemes = {
        'qualified_pool': qualified_pool,
        'scheme1': scheme1,
        'scheme2': scheme2,
        'scheme3': scheme3,
        'scheme4': scheme4,
    }

    print("方案一（全量）: {} 个".format(len(scheme1)))
    print("方案二（去共线性）: {} 个".format(len(scheme2)))
    print("方案三（高IV）: {} 个".format(len(scheme3)))
    print("方案四（稳定性优先）: {} 个".format(len(scheme4)))

    # 7. 生成报告
    print("\n" + "=" * 60)
    print("Step 7: 生成报告")
    print("=" * 60)

    report = build_report(
        data=data, features=features, target=target, data_path=data_path,
        stats_results=stats_results, iv_results=iv_results, psi_results=psi_results,
        corr_matrix=corr_matrix, high_corr_pairs=high_corr_pairs,
        schemes=schemes, iv_calc=iv_calc,
        top_n=top_n, specified_features=specified_features
    )

    # 使用 ResultEmitter 落盘
    emitter = ResultEmitter(skill="feature-analysis", output_dir=output_dir)
    report_path = output_dir / f"{output_name}.md"
    report_path.write_text(report, encoding='utf-8')
    emitter.add_report(report_path)

    # Emit project state update
    iv_top20 = sorted(iv_results.items(), key=lambda x: x[1], reverse=True)[:20]
    emitter.update_metrics({
        "total_features_analyzed": len(features),
        "iv_top20": [{name: round(iv, 4)} for name, iv in iv_top20],
        "schemes": {
            "qualified_pool": len(scheme1),
            "decorrelated": len(scheme2),
            "high_iv": len(scheme3),
            "stability_first": len(scheme4),
        },
    })
    emitter.set_summary(f"特征分析完成: {len(features)} 个特征，方案二筛选 {len(scheme2)} 个")
    emitter.write(output_dir / "result.json")

    print("✅ 报告已保存至: {}".format(report_path))
    print("📊 分析完成：{} 个特征，样本量 {}".format(len(features), len(data)))
    
    return {"path": str(report_path)}

def main():
    parser = argparse.ArgumentParser(
        description="建模特征分析工具 - 生成包含统计、IV、PSI、相关性的完整特征分析报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--data_path', required=True, help='数据文件路径（parquet/csv）')
    parser.add_argument('--target', required=True, help='目标变量列名（0/1二分类）')
    parser.add_argument('--exclude_cols', default='', help='排除列，逗号分隔')
    parser.add_argument('--baseline_filter', help='PSI基准数据筛选条件（pandas query字符串）')
    parser.add_argument('--comparison_filter', help='PSI对比数据筛选条件（pandas query字符串）')
    parser.add_argument('--top_n', type=int, default=10, help='输出IV排名前N的特征分箱明细（默认10）')
    parser.add_argument('--specified_features', help='指定特征列表，逗号分隔')
    parser.add_argument('--output', default=None, help='报告输出名称')
    parser.add_argument('--output_dir', default=None, help='产物输出目录（默认 ./outputs/<timestamp>）')

    args = parser.parse_args()

    exclude_cols = [c.strip() for c in args.exclude_cols.split(',') if c.strip()]
    specified_features = [c.strip() for c in args.specified_features.split(',') if c.strip()] \
        if args.specified_features else None

    output_name = args.output if args.output else 'feature_analysis_report'
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_analysis(
        data_path=args.data_path,
        target=args.target,
        exclude_cols=exclude_cols,
        baseline_filter=args.baseline_filter,
        comparison_filter=args.comparison_filter,
        top_n=args.top_n,
        specified_features=specified_features,
        output_name=output_name,
        output_dir=output_dir,
    )

if __name__ == '__main__':
    main()
