#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单变量分析主脚本

支持两种模式：
1. 数据探索模式（无目标变量）：分布分析、交叉分布
2. 特征筛选模式（有目标变量）：IV值计算、筛选建议
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from data_utils import read_dataset  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

# 添加脚本目录到路径
sys.path.insert(0, str(_SCRIPT_DIR))

from binning import BinningCalculator, calculate_basic_statistics  # noqa: E402
from cross_distribution import CrossDistributionCalculator, format_cross_table_markdown  # noqa: E402

def load_data(data_path: str) -> pd.DataFrame:
    """
    加载数据文件（不强制 0/1 过滤，保持原行为）

    Args:
        data_path: 数据文件路径（支持 parquet/csv）

    Returns:
        DataFrame
    """
    return read_dataset(data_path, label_filter=None)

def parse_features(features_str: str, df: pd.DataFrame, exclude_cols: Optional[str] = None) -> List[str]:
    """
    解析特征列表
    
    Args:
        features_str: 特征字符串，逗号分隔
        df: 数据框
        exclude_cols: 排除的列
        
    Returns:
        特征列表
    """
    if features_str == 'all':
        features = df.columns.tolist()
    else:
        features = [f.strip() for f in features_str.split(',')]
    
    # 排除指定列
    if exclude_cols:
        exclude_list = [c.strip() for c in exclude_cols.split(',')]
        features = [f for f in features if f not in exclude_list]
    
    # 验证特征存在
    valid_features = [f for f in features if f in df.columns]
    invalid_features = [f for f in features if f not in df.columns]
    
    if invalid_features:
        print('警告: 以下特征不存在于数据中: {}'.format(', '.join(invalid_features)))
    
    return valid_features

def identify_quality_issues(stats_list: List[Dict], df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    识别数据质量问题
    
    Args:
        stats_list: 基础统计结果列表
        df: 数据框
        
    Returns:
        质量问题字典
    """
    issues = {
        'high_missing': [],      # 缺失率 > 30%
        'low_cardinality': [],   # 基数 <= 2
        'single_value': [],      # 单一值（基数=1）
        'high_mode_ratio': [],   # 众数占比 > 90%
    }
    
    for stat in stats_list:
        feature = stat['feature']
        
        # 缺失率检查
        missing_rate_str = stat.get('missing_rate', '0%')
        try:
            missing_rate = float(missing_rate_str.rstrip('%')) / 100
            if missing_rate > 0.3:
                issues['high_missing'].append(feature)
        except:
            pass
        
        # 基数检查
        cardinality = stat.get('cardinality', 0)
        if cardinality == 1:
            issues['single_value'].append(feature)
        elif cardinality <= 2:
            issues['low_cardinality'].append(feature)
        
        # 众数占比检查
        if feature in df.columns:
            series = df[feature].dropna()
            if len(series) > 0:
                mode_ratio = series.value_counts().iloc[0] / len(series)
                if mode_ratio > 0.9:
                    issues['high_mode_ratio'].append(feature)
    
    return issues

def build_exploration_report(
    df: pd.DataFrame,
    features: List[str],
    stats_list: List[Dict],
    distribution_results: List[Dict],
    cross_result: Optional[Dict],
    quality_issues: Dict,
    binning_method: str,
    n_bins: int
) -> str:
    """
    构建数据探索模式的报告
    
    Args:
        df: 数据框
        features: 特征列表
        stats_list: 基础统计结果
        distribution_results: 分布分析结果
        cross_result: 交叉分布结果
        quality_issues: 数据质量问题
        binning_method: 分箱方法
        n_bins: 分箱数量
        
    Returns:
        Markdown 格式报告
    """
    sections = []
    
    # 标题
    sections.append('# 单变量分析报告（数据探索模式）\n')
    sections.append('> 生成时间: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # 1. 数据概览
    sections.append('\n## 1. 数据概览\n')
    sections.append('| 指标 | 值 |')
    sections.append('|------|-----|')
    sections.append('| 样本量 | {:,} |'.format(len(df)))
    sections.append('| 分析特征数 | {} |'.format(len(features)))
    sections.append('| 分箱方式 | {} |'.format('等频' if binning_method == 'quantile' else '等距'))
    sections.append('| 分箱数量 | {} |'.format(n_bins))
    
    # 2. 基础统计
    sections.append('\n## 2. 基础统计\n')
    
    stats_df = pd.DataFrame(stats_list)
    cols_order = ['feature', 'count', 'missing_count', 'missing_rate', 'cardinality', 
                  'mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']
    cols_present = [c for c in cols_order if c in stats_df.columns]
    stats_df = stats_df[cols_present]
    
    col_names = {
        'feature': '特征', 'count': '样本数', 'missing_count': '缺失数',
        'missing_rate': '缺失率', 'cardinality': '基数', 'mean': '均值',
        'median': '中位数', 'std': '标准差', 'min': '最小值', 'max': '最大值',
        'skewness': '偏度', 'kurtosis': '峰度'
    }
    stats_df = stats_df.rename(columns=col_names)
    sections.append(stats_df.to_markdown(index=False))
    
    # 3. 分布分析
    sections.append('\n## 3. 分布分析\n')
    
    for dist in distribution_results:
        feature = dist['feature']
        method_name = '等频' if dist['method'] == 'quantile' else '等距'
        
        sections.append('\n### 特征: {} | 分箱方式: {} | 分箱数: {}\n'.format(
            feature, method_name, dist['n_bins']))
        
        # 分布形态描述
        if feature in df.columns:
            series = df[feature].dropna()
            if len(series) > 0 and pd.api.types.is_numeric_dtype(series):
                skew = series.skew()
                kurt = series.kurtosis()
                shape_parts = []
                if abs(skew) < 0.5:
                    shape_parts.append('近似对称')
                elif skew > 2:
                    shape_parts.append('高度右偏')
                elif skew > 0.5:
                    shape_parts.append('右偏')
                elif skew < -2:
                    shape_parts.append('高度左偏')
                else:
                    shape_parts.append('左偏')
                if kurt > 3:
                    shape_parts.append('尖峰')
                elif kurt < -1:
                    shape_parts.append('平坦')
                sections.append('> 分布形态：{}（偏度={:.2f}，峰度={:.2f}）\n'.format(
                    '、'.join(shape_parts), skew, kurt))
        
        if dist['distribution_table'] is not None:
            table = dist['distribution_table'].copy()
            raw_table = dist['distribution_table']
            table.columns = ['区间', '样本数', '占比', '累计占比']
            sections.append(table.to_markdown(index=False))

            # 分布图 Artifact
            try:
                import json
                bins_list = [str(b) for b in raw_table.iloc[:, 0].values]
                counts_list = [int(c) for c in raw_table.iloc[:, 1].values]
                payload = json.dumps({
                    'bins': bins_list,
                    'counts': counts_list,
                    'positive_rates': [],
                }, ensure_ascii=False)
                sections.append('')
                sections.append('<!--ARTIFACT_START--><artifact type="binning-chart" title="{} 分布图">'.format(feature))
                sections.append(payload)
                sections.append('</artifact><!--ARTIFACT_END-->')
            except Exception:
                pass
        else:
            sections.append('无法计算分布（数据为空）')
    
    # 4. 交叉分布
    if cross_result and cross_result.get('status') == 'success':
        sections.append('\n## 4. 交叉分布\n')
        sections.append(format_cross_table_markdown(cross_result, include_ratio=True))
    
    # 5. 数据质量标记
    sections.append('\n## 5. 数据质量标记\n')
    
    has_issues = False
    
    if quality_issues['high_missing']:
        has_issues = True
        sections.append('\n### 高缺失率特征（>30%）\n')
        sections.append('- ' + '\n- '.join(quality_issues['high_missing']))
    
    if quality_issues['single_value']:
        has_issues = True
        sections.append('\n### 单一值特征（建议剔除）\n')
        sections.append('- ' + '\n- '.join(quality_issues['single_value']))
    
    if quality_issues['low_cardinality']:
        has_issues = True
        sections.append('\n### 低基数特征（基数≤2）\n')
        sections.append('- ' + '\n- '.join(quality_issues['low_cardinality']))
    
    if quality_issues['high_mode_ratio']:
        has_issues = True
        sections.append('\n### 高众数占比特征（>90%）\n')
        sections.append('- ' + '\n- '.join(quality_issues['high_mode_ratio']))
    
    if not has_issues:
        sections.append('\n无明显数据质量问题。')
    
    # 6. 报告总结
    sections.append('\n## 6. 探索总结\n')
    numeric_count = sum(1 for f in features if pd.api.types.is_numeric_dtype(df[f]))
    cat_count = len(features) - numeric_count
    issue_count = sum(len(v) for v in quality_issues.values())
    sections.append('共分析 {} 个特征（数值型 {} 个、类别型 {} 个），样本量 {:,}。'.format(
        len(features), numeric_count, cat_count, len(df)))
    if issue_count > 0:
        sections.append('发现 {} 个质量问题特征需关注。'.format(issue_count))
    else:
        sections.append('数据质量良好，可进入下一步分析。')
    
    return '\n'.join(sections)

def build_screening_report(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    stats_list: List[Dict],
    iv_results: List[Dict],
    quality_issues: Dict,
    distribution_results: List[Dict] = None
) -> str:
    """
    构建特征筛选模式的报告
    
    Args:
        df: 数据框
        features: 特征列表
        target: 目标变量
        stats_list: 基础统计结果
        iv_results: IV 计算结果
        quality_issues: 数据质量问题
        distribution_results: 等频分箱分布结果（可选，用于对比展示）
        
    Returns:
        Markdown 格式报告
    """
    sections = []
    
    # 标题
    sections.append('# 单变量分析报告（特征筛选模式）\n')
    sections.append('> 生成时间: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # 计算正样本率
    positive_rate = df[target].mean()
    
    # 1. 数据概览
    sections.append('\n## 1. 数据概览\n')
    sections.append('| 指标 | 值 |')
    sections.append('|------|-----|')
    sections.append('| 样本量 | {:,} |'.format(len(df)))
    sections.append('| 目标变量 | {} |'.format(target))
    sections.append('| 正样本率 | {:.2%} |'.format(positive_rate))
    sections.append('| 分析特征数 | {} |'.format(len(features)))
    
    # 2. 基础统计
    sections.append('\n## 2. 基础统计\n')
    
    stats_df = pd.DataFrame(stats_list)
    cols_order = ['feature', 'count', 'missing_count', 'missing_rate', 'cardinality', 
                  'mean', 'median', 'std', 'min', 'max']
    cols_present = [c for c in cols_order if c in stats_df.columns]
    stats_df = stats_df[cols_present]
    
    col_names = {
        'feature': '特征', 'count': '样本数', 'missing_count': '缺失数',
        'missing_rate': '缺失率', 'cardinality': '基数', 'mean': '均值',
        'median': '中位数', 'std': '标准差', 'min': '最小值', 'max': '最大值'
    }
    stats_df = stats_df.rename(columns=col_names)
    sections.append(stats_df.to_markdown(index=False))
    
    # 3. IV 值分析
    sections.append('\n## 3. IV值分析\n')
    
    # IV 排名表
    valid_iv = [r for r in iv_results if r['status'] == 'success']
    if valid_iv:
        iv_summary = []
        for r in valid_iv:
            iv_summary.append({
                'feature': r['feature'],
                'iv': r['iv_value'],
                'level': _iv_level(r['iv_value'])
            })
        
        iv_df = pd.DataFrame(iv_summary)
        iv_df = iv_df.sort_values('iv', ascending=False)
        iv_df['iv'] = iv_df['iv'].apply(lambda x: '{:.4f}'.format(x))
        iv_df.columns = ['特征', 'IV值', '预测力等级']
        
        sections.append('\n### IV 排名\n')
        sections.append(iv_df.to_markdown(index=False))
        
        # IV 分布总结
        strong_count = sum(1 for r in valid_iv if r['iv_value'] >= 0.1)
        medium_count = sum(1 for r in valid_iv if 0.02 <= r['iv_value'] < 0.1)
        weak_count = sum(1 for r in valid_iv if r['iv_value'] < 0.02)
        sections.append('\n**IV 分布总结**: {} 个强预测力特征（IV≥0.1），{} 个弱预测力（IV 0.02~0.1），{} 个无预测能力（IV<0.02）'.format(
            strong_count, medium_count, weak_count))
        
        # 各特征分箱明细
        sections.append('\n### 分箱明细\n')
        for r in sorted(valid_iv, key=lambda x: x['iv_value'], reverse=True):
            sections.append('\n#### 特征: {} | IV = {:.4f}\n'.format(r['feature'], r['iv_value']))
            if r['iv_table'] is not None:
                table = r['iv_table'].copy()
                table.columns = ['区间', '样本数', '占比', '正样本数', '正样本率', 'WoE', 'IV']
                sections.append(table.to_markdown(index=False))

                # 分箱分布图 Artifact
                try:
                    bins_list = [str(b) for b in r['iv_table'].iloc[:, 0].values]
                    counts_list = [int(c) for c in r['iv_table'].iloc[:, 1].values]
                    pos_rates_list = r.get('positive_rates', [])
                    import json
                    binning_payload = {
                        'bins': bins_list,
                        'counts': counts_list,
                        'positive_rates': pos_rates_list,
                        'overall_rate': round(float(positive_rate), 6),
                    }
                    binning_json = json.dumps(binning_payload, ensure_ascii=False)
                    sections.append('')
                    sections.append('<!--ARTIFACT_START--><artifact type="binning-chart" title="{} 最优分箱分布图">'.format(r['feature']))
                    sections.append(binning_json)
                    sections.append('</artifact><!--ARTIFACT_END-->')
                except Exception:
                    pass  # 分箱图生成失败不影响报告
                
                # WoE 趋势描述
                woe_vals = r['iv_table'].iloc[:, 5].values  # WoE 列
                if len(woe_vals) >= 3:
                    is_monotonic_inc = all(woe_vals[i] <= woe_vals[i+1] for i in range(len(woe_vals)-1))
                    is_monotonic_dec = all(woe_vals[i] >= woe_vals[i+1] for i in range(len(woe_vals)-1))
                    if is_monotonic_inc:
                        sections.append('\n> WoE 单调递增，分箱效果良好')
                    elif is_monotonic_dec:
                        sections.append('\n> WoE 单调递减，分箱效果良好')
                    else:
                        sections.append('\n> WoE 非单调，分箱可能需调整或特征存在非线性关系')
    else:
        sections.append('\n无有效 IV 计算结果。')
    
    # 4. 等频分箱分布对比（可选）
    if distribution_results:
        sections.append('\n## 4. 等频分箱分布对比\n')
        sections.append('> 以下展示等频分箱（Quantile Binning）的分布情况，与上方的最优分箱（Optimal Binning）形成对比。等频分箱每箱样本量均衡，适合观察原始数据分布形态。\n')
        
        for dist in distribution_results:
            feature = dist['feature']
            method_name = '等频' if dist['method'] == 'quantile' else '等距'
            
            sections.append('\n### 特征: {} | 分箱方式: {} | 分箱数: {}\n'.format(
                feature, method_name, dist['n_bins']))
            
            if dist['distribution_table'] is not None:
                table = dist['distribution_table'].copy()
                raw_table = dist['distribution_table']
                table.columns = ['区间', '样本数', '占比', '累计占比']
                sections.append(table.to_markdown(index=False))
                
                # 等频分箱分布图 Artifact
                try:
                    import json
                    bins_list = [str(b) for b in raw_table.iloc[:, 0].values]
                    counts_list = [int(c) for c in raw_table.iloc[:, 1].values]
                    payload = json.dumps({
                        'bins': bins_list,
                        'counts': counts_list,
                        'positive_rates': [],
                    }, ensure_ascii=False)
                    sections.append('')
                    sections.append('<!--ARTIFACT_START--><artifact type="binning-chart" title="{} 等频分箱分布图">'.format(feature))
                    sections.append(payload)
                    sections.append('</artifact><!--ARTIFACT_END-->')
                except Exception:
                    pass
            else:
                sections.append('无法计算分布（数据为空）')
    
    # 5. 数据质量标记
    sections.append('\n## 5. 数据质量标记\n')
    
    has_issues = False
    
    if quality_issues['high_missing']:
        has_issues = True
        sections.append('\n### 高缺失率特征（>30%）\n')
        sections.append('- ' + '\n- '.join(quality_issues['high_missing']))
    
    if quality_issues['single_value']:
        has_issues = True
        sections.append('\n### 单一值特征（建议剔除）\n')
        sections.append('- ' + '\n- '.join(quality_issues['single_value']))
    
    if not has_issues:
        sections.append('\n无明显数据质量问题。')
    
    # 6. 筛选建议
    sections.append('\n## 6. 筛选建议\n')
    
    # 建议保留的特征
    recommend_keep = []
    recommend_remove = []
    
    for r in iv_results:
        feature = r['feature']
        
        # 排除条件
        if feature in quality_issues['single_value']:
            recommend_remove.append({'feature': feature, 'reason': '单一值'})
            continue
        if feature in quality_issues['high_missing']:
            recommend_remove.append({'feature': feature, 'reason': '高缺失率(>30%)'})
            continue
        
        if r['status'] == 'success':
            if r['iv_value'] >= 0.02:
                recommend_keep.append({'feature': feature, 'iv': r['iv_value']})
            else:
                recommend_remove.append({'feature': feature, 'reason': 'IV过低(<0.02)'})
        else:
            recommend_remove.append({'feature': feature, 'reason': 'IV计算失败'})
    
    sections.append('\n### 建议保留（IV≥0.02）\n')
    if recommend_keep:
        keep_df = pd.DataFrame(recommend_keep)
        keep_df = keep_df.sort_values('iv', ascending=False)
        keep_df['reason'] = keep_df['iv'].apply(
            lambda x: '强预测力' if x >= 0.1 else ('中等预测力' if x >= 0.02 else '弱'))
        keep_df['iv'] = keep_df['iv'].apply(lambda x: '{:.4f}'.format(x))
        keep_df.columns = ['特征', 'IV值', '保留理由']
        sections.append(keep_df.to_markdown(index=False))
        sections.append('\n共 {} 个特征建议保留。'.format(len(recommend_keep)))
    else:
        sections.append('无')
    
    sections.append('\n### 建议剔除\n')
    if recommend_remove:
        remove_df = pd.DataFrame(recommend_remove)
        remove_df.columns = ['特征', '剔除原因']
        sections.append(remove_df.to_markdown(index=False))
        sections.append('\n共 {} 个特征建议剔除。'.format(len(recommend_remove)))
    else:
        sections.append('无')
    
    # 7. 筛选总结
    sections.append('\n## 7. 筛选总结\n')
    strong = sum(1 for r in iv_results if r['status'] == 'success' and r['iv_value'] >= 0.1)
    medium = sum(1 for r in iv_results if r['status'] == 'success' and 0.02 <= r['iv_value'] < 0.1)
    sections.append('共分析 {} 个特征，{} 个建议保留（其中 {} 个强预测力、{} 个中等预测力），{} 个建议剔除。'.format(
        len(features), len(recommend_keep), strong, medium, len(recommend_remove)))
    if strong >= 3:
        sections.append('预测力充足，可直接进入建模阶段。')
    elif len(recommend_keep) > 0:
        sections.append('预测力一般，建议补充更多特征或进行特征工程。')
    else:
        sections.append('当前特征预测力不足，建议重新审视数据源。')
    
    return '\n'.join(sections)

def _iv_level(iv: float) -> str:
    """IV 值等级判断"""
    if iv < 0.02:
        return '无预测力'
    elif iv < 0.1:
        return '弱'
    elif iv < 0.3:
        return '中等'
    elif iv < 0.5:
        return '强'
    else:
        return '极强（可能过拟合）'

def run_analysis(args) -> str:
    """
    执行单变量分析
    
    Args:
        args: 命令行参数
        
    Returns:
        报告内容
    """
    # 加载数据
    print('加载数据: {}'.format(args.data_path))
    df = load_data(args.data_path)
    print('数据加载完成，样本量: {:,}'.format(len(df)))
    
    # 解析特征
    features = parse_features(args.features, df, getattr(args, 'exclude_cols', None))
    
    # 排除目标变量
    if args.target and args.target in features:
        features.remove(args.target)
    
    print('分析特征数: {}'.format(len(features)))
    
    # 判断模式
    has_target = args.target and args.target in df.columns
    mode = '特征筛选' if has_target else '数据探索'
    print('运行模式: {}'.format(mode))
    
    # 初始化计算器
    binning_calc = BinningCalculator(n_bins=args.n_bins, method=args.binning_method)
    
    # 1. 计算基础统计
    print('计算基础统计...')
    stats_list = []
    for feature in features:
        stats = calculate_basic_statistics(df[feature], feature)
        stats_list.append(stats)
    
    # 2. 识别数据质量问题
    quality_issues = identify_quality_issues(stats_list, df)
    
    if has_target:
        # 特征筛选模式
        print('计算 IV 值...')
        iv_results = []
        for feature in features:
            result = binning_calc.calculate_iv_distribution(
                df[feature], df[args.target], feature
            )
            iv_results.append(result)
            if result['status'] == 'success':
                print('  {} IV = {:.4f}'.format(feature, result['iv_value']))
            else:
                err_msg = result.get('error_message', '')
                print('  {} 计算失败: {} {}'.format(feature, result['status'], err_msg))
        
        # 额外计算等频分箱分布（用于对比展示）
        print('计算等频分箱分布（用于对比）...')
        quantile_calc = BinningCalculator(n_bins=args.n_bins, method='quantile')
        distribution_results = []
        for feature in features:
            result = quantile_calc.calculate_distribution(df[feature], feature)
            distribution_results.append(result)
        
        report = build_screening_report(
            df, features, args.target, stats_list, iv_results, quality_issues,
            distribution_results=distribution_results
        )
    else:
        # 数据探索模式
        print('计算分布...')
        distribution_results = []
        for feature in features:
            result = binning_calc.calculate_distribution(df[feature], feature)
            distribution_results.append(result)
        
        # 交叉分布
        cross_result = None
        if args.cross:
            cross_features = [f.strip() for f in args.cross.split(',')]
            if len(cross_features) == 2:
                print('计算交叉分布: {} × {}'.format(*cross_features))
                cross_calc = CrossDistributionCalculator(n_bins=5, method=args.binning_method)
                cross_result = cross_calc.calculate(df, cross_features[0], cross_features[1])
        
        report = build_exploration_report(
            df, features, stats_list, distribution_results, cross_result,
            quality_issues, args.binning_method, args.n_bins
        )
    
    return report

def main():
    parser = argparse.ArgumentParser(description='单变量分析')
    
    parser.add_argument('--data_path', required=True, help='数据文件路径')
    parser.add_argument('--features', required=True, help='特征列名，逗号分隔，或 "all"')
    parser.add_argument('--target', help='目标变量列名（可选，有则进入筛选模式）')
    parser.add_argument('--exclude_cols', help='排除的列，逗号分隔')
    parser.add_argument('--binning_method', default='quantile', 
                        choices=['quantile', 'distance'], help='分箱方式')
    parser.add_argument('--n_bins', type=int, default=10, help='分箱数量')
    parser.add_argument('--cross', help='交叉分布的两个特征，逗号分隔')
    parser.add_argument('--output', help='报告输出名称')
    parser.add_argument('--output_dir', default=None, help='产物输出目录（默认 ./outputs/<timestamp>）')
    
    args = parser.parse_args()
    
    # 执行分析
    report = run_analysis(args)
    
    # 使用 ResultEmitter 落盘
    output_name = args.output if args.output else 'univariate_analysis_report'
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emitter = ResultEmitter(skill="univariate-analysis", output_dir=output_dir)
    report_path = output_dir / f"{output_name}.md"
    report_path.write_text(report, encoding='utf-8')
    emitter.add_report(report_path)
    emitter.set_summary("单变量分析完成")
    emitter.write(output_dir / "result.json")
    
    print('\n报告已保存至: {}'.format(report_path))

if __name__ == '__main__':
    main()
