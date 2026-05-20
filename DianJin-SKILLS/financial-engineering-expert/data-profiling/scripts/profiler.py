# -*- coding: utf-8 -*-

"""
数据洞察分析器

对数据集进行轮廓分析，生成数据概况报告
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
from report_writer import ReportWriter

MAX_DISPLAY = 20

def get_file_size(file_path: str) -> str:
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def infer_column_type(series: pd.Series) -> str:
    """推断字段基础存储类型"""
    dtype = series.dtype
    non_null = series.dropna()

    if series.nunique() == 2:
        vals = set(non_null.unique())
        if vals <= {0, 1} or vals <= {True, False} or vals <= {'是', '否'} or vals <= {'yes', 'no'}:
            return "布尔型"

    if pd.api.types.is_numeric_dtype(dtype):
        return "数值型"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "时间型"

    if dtype == object:
        try:
            pd.to_datetime(non_null.iloc[:100], errors='raise')
            return "时间型"
        except:
            pass

    if dtype == object or pd.api.types.is_categorical_dtype(dtype):
        if len(series) > 0 and series.nunique() / len(series) > 0.8:
            return "文本型"
        return "类别型"

    return "其他"

def _section_title(base: str, shown: int, total: int) -> str:
    if shown < total:
        return f"{base}（展示前 {shown} 个，共 {total} 个）"
    return base

def _artifact(chart_type: str, title: str, data: dict) -> List[str]:
    return [
        f'<!--ARTIFACT_START--><artifact type="chart" chart-type="{chart_type}" title="{title}">',
        json.dumps(data, ensure_ascii=False),
        '</artifact><!--ARTIFACT_END-->',
        '',
    ]

def generate_profiling_report(data_path: str, output_name: str = None) -> str:
    if output_name is None:
        output_name = 'data_profiling_report'

    print(f"正在读取数据: {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("仅支持 parquet 或 csv 格式文件")

    file_name = os.path.basename(data_path)
    file_format = "Parquet" if data_path.endswith('.parquet') else "CSV"
    file_size = get_file_size(data_path)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    n_rows, n_cols = df.shape

    # 推断类型
    column_types = {col: infer_column_type(df[col]) for col in df.columns}
    type_counts = {}
    for t in column_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    numeric_cols = [col for col, t in column_types.items() if t == "数值型"]
    cat_cols = [col for col, t in column_types.items() if t == "类别型"]

    lines = []

    # === 1. 数据概览 ===
    lines.append("# 数据洞察报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")
    lines.append("## 1. 数据概览\n")
    lines.append("| 属性 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 文件名 | {file_name} |")
    lines.append(f"| 文件格式 | {file_format} |")
    lines.append(f"| 文件大小 | {file_size} |")
    lines.append(f"| 内存占用 | {memory_mb:.2f} MB |")
    lines.append(f"| 总行数 | {n_rows:,} |")
    lines.append(f"| 总列数 | {n_cols} |")
    lines.append("")

    # 类型分布饼图
    type_order = ["数值型", "类别型", "布尔型", "时间型", "文本型", "其他"]
    type_labels = [t for t in type_order if t in type_counts]
    type_values = [type_counts[t] for t in type_labels]
    if type_labels:
        lines.append("| 类型 | 数量 | 占比 |")
        lines.append("|------|:----:|:----:|")
        for t in type_labels:
            lines.append(f"| {t} | {type_counts[t]} | {type_counts[t] / n_cols * 100:.1f}% |")
        lines.append("")
        lines += _artifact("pie", "字段类型分布", {"labels": type_labels, "values": type_values})

    # === 2. 字段详情清单 ===
    lines.append("## 2. 字段详情清单\n")
    lines.append("| # | 字段名 | 类型 | 缺失率 | 唯一值数 | 示例值 |")
    lines.append("|:--:|--------|------|:------:|:--------:|--------|")
    for idx, col in enumerate(df.columns, 1):
        series = df[col]
        col_type = column_types[col]
        missing_pct = series.isna().mean() * 100
        n_unique = series.nunique()
        samples = series.dropna().head(3).tolist()
        sample_str = ", ".join(str(v)[:15] for v in samples)
        if len(sample_str) > 35:
            sample_str = sample_str[:32] + "..."
        lines.append(f"| {idx} | {col} | {col_type} | {missing_pct:.1f}% | {n_unique:,} | {sample_str} |")
    lines.append("")

    # === 3. 数值特征分析 ===
    if numeric_cols:
        show_cols = numeric_cols[:MAX_DISPLAY]
        title = _section_title("## 3. 数值特征分析", len(show_cols), len(numeric_cols))
        lines.append(f"{title}\n")
        lines.append("| 特征名 | 均值 | 标准差 | 最小值 | P25 | P50 | P75 | 最大值 | 偏度 | 峰度 | 缺失率 |")
        lines.append("|--------|------|--------|--------|-----|-----|-----|--------|------|------|--------|")
        for col in show_cols:
            s = df[col].dropna()
            if len(s) == 0:
                lines.append(f"| {col} | - | - | - | - | - | - | - | - | - | {df[col].isna().mean()*100:.1f}% |")
                continue
            lines.append(
                f"| {col} | {s.mean():.4g} | {s.std():.4g} | {s.min():.4g} "
                f"| {s.quantile(0.25):.4g} | {s.quantile(0.5):.4g} | {s.quantile(0.75):.4g} "
                f"| {s.max():.4g} | {s.skew():.2f} | {s.kurtosis():.2f} "
                f"| {df[col].isna().mean()*100:.1f}% |"
            )
        lines.append("")

        # histogram: Top 10 特征各自分箱
        hist_cols = show_cols[:10]
        for col in hist_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            counts, bin_edges = np.histogram(s, bins=10)
            labels = [f"{bin_edges[i]:.3g}~{bin_edges[i+1]:.3g}" for i in range(len(counts))]
            lines += _artifact("bar", f"{col} 分布", {"labels": labels, "values": counts.tolist()})

    # === 4. 类别特征分析 ===
    if cat_cols:
        show_cols = cat_cols[:MAX_DISPLAY]
        title = _section_title("## 4. 类别特征分析", len(show_cols), len(cat_cols))
        lines.append(f"{title}\n")
        lines.append("| 特征名 | 唯一值数 | Top1值 | Top1占比 | Top2值 | Top2占比 | Top5集中度 | 缺失率 |")
        lines.append("|--------|:--------:|--------|:--------:|--------|:--------:|:----------:|:------:|")
        for col in show_cols:
            s = df[col].dropna()
            n_unique = s.nunique()
            vc = s.value_counts()
            top1_val = str(vc.index[0])[:12] if len(vc) > 0 else '-'
            top1_pct = vc.iloc[0] / len(s) * 100 if len(vc) > 0 else 0
            top2_val = str(vc.index[1])[:12] if len(vc) > 1 else '-'
            top2_pct = vc.iloc[1] / len(s) * 100 if len(vc) > 1 else 0
            top5_pct = vc.iloc[:5].sum() / len(s) * 100 if len(vc) > 0 else 0
            missing_pct = df[col].isna().mean() * 100
            lines.append(
                f"| {col} | {n_unique} | {top1_val} | {top1_pct:.1f}% "
                f"| {top2_val} | {top2_pct:.1f}% | {top5_pct:.1f}% | {missing_pct:.1f}% |"
            )
        lines.append("")

        # bar chart: Top 10 类别特征各自 Top 5 值
        chart_cols = show_cols[:10]
        for col in chart_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            vc = s.value_counts().head(5)
            labels = [str(v)[:15] for v in vc.index]
            values = (vc / len(s) * 100).round(2).tolist()
            lines += _artifact("bar", f"{col} Top5 分布(%)", {"labels": labels, "values": values})

    # === 5. 缺失值分析 ===
    missing_info = []
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > 0:
            missing_info.append((col, rate))
    if missing_info:
        missing_info.sort(key=lambda x: x[1], reverse=True)
        lines.append("## 5. 缺失值分析\n")
        top_missing = missing_info[:10]
        lines.append("| 字段名 | 缺失率 | 缺失数 |")
        lines.append("|--------|:------:|:------:|")
        for col, rate in top_missing:
            lines.append(f"| {col} | {rate*100:.1f}% | {int(rate * n_rows):,} |")
        lines.append("")
        lines += _artifact("bar", "缺失率 Top 10（%）", {
            "labels": [m[0] for m in top_missing],
            "values": [round(m[1] * 100, 2) for m in top_missing]
        })

    # === 6. 数据质量 ===
    dup_count = df.duplicated().sum()
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    constant_cols = [col for col in df.columns if df[col].nunique() == 1 and df[col].notna().all()]
    high_missing_cols = [(col, rate) for col, rate in missing_info if rate > 0.5]

    has_quality_issues = dup_count > 0 or empty_cols or constant_cols or high_missing_cols
    if has_quality_issues:
        lines.append("## 6. 数据质量\n")
        if dup_count > 0:
            lines.append(f"- **重复行**: {dup_count:,} 行（占比 {dup_count/n_rows*100:.2f}%）")
        if empty_cols:
            lines.append(f"- **完全空列**（{len(empty_cols)}个）: {', '.join(empty_cols)}")
        if constant_cols:
            lines.append(f"- **常量列**（{len(constant_cols)}个）: {', '.join(constant_cols)}")
        if high_missing_cols:
            lines.append(f"- **高缺失率字段**（>50%）:")
            for col, rate in high_missing_cols:
                lines.append(f"  - {col}: {rate*100:.1f}%")
        lines.append("")

    # === 7. 样本预览 ===
    lines.append("## 7. 样本预览（前 5 行）\n")
    preview_df = df.head(5)
    show_preview_cols = preview_df.columns[:8]
    header = "| " + " | ".join(str(c) for c in show_preview_cols) + " |"
    sep = "|" + "|".join(["---"] * len(show_preview_cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in preview_df[show_preview_cols].iterrows():
        vals = [str(row[c])[:20] for c in show_preview_cols]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df.columns) > 8:
        lines.append(f"\n> 仅显示前 8 列，共 {len(df.columns)} 列")
    lines.append("")

    # 写入报告
    report_content = "\n".join(lines)
    conv_id = os.environ.get('CONV_ID')
    writer = ReportWriter(conv_id=conv_id)
    result = writer.write_markdown(report_content, output_name)

    # Emit state update
    missing_rates = {}
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > 0:
            missing_rates[col] = round(rate, 4)
    writer.emit_state_update({
        "dataset": {
            "name": file_name,
            "path": data_path,
            "rows": n_rows,
            "cols": n_cols,
        },
        "profiling": {
            "type_counts": type_counts,
            "constant_cols": constant_cols,
            "empty_cols": empty_cols,
            "missing_rates": missing_rates,
        },
    })

    print(f"报告已生成: {result['path']}")
    return result['path']

def main():
    parser = argparse.ArgumentParser(description='数据洞察分析器')
    parser.add_argument('--data_path', required=True, help='数据文件路径')
    parser.add_argument('--output', default=None, help='报告输出路径')

    args = parser.parse_args()
    output_name = None
    if args.output:
        output_name = os.path.splitext(os.path.basename(args.output))[0]
    generate_profiling_report(args.data_path, output_name)

if __name__ == '__main__':
    main()
