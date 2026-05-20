#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分群自主探索主入口脚本
Try → Measure → Keep/Discard → Repeat

用法:
    python run_segment.py --data_path /path/to/data.parquet \
                          --target y_label \
                          --mode auto \
                          --max_rounds 5

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from data_utils import load_and_split_data, render_split_section  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402

# 脚本目录子模块
sys.path.insert(0, str(_SCRIPT_DIR))

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='分群自主探索 - 自动探索最优分群策略'
    )
    
    # 必选参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据文件路径 (parquet/csv)')
    parser.add_argument('--target', type=str, required=True,
                        help='目标变量列名 (0/1 二分类)')
    
    # 模式参数
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'manual'],
                        help='模式: auto(自主探索) / manual(指定策略)')
    parser.add_argument('--max_rounds', type=int, default=5,
                        help='自主探索最大轮数 (默认: 5)')
    
    # 分群策略参数
    parser.add_argument('--segment_rules', type=str, default=None,
                        help='规则分群，JSON格式')
    parser.add_argument('--segment_col', type=str, default=None,
                        help='直接指定分群列名')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='聚类分群数 (默认: 3)')
    parser.add_argument('--tree_depth', type=int, default=2,
                        help='决策树分群深度 (默认: 2)')
    parser.add_argument('--tree_features', type=str, default=None,
                        help='决策树使用的特征，逗号分隔')
    
    # 评估参数
    parser.add_argument('--min_segment_ratio', type=float, default=0.05,
                        help='最小分群占比 (默认: 0.05)')
    parser.add_argument('--metric', type=str, default='ks',
                        choices=['auc', 'ks'],
                        help='优化指标 (默认: ks)')
    parser.add_argument('--significance', type=float, default=2.0,
                        help='显著性阈值 (默认: 2.0)')
    
    # 数据切分参数（统一三段式 train / val / oot）
    parser.add_argument('--time_col', type=str, default='busi_dt',
                        help='时间列名 (默认: busi_dt)')
    parser.add_argument('--train_filter', type=str, default=None,
                        help='训练集筛选条件')
    parser.add_argument('--val_filter', type=str, default=None,
                        help='验证集 val 筛选条件')
    parser.add_argument('--test_filter', type=str, default=None,
                        help='[Deprecated] 等价 --val_filter，仅作向后兼容')
    parser.add_argument('--oot_filter', type=str, default=None,
                        help='OOT测试集筛选条件')
    parser.add_argument('--val_ratio', type=float, default=0.25,
                        help='val 在 train_full 内占比 (默认: 0.25, 对应 60/20/20)')
    parser.add_argument('--oot_ratio', type=float, default=0.20,
                        help='OOT 在全量内占比 (默认: 0.20)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='产物输出目录（默认 ./outputs/<timestamp>）')
    parser.add_argument('--report_output', type=str, default=None,
                        help='报告输出名称（可不含扩展名）')
    # 旧名兼容
    parser.add_argument('--output', type=str, default=None,
                        help='[Deprecated] 等价 --report_output')
    
    return parser.parse_args()

def prepare_data_and_indices(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """统一通过 shared.data_utils 加载与切分数据，再按段生成 indices。

    Returns:
        (data, train_idx, val_idx, oot_idx, split_meta)
    """
    val_filter_arg = args.val_filter
    if val_filter_arg is None and args.test_filter is not None:
        print("[警告] --test_filter 已废弃，请改用 --val_filter；本次将其作为 val_filter 使用")
        val_filter_arg = args.test_filter

    split_result = load_and_split_data(
        data_path=args.data_path,
        target=args.target,
        time_col=args.time_col,
        train_filter=args.train_filter,
        test_filter=val_filter_arg,
        oot_filter=args.oot_filter,
        oot_ratio=args.oot_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        return_features=False,
        return_meta=True,
    )

    train_df = split_result.train.reset_index(drop=True)
    val_df = split_result.val.reset_index(drop=True)
    oot_df = split_result.oot.reset_index(drop=True)

    n_train, n_val, n_oot = len(train_df), len(val_df), len(oot_df)
    data = pd.concat([train_df, val_df, oot_df], ignore_index=True)

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    oot_idx = np.arange(n_train + n_val, n_train + n_val + n_oot)

    print(f"✓ 数据切分: Train={n_train}, Val={n_val}, OOT={n_oot}")
    return data, train_idx, val_idx, oot_idx, split_result.meta or {}

def generate_report(result: dict, args: argparse.Namespace) -> str:
    """生成Markdown报告"""
    
    n_rounds = result.get('n_rounds', 0)
    baseline_metric = result.get('baseline_metric', 0)
    best_metric = result.get('best_metric', 0)
    total_improvement = result.get('total_improvement', 0)
    improvement_pct = total_improvement / max(baseline_metric, 0.0001) * 100
    best_strategy = result.get('best_strategy', '基线')
    results_list = result.get('results', [])
    kept_results = [r for r in results_list if r['decision'] == 'KEEP']
    
    report = f"""# 分群自主探索报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 探索配置

| 配置项 | 值 |
|--------|-----|
| 数据文件 | `{args.data_path}` |
| 目标变量 | `{args.target}` |
| 探索模式 | {args.mode} |
| 探索轮数 | {n_rounds} |
| 优化指标 | {args.metric.upper()} |
| 最小分群占比 | {args.min_segment_ratio:.1%} |
| 显著性阈值 | {args.significance}× MAD |

## 2. 探索结果

### 2.1 最优方案

| 指标 | 基线（单一模型） | 最优分群 | 改进 |
|------|------|------|------|
| {args.metric.upper()} | {baseline_metric:.4f} | {best_metric:.4f} | {total_improvement:+.4f} ({improvement_pct:+.2f}%) |

**最优策略**: {best_strategy}

"""
    # 最优方案评价
    if total_improvement > 0.01:
        report += f"> **分群建模效果显著**：相比单一模型，{args.metric.upper()} 提升 {improvement_pct:+.2f}%，分群策略「{best_strategy}」能有效捕获不同人群的差异化模式。\n\n"
    elif total_improvement > 0:
        report += f"> **分群建模有小幅提升**：{args.metric.upper()} 提升 {improvement_pct:+.2f}%，分群带来的增量有限，需权衡复杂度与收益。\n\n"
    else:
        report += f"> **分群建模未带来提升**：数据可能较为同质，单一模型已能充分捕获模式，建议沿用单一模型。\n\n"

    # 分群详情及规则解读
    best_info = result.get('best_segment_info', {})
    if best_info.get('n_segments', 1) > 1:
        report += f"**分群数**: {best_info.get('n_segments', 1)}\n\n"
        report += "**分群规则及解读**:\n\n"
        for seg_name, rule in best_info.get('segment_rules', {}).items():
            report += f"- **{seg_name}**: `{rule}`\n"
        report += "\n"
    
    report += f"""### 2.2 探索历史

| Round | 策略 | 分群数 | {args.metric.upper()} | 改进 | 决策 |
|-------|------|--------|-----|------|------|
"""
    
    for r in results_list:
        decision_icon = "✅ 采纳" if r['decision'] == 'KEEP' else "❌ 丢弃"
        _val_m = r.get(f'val_{args.metric}', r.get('val_auc', 0))
        report += f"| {r['round']} | {r['description']} | {r['n_segments']} | {_val_m:.4f} | {r['improvement']:+.4f} | {decision_icon} |\n"

    # 子模型详情 - 增强版
    if kept_results:
        last_kept = kept_results[-1]
        if last_kept.get('segment_details'):
            _mk = args.metric  # 'ks' or 'auc'
            _MK = _mk.upper()
            _train_key = f'train_{_mk}'
            _val_key = f'val_{_mk}'
            _oot_key = f'oot_{_mk}'
            # 回退: 如果 segment_details 中没有 metric-specific key，用 auc key
            _first_seg = last_kept['segment_details'][0] if last_kept['segment_details'] else {}
            if _val_key not in _first_seg:
                _train_key, _val_key, _oot_key = 'train_auc', 'val_auc', 'oot_auc'
                _MK = 'AUC'
            has_oot = any(seg.get(_oot_key) is not None for seg in last_kept['segment_details'])
        
            report += "\n### 2.3 子模型详情\n\n"
            if has_oot:
                report += f"| 分群 | 样本数 | 占比 | 正样本率 | Train {_MK} | Val {_MK} | OOT {_MK} | 过拟合Gap |\n"
                report += "|------|--------|------|----------|-----------|---------|---------|------------|\n"
            else:
                report += f"| 分群 | 样本数 | 占比 | 正样本率 | Train {_MK} | Val {_MK} | 过拟合Gap |\n"
                report += "|------|--------|------|----------|-----------|---------|------------|\n"
        
            for seg in last_kept['segment_details']:
                sample_count = seg['sample_count']
                sample_ratio = seg['sample_ratio']
                train_m = seg[_train_key]
                val_m = seg[_val_key]
                pos_rate = seg.get('pos_rate', '-')
                pos_rate_str = f"{pos_rate:.2%}" if isinstance(pos_rate, (int, float)) else str(pos_rate)
                gap = train_m - val_m
                gap_flag = " ⚠️" if gap > 0.05 else ""
        
                if has_oot:
                    oot_m = seg.get(_oot_key, '-')
                    oot_str = f"{oot_m:.4f}" if isinstance(oot_m, (int, float)) else str(oot_m)
                    report += f"| {seg['segment_name']} | {sample_count:,} | {sample_ratio:.1%} | {pos_rate_str} | {train_m:.4f} | {val_m:.4f} | {oot_str} | {gap:.4f}{gap_flag} |\n"
                else:
                    report += f"| {seg['segment_name']} | {sample_count:,} | {sample_ratio:.1%} | {pos_rate_str} | {train_m:.4f} | {val_m:.4f} | {gap:.4f}{gap_flag} |\n"
        
            # 分群画像对比
            report += "\n### 2.4 分群画像对比\n\n"
            segs = last_kept['segment_details']
            best_seg = max(segs, key=lambda s: s[_val_key])
            worst_seg = min(segs, key=lambda s: s[_val_key])
            largest_seg = max(segs, key=lambda s: s['sample_count'])
            smallest_seg = min(segs, key=lambda s: s['sample_count'])
        
            report += f"- **最佳分群**: {best_seg['segment_name']}（Val {_MK} = {best_seg[_val_key]:.4f}）\n"
            report += f"- **最差分群**: {worst_seg['segment_name']}（Val {_MK} = {worst_seg[_val_key]:.4f}）\n"
            report += f"- **{_MK} 极差**: {best_seg[_val_key] - worst_seg[_val_key]:.4f}\n"
            report += f"- **最大分群**: {largest_seg['segment_name']}（占比 {largest_seg['sample_ratio']:.1%}）\n"
            report += f"- **最小分群**: {smallest_seg['segment_name']}（占比 {smallest_seg['sample_ratio']:.1%}）\n"
        
            metric_spread = best_seg[_val_key] - worst_seg[_val_key]
            if metric_spread > 0.05:
                report += f"\n> 各分群间 {_MK} 差异较大（{metric_spread:.4f}），说明不同人群的风险模式存在显著差异，分群建模是合理的。\n"
            elif metric_spread > 0.02:
                report += f"\n> 各分群间 {_MK} 有一定差异（{metric_spread:.4f}），分群建模有其价值。\n"
            else:
                report += f"\n> 各分群间 {_MK} 差异较小（{metric_spread:.4f}），分群并未显著区分不同风险模式。\n"

            # 检查过拟合问题（train vs val）
            overfit_segs = [s for s in segs if s[_train_key] - s[_val_key] > 0.05]
            if overfit_segs:
                report += "\n> ⚠️ 以下分群存在过拟合风险（Train-Val Gap > 0.05）："
                report += "，".join(s['segment_name'] for s in overfit_segs)
                report += "，建议对这些分群调整正则化参数或增加样本量。\n"
    
    # 与单一模型对比
    report += "\n### 2.5 与单一模型对比\n\n"
    report += f"| 对比项 | 单一模型 | 分群模型 |\n"
    report += f"|--------|----------|----------|\n"
    report += f"| {args.metric.upper()} | {baseline_metric:.4f} | {best_metric:.4f} |\n"
    n_segs = best_info.get('n_segments', 1) if total_improvement > 0 else 1
    report += f"| 模型数量 | 1 | {n_segs} |\n"
    complexity = '中' if n_segs <= 3 else '高'
    interpretability = '中' if n_segs <= 3 else '弱'
    report += f"| 复杂度 | 低 | {complexity} |\n"
    report += f"| 可解释性 | 强 | {interpretability} |\n"
    
    if total_improvement > 0.01:
        report += f"\n> 分群模型在 {args.metric.upper()} 上显著优于单一模型，额外复杂度带来了明确的收益。\n"
    elif total_improvement > 0:
        report += f"\n> 分群模型略优于单一模型，但提升幅度有限，需权衡维护成本与收益。\n"
    else:
        report += f"\n> 单一模型已充分捕获数据模式，分群未带来额外收益，建议使用单一模型。\n"
    
    report += "\n## 3. 统计摘要\n\n"
    stats = result.get('stats', {})
    mad_val = stats.get('mad', 0)
    report += f"""| 统计项 | 值 | 说明 |
|--------|-----|------|
| 观测数 | {stats.get('n_samples', 0)} | 实验中的评估次数 |
| MAD (噪声估计) | {mad_val:.6f} | 指标波动的稳健估计量 |
| 中位数 | {stats.get('median', 0):.4f} | 各轮{args.metric.upper()}的中位水平 |
| 均值 | {stats.get('mean', 0):.4f} | 各轮{args.metric.upper()}的平均水平 |
"""

    report += "\n## 4. 建议\n\n"
    
    if total_improvement > 0.01:
        report += f"1. **采纳分群方案**：策略「{best_strategy}」带来 {improvement_pct:+.2f}% 提升，建议采用\n"
        report += "2. **子群特征优化**：对每个子群分别运行 auto-experiment，探索各子群的最优特征组合\n"
        report += "3. **参数调优**：对各子模型分别进行 xgb-tuning，可进一步挖掘潜力\n"
        if any(s.get(_train_key, s.get('train_auc', 0)) - s.get(_val_key, s.get('val_auc', 0)) > 0.05 for s in (kept_results[-1].get('segment_details', []) if kept_results else [])):
            report += "4. **过拟合处理**：部分分群存在过拟合风险，建议调整正则化参数或考虑样本增强\n"
    elif total_improvement > 0:
        report += f"1. **谨慎采用**：分群模型有小幅提升（{improvement_pct:+.2f}%），但增加了系统复杂度，需业务综合判断\n"
        report += "2. **探索更多策略**：尝试不同分群维度（如按金额、行为频次、客群类型）\n"
        report += "3. **调整分群粒度**：当前分群可能太粗或太细，可调整参数重试\n"
    else:
        report += "1. **使用单一模型**：分群未带来提升，建议沿用单一模型，降低系统复杂度\n"
        report += "2. **原因分析**：可能数据较为同质，或当前特征已充分覆盖各群体差异\n"
        report += "3. **替代方案**：可考虑将精力投入特征工程（auto-experiment）或参数调优（xgb-tuning）\n"
    
    # 分群探索结论
    report += "\n## 5. 探索结论\n\n"
    report += f"本次探索共进行 {n_rounds} 轮分群策略实验，其中 {len(kept_results)} 轮被采纳、{n_rounds - len(kept_results)} 轮被丢弃。"
    if total_improvement > 0:
        report += f" 最优方案「{best_strategy}」将 {args.metric.upper()} 从 {baseline_metric:.4f} 提升至 {best_metric:.4f}（{improvement_pct:+.2f}%）。"
        if total_improvement > 0.01:
            report += " 分群建模效果显著，建议纳入正式流程。"
        else:
            report += " 提升幅度有限，是否采纳需结合业务复杂度要求综合判断。"
    else:
        report += f" 分群建模未带来提升，基线单一模型 {args.metric.upper()} = {baseline_metric:.4f} 已是最优。"
    report += "\n"
    
    report += "\n---\n\n*此报告由 segment-modeling 技能自动生成*\n"
    
    return report

def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 分群自主探索 (Segment Auto Explorer)")
    print("=" * 60)
    
    # 1. 加载 & 切分数据（三段式 train / val / oot）
    print("\n📥 加载与切分数据...")
    df, train_idx, val_idx, oot_idx, split_meta = prepare_data_and_indices(args)

    split_section = render_split_section(split_meta)
    if split_section:
        print("\n" + split_section)
    
    # 3. 解析用户规则
    user_rules = None
    if args.segment_rules:
        try:
            user_rules = json.loads(args.segment_rules)
        except json.JSONDecodeError:
            print(f"⚠️ 无法解析分群规则: {args.segment_rules}")
    
    # 4. 运行探索
    from segment_explorer import SegmentAutoExplorer
    
    explorer = SegmentAutoExplorer(
        data=df,
        target=args.target,
        train_idx=train_idx,
        val_idx=val_idx,
        oot_idx=oot_idx if len(oot_idx) > 0 else None,
        metric=args.metric,
        significance_threshold=args.significance,
        min_segment_ratio=args.min_segment_ratio,
        verbose=True,
    )
    
    if args.mode == 'auto':
        result = explorer.run_exploration(
            max_rounds=args.max_rounds,
            user_rules=user_rules,
        )
    else:
        # 手动模式：只运行一种策略
        result = explorer.run_exploration(
            max_rounds=2,  # 基线 + 指定策略
            user_rules=user_rules,
        )
    
    # 5. 生成报告
    print("\n📝 生成报告...")
    report = generate_report(result, args)
    
    # 使用 ResultEmitter 落盘
    report_output_arg = args.report_output or args.output
    if report_output_arg:
        output_name = os.path.splitext(os.path.basename(report_output_arg))[0]
    else:
        output_name = 'segment_exploration_report'
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emitter = ResultEmitter(skill="segment-modeling", output_dir=output_dir)
    report_path = output_dir / f"{output_name}.md"
    report_path.write_text(report, encoding='utf-8')
    emitter.add_report(report_path)
    emitter.set_summary(f"分群探索完成: 最优策略={result.get('best_strategy', '基线')}")
    emitter.update_metrics({"baseline": result.get('baseline_metric', 0), "best": result.get('best_metric', 0)})
    emitter.write(output_dir / "result.json")
    
    print(f"✓ 报告已保存: {report_path}")
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("🏁 探索完成")
    print("=" * 60)
    print(f"   基线 {args.metric.upper()}: {result.get('baseline_metric', 0):.4f}")
    print(f"   最优 {args.metric.upper()}: {result.get('best_metric', 0):.4f}")
    print(f"   总改进: {result.get('total_improvement', 0):+.4f}")
    print(f"   最优策略: {result.get('best_strategy', '基线')}")
    print(f"   探索轮数: {result.get('n_rounds', 0)}")
    
    return result

if __name__ == '__main__':
    main()
