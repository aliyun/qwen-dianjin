#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主实验循环主入口脚本
Try → Measure → Keep/Discard → Repeat

用法:
    python run_experiment.py --data_path /path/to/data.parquet \
                             --target y_label \
                             --exploration "mob3相关特征" \
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
        description='自主实验循环 - 在指定方向内自动探索最优特征组合'
    )
    
    # 必选参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据文件路径 (parquet/csv)')
    parser.add_argument('--target', type=str, required=True,
                        help='目标变量列名 (0/1 二分类)')
    parser.add_argument('--exploration', type=str, required=True,
                        help='探索方向描述，如 "mob3相关特征"')
    
    # 可选参数
    parser.add_argument('--max_rounds', type=int, default=5,
                        help='最大实验轮数 (默认: 5)')
    parser.add_argument('--metric', type=str, default='ks',
                        choices=['auc', 'ks'],
                        help='优化指标 (默认: ks)')
    parser.add_argument('--direction', type=str, default='maximize',
                        choices=['maximize', 'minimize'],
                        help='优化方向 (默认: maximize)')
    parser.add_argument('--significance', type=float, default=2.0,
                        help='显著性阈值，MAD倍数 (默认: 2.0)')
    parser.add_argument('--baseline_features', type=str, default=None,
                        help='基线特征列表，JSON格式')
    
    # 数据切分参数（统一三段式 train / val / oot）
    parser.add_argument('--time_col', type=str, default='busi_dt',
                        help='时间列名 (默认: busi_dt)')
    parser.add_argument('--train_filter', type=str, default=None,
                        help='训练集筛选条件 (pandas query)')
    parser.add_argument('--val_filter', type=str, default=None,
                        help='验证集 val 筛选条件 (pandas query)')
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
    parser.add_argument('--session_file', type=str, default=None,
                        help='会话文档路径')
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
        data 为 train/val/oot 顺序拼接后的 DataFrame，indices 为对应行号。
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
    baseline = result['baseline']
    final = result['final']
    results = result.get('results', [])
    n_rounds = len(results)
    kept_count = sum(1 for r in results if r['decision'] == 'KEEP')
    discarded_count = sum(1 for r in results if r['decision'] != 'KEEP')
    improvement = result['improvement']
    improvement_pct = result['improvement_pct']
    
    report = f"""# 自主实验循环报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 实验配置

| 配置项 | 值 |
|--------|-----|
| 数据文件 | `{args.data_path}` |
| 目标变量 | `{args.target}` |
| 探索方向 | {args.exploration} |
| 优化指标 | {args.metric.upper()} ({args.direction}) |
| 实验轮数 | {n_rounds} |
| 显著性阈值 | {args.significance}× MAD |
| 基线特征数 | {len(baseline['features'])} |

## 2. 实验结果

### 2.1 总体改进

| 指标 | 基线 | 最终 | 改进 |
|------|------|------|------|
| {args.metric.upper()} | {baseline['metric']:.4f} | {final['metric']:.4f} | {improvement:+.4f} ({improvement_pct:+.2f}%) |
| 特征数 | {len(baseline['features'])} | {len(final['features'])} | {len(final['features']) - len(baseline['features']):+d} |

"""
    # 总体改进评价
    if improvement_pct > 5:
        report += f"> **实验效果显著**：经过 {n_rounds} 轮实验，{args.metric.upper()} 提升 {improvement_pct:+.2f}%，探索方向「{args.exploration}」对模型有明显增益。\n\n"
    elif improvement_pct > 1:
        report += f"> **实验有一定效果**：经过 {n_rounds} 轮实验，{args.metric.upper()} 提升 {improvement_pct:+.2f}%，探索方向有正向贡献但幅度有限。\n\n"
    elif improvement_pct > 0:
        report += f"> **实验效果微弱**：{args.metric.upper()} 仅提升 {improvement_pct:+.2f}%，探索方向的增量价值较低。\n\n"
    else:
        report += f"> **实验未带来改进**：{args.metric.upper()} 变化 {improvement_pct:+.2f}%，探索方向「{args.exploration}」对当前模型无正向贡献。\n\n"

    report += f"""### 2.2 实验详情

| Round | 假设 | {args.metric.upper()} | 改进 | 置信度 | 决策 |
|-------|------|-------|------|--------|------|
"""
    
    metric_values = []
    for r in results:
        decision_icon = "✅ 保留" if r['decision'] == 'KEEP' else "❌ 丢弃"
        conf = r['confidence']
        conf_str = f"{conf:.1f}× MAD" if conf else "-"
        metric_val = r['metrics'].get('val', {}).get(args.metric, 0)
        metric_values.append(metric_val)
        report += f"| {r['round']} | {r['hypothesis']} | {metric_val:.4f} | {r['improvement']:+.4f} | {conf_str} | {decision_icon} |\n"
    
    # 实验趋势分析
    if len(metric_values) >= 2:
        report += "\n### 2.3 实验趋势分析\n\n"
        # 检查趋势方向
        improving_rounds = sum(1 for i in range(1, len(metric_values)) if metric_values[i] > metric_values[i-1])
        declining_rounds = sum(1 for i in range(1, len(metric_values)) if metric_values[i] < metric_values[i-1])
        best_round = metric_values.index(max(metric_values)) + 1
        worst_round = metric_values.index(min(metric_values)) + 1
        amplitude = max(metric_values) - min(metric_values)
        
        report += f"- **趋势概况**: {n_rounds} 轮实验中，{improving_rounds} 轮上升、{declining_rounds} 轮下降\n"
        report += f"- **最佳轮次**: Round {best_round}（{args.metric.upper()} = {max(metric_values):.4f}）\n"
        report += f"- **最差轮次**: Round {worst_round}（{args.metric.upper()} = {min(metric_values):.4f}）\n"
        report += f"- **波动幅度**: {amplitude:.4f}\n"
        report += f"- **采纳率**: {kept_count}/{n_rounds} = {kept_count/max(n_rounds,1)*100:.0f}%\n"
        
        if kept_count == 0:
            report += "\n> 所有实验轮次均被丢弃，表明该探索方向的特征对模型无显著贡献。\n"
        elif kept_count == n_rounds:
            report += "\n> 所有实验轮次均被保留，探索方向非常有效，每轮都带来了显著提升。\n"
        elif amplitude < 0.005:
            report += "\n> 各轮指标波动极小，模型已趋于饱和，继续同方向探索的边际收益有限。\n"
    
    report += f"\n### {'2.4' if len(metric_values) >= 2 else '2.3'} 特征变化\n\n"
    report += "**保留的新特征:**\n\n"
    if result['kept_features']:
        for f in result['kept_features']:
            report += f"- ✅ {f}\n"
    else:
        report += "- （无新增特征）\n"
    
    report += "\n**丢弃的特征:**\n\n"
    if result['discarded_features']:
        for f in result['discarded_features']:
            report += f"- ❌ {f}\n"
    else:
        report += "- （无丢弃特征）\n"
    
    # 最终特征列表
    final_section = '2.5' if len(metric_values) >= 2 else '2.4'
    report += f"\n### {final_section} 最终特征列表\n\n"
    report += f"最终模型使用 **{len(final['features'])}** 个特征：\n\n"
    for i, feat in enumerate(final['features'], 1):
        is_new = "🆕" if feat in result.get('kept_features', []) else ""
        report += f"{i}. `{feat}` {is_new}\n"
    
    report += "\n## 3. 统计摘要\n\n"
    stats = result.get('stats', {})
    mad_val = stats.get('mad', 0)
    median_val = stats.get('median', 0)
    mean_val = stats.get('mean', 0)
    std_val = stats.get('std', 0)
    
    report += f"""| 统计项 | 值 | 说明 |
|--------|-----|------|
| 观测数 | {stats.get('n_samples', 0)} | 实验中的评估次数 |
| MAD (噪声估计) | {mad_val:.6f} | 衡量指标波动的稳健估计量 |
| 中位数 | {median_val:.4f} | 各轮{args.metric.upper()}的中位水平 |
| 均值 | {mean_val:.4f} | 各轮{args.metric.upper()}的平均水平 |
| 标准差 | {std_val:.6f} | 指标波动程度 |
"""
    
    # 统计解读
    report += "\n**解读**："
    if mad_val < 0.001:
        report += "MAD 极小，指标波动很低，实验结果稳定可信。"
    elif mad_val < 0.005:
        report += "MAD 较小，指标波动在正常范围内，实验结果基本可信。"
    else:
        report += f"MAD = {mad_val:.6f}，指标存在一定波动，需结合置信度谨慎判断每轮结果。"
    
    if abs(mean_val - median_val) > 0.005:
        report += f" 均值与中位数差异较大（{abs(mean_val - median_val):.4f}），说明存在异常轮次拉偏均值。"
    report += "\n"
    
    report += "\n## 4. 下一步建议\n\n"
    suggestions = result.get('suggestions', [])
    if suggestions:
        for s in suggestions:
            report += f"- {s}\n"
    else:
        # 根据实验结果生成针对性建议
        if improvement_pct > 5:
            report += f"- 探索方向「{args.exploration}」效果显著，建议在此方向上继续深入，尝试更多衍生特征\n"
            report += "- 可使用 xgb-tuning 对新特征组合进行参数调优以进一步提升\n"
            report += "- 建议运行 model-explanation 分析新增特征的具体贡献机制\n"
        elif improvement_pct > 0:
            report += f"- 探索方向「{args.exploration}」有一定效果，但提升幅度有限\n"
            report += "- 可考虑扩展到相关联的其他时间窗口（如 mob6、mob12 等）\n"
            report += "- 建议尝试其他探索方向，如行为频次、金额分布等维度\n"
        else:
            report += f"- 探索方向「{args.exploration}」对当前模型无正向贡献，建议暂时放弃此方向\n"
            report += "- 可能原因：该类特征与已有特征高度相关，信息已被充分捕获\n"
            report += "- 建议尝试差异化更大的方向，如跨域特征或交互特征\n"
        
        if kept_count == 0 and n_rounds >= 3:
            report += "- ⚠️ 多轮实验均未通过显著性检验，建议检查探索假设是否合理\n"
    
    # 实验结论
    report += "\n## 5. 实验结论\n\n"
    report += f"本次实验围绕「{args.exploration}」方向共进行 {n_rounds} 轮自主探索，"
    report += f"其中 {kept_count} 轮被采纳、{discarded_count} 轮被丢弃。"
    if improvement_pct > 0:
        report += f" 最终 {args.metric.upper()} 从基线 {baseline['metric']:.4f} 提升至 {final['metric']:.4f}（{improvement_pct:+.2f}%），"
        report += f"净增 {len(result['kept_features'])} 个有效特征。"
        if improvement_pct > 5:
            report += " 实验效果显著，建议将结果纳入正式模型。"
        else:
            report += " 改进幅度有限，是否采纳需结合业务综合判断。"
    else:
        report += f" 最终 {args.metric.upper()} 为 {final['metric']:.4f}，未实现改进。"
        report += " 建议转向其他探索方向或重新审视特征工程策略。"
    report += "\n"
    
    report += "\n---\n\n*此报告由 auto-experiment 技能自动生成*\n"
    
    # 最终模型评估可视化（KS/ROC 已由 shared/report_artifact 统一渲染）
    visualization = result.get('visualization', {})
    if visualization:
        ks_md = visualization.get('ks_curve_md', '')
        roc_md = visualization.get('roc_curve_md', '')
        if ks_md or roc_md:
            report += "\n## 6. 最终模型评估可视化\n\n"
        if ks_md:
            report += ks_md + "\n"
        if roc_md:
            report += roc_md + "\n"

    return report

def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 自主实验循环 (Auto Experiment Loop)")
    print("=" * 60)
    
    # 1. 加载 & 切分数据（三段式 train / val / oot）
    print("\n📥 加载与切分数据...")
    df, train_idx, val_idx, oot_idx, split_meta = prepare_data_and_indices(args)

    # 输出一个切分小节（Markdown）供上层记录
    split_section = render_split_section(split_meta)
    if split_section:
        print("\n" + split_section)
    
    # 3. 解析基线特征
    baseline_features = None
    if args.baseline_features:
        try:
            baseline_features = json.loads(args.baseline_features)
        except json.JSONDecodeError:
            # 尝试作为逗号分隔的列表
            baseline_features = [f.strip() for f in args.baseline_features.split(',')]
    
    # 4. 初始化会话追踪器
    from session_tracker import SessionTracker
    
    session_file = args.session_file
    if not session_file:
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        session_file = str(output_dir / "modeling_session.json")
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    session = SessionTracker(session_file)
    
    # 5. 运行实验循环
    from experiment_loop import AutoExperimentLoop
    
    loop = AutoExperimentLoop(
        data=df,
        target=args.target,
        train_idx=train_idx,
        val_idx=val_idx,
        oot_idx=oot_idx if len(oot_idx) > 0 else None,
        metric=args.metric,
        direction=args.direction,
        significance_threshold=args.significance,
        session_tracker=session,
        verbose=True,
    )
    
    result = loop.run_loop(
        exploration=args.exploration,
        max_rounds=args.max_rounds,
        baseline_features=baseline_features,
    )
    
    # 6. 生成报告
    print("\n📝 生成报告...")
    report = generate_report(result, args)

    # 输出可视化部分到 stdout（供前端解析 artifact）— 复用 shared/report_artifact 渲染结果
    visualization = result.get('visualization', {})
    if visualization:
        ks_md = visualization.get('ks_curve_md', '')
        roc_md = visualization.get('roc_curve_md', '')
        if ks_md:
            print("\n" + ks_md)
        if roc_md:
            print("\n" + roc_md)
    
    # 使用 ResultEmitter 落盘
    report_output_arg = args.report_output or args.output
    if report_output_arg:
        output_name = os.path.splitext(os.path.basename(report_output_arg))[0]
    else:
        output_name = 'auto_experiment_report'
    
    emitter = ResultEmitter(skill="auto-experiment", output_dir=output_dir)
    report_path = output_dir / f"{output_name}.md"
    report_path.write_text(report, encoding='utf-8')
    emitter.add_report(report_path)
    emitter.set_summary(f"实验完成: 改进 {result['improvement']:+.4f} ({result['improvement_pct']:+.2f}%)")
    emitter.update_metrics({"baseline": result['baseline']['metric'], "final": result['final']['metric']})
    emitter.write(output_dir / "result.json")
    
    print(f"✓ 报告已保存: {report_path}")
    
    # 输出会话文档路径
    print(f"✓ 会话文档: {session_file}")
    
    # 返回结果摘要
    print("\n" + "=" * 60)
    print("🏁 实验完成")
    print("=" * 60)
    print(f"   基线 {args.metric.upper()}: {result['baseline']['metric']:.4f}")
    print(f"   最终 {args.metric.upper()}: {result['final']['metric']:.4f}")
    print(f"   总改进: {result['improvement']:+.4f} ({result['improvement_pct']:+.2f}%)")
    print(f"   保留特征: {len(result['kept_features'])} 个")
    print(f"   丢弃特征: {len(result['discarded_features'])} 个")
    
    return result

if __name__ == '__main__':
    main()
