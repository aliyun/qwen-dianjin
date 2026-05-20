#!/usr/bin/env python3
"""
风险分类评估脚本
根据五级分类政策评估当前风险分类是否准确。

Usage:
    python scripts/evaluate_risk_classification.py --input <assessment_data.json> --policy <five-classification-policy.md>

Exit codes:
    0: 分类准确，无需调整
    1: 分类需调整（建议在 stdout 中输出）
"""

import argparse
import json
import sys


OVERDUE_THRESHOLDS = {
    "关注": 90,
    "次级": 270,
    "可疑": 360,
    "损失": 365,
}

WARNING_SIGNAL_WEIGHTS = {
    "黄色": 1,
    "橙色": 2,
    "红色": 3,
}


def evaluate_classification(assessment_data: dict) -> dict:
    """评估风险分类。"""
    result = {
        "current_classification": assessment_data.get("current_classification", "未知"),
        "overdue_days": assessment_data.get("overdue_days", 0),
        "warning_signals": assessment_data.get("warning_signals", []),
        "repayment_ability": assessment_data.get("repayment_ability", "未知"),
        "collateral_adequacy": assessment_data.get("collateral_adequacy", "未知"),
    }

    # 计算预警信号加权总分
    total_weight = 0
    red_count = 0
    for signal in result["warning_signals"]:
        level = signal.get("level", "黄色")
        total_weight += WARNING_SIGNAL_WEIGHTS.get(level, 1)
        if level == "红色":
            red_count += 1

    result["signal_weight_total"] = total_weight
    result["red_signal_count"] = red_count

    # 基于逾期天数建议最低分类
    suggested_min = "正常"
    for classification, days in sorted(OVERDUE_THRESHOLDS.items(), key=lambda x: x[1]):
        if result["overdue_days"] > days:
            suggested_min = classification

    result["suggested_min_classification"] = suggested_min

    # 综合判断
    if red_count > 0:
        result["recommendation"] = "建议下调分类（存在红色预警信号）"
        result["needs_adjustment"] = True
    elif total_weight >= 4:
        result["recommendation"] = "建议关注，可能存在分类下调风险"
        result["needs_adjustment"] = total_weight >= 6
    elif result["overdue_days"] > 90 and result["current_classification"] == "正常":
        result["recommendation"] = "逾期超过 90 天，建议至少调整为关注类"
        result["needs_adjustment"] = True
    else:
        result["recommendation"] = "当前分类合理，无需调整"
        result["needs_adjustment"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="风险分类评估")
    parser.add_argument("--input", required=True, help="评估数据 JSON 文件路径")
    parser.add_argument("--policy", required=False, help="分类政策文件路径")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            assessment_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    result = evaluate_classification(assessment_data)

    print(f"当前分类: {result['current_classification']}")
    print(f"逾期天数: {result['overdue_days']} 天")
    print(f"预警信号数: {len(result['warning_signals'])} (加权总分: {result['signal_weight_total']})")
    print(f"建议最低分类: {result['suggested_min_classification']}")
    print(f"建议: {result['recommendation']}")

    if result["needs_adjustment"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
