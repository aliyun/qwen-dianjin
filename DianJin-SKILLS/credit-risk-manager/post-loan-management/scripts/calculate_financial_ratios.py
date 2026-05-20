#!/usr/bin/env python3
"""
财务指标计算与行业基准对比脚本

Usage:
    python scripts/calculate_financial_ratios.py --input <financial_data.json> --benchmarks <industry-benchmarks.md>

Exit codes:
    0: 计算完成，无预警
    1: 计算完成，发现预警指标
"""

import argparse
import json
import sys


def calculate_ratios(financial_data: dict) -> dict:
    """计算关键财务指标。"""
    ratios = {}
    bs = financial_data.get("balance_sheet", {})
    is_ = financial_data.get("income_statement", {})
    cfs = financial_data.get("cash_flow_statement", {})

    # 流动比率 = 流动资产 / 流动负债
    if bs.get("current_assets") and bs.get("current_liabilities") and bs["current_liabilities"] > 0:
        ratios["current_ratio"] = round(bs["current_assets"] / bs["current_liabilities"], 2)

    # 速动比率 = (流动资产 - 存货) / 流动负债
    if bs.get("current_assets") and bs.get("inventory") and bs.get("current_liabilities") and bs["current_liabilities"] > 0:
        ratios["quick_ratio"] = round((bs["current_assets"] - bs["inventory"]) / bs["current_liabilities"], 2)

    # 资产负债率 = 总负债 / 总资产
    if bs.get("total_liabilities") and bs.get("total_assets") and bs["total_assets"] > 0:
        ratios["debt_ratio"] = round(bs["total_liabilities"] / bs["total_assets"], 4)

    # 利息保障倍数 = EBIT / 利息支出
    ebit = is_.get("operating_profit", 0)
    interest = is_.get("interest_expense", 0)
    if ebit and interest and interest > 0:
        ratios["interest_coverage"] = round(ebit / interest, 2)

    # 经营现金流/总负债
    if cfs.get("operating_cash_flow") and bs.get("total_liabilities") and bs["total_liabilities"] > 0:
        ratios["ocf_to_debt"] = round(cfs["operating_cash_flow"] / bs["total_liabilities"], 4)

    return ratios


def compare_with_benchmarks(ratios: dict, benchmarks: dict) -> list[str]:
    """将计算结果与行业基准对比。"""
    warnings = []
    # 流动比率
    if "current_ratio" in ratios:
        if ratios["current_ratio"] < 1.0:
            warnings.append(f"流动比率 {ratios['current_ratio']} < 预警阈值 1.0")
        elif ratios["current_ratio"] < 1.5:
            warnings.append(f"流动比率 {ratios['current_ratio']} 处于关注区间 (1.0-1.5)")

    # 速动比率
    if "quick_ratio" in ratios:
        if ratios["quick_ratio"] < 0.7:
            warnings.append(f"速动比率 {ratios['quick_ratio']} < 预警阈值 0.7")
        elif ratios["quick_ratio"] < 1.0:
            warnings.append(f"速动比率 {ratios['quick_ratio']} 处于关注区间 (0.7-1.0)")

    # 资产负债率
    if "debt_ratio" in ratios:
        if ratios["debt_ratio"] > 0.70:
            warnings.append(f"资产负债率 {ratios['debt_ratio']:.1%} > 预警阈值 70%")
        elif ratios["debt_ratio"] > 0.60:
            warnings.append(f"资产负债率 {ratios['debt_ratio']:.1%} 处于关注区间 (60%-70%)")

    # 利息保障倍数
    if "interest_coverage" in ratios:
        if ratios["interest_coverage"] < 1.5:
            warnings.append(f"利息保障倍数 {ratios['interest_coverage']} < 预警阈值 1.5")
        elif ratios["interest_coverage"] < 3:
            warnings.append(f"利息保障倍数 {ratios['interest_coverage']} 处于关注区间 (1.5-3)")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="财务指标计算与行业基准对比")
    parser.add_argument("--input", required=True, help="财务数据 JSON 文件路径")
    parser.add_argument("--benchmarks", required=False, help="行业基准文件路径")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            financial_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    ratios = calculate_ratios(financial_data)
    print("计算指标:")
    for k, v in ratios.items():
        print(f"  {k}: {v}")

    # 简化基准对比（实际应从 benchmarks 文件读取）
    warnings = compare_with_benchmarks(ratios, {})
    if warnings:
        print(f"\n发现 {len(warnings)} 项预警:")
        for w in warnings:
            print(f"  - {w}")
        sys.exit(1)
    else:
        print("\n所有指标在正常区间")
        sys.exit(0)


if __name__ == "__main__":
    main()
