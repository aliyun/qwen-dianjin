#!/usr/bin/env python3
"""
贷后数据验证脚本
验证输入数据的完整性和基本勾稽关系。

Usage:
    python scripts/validate_post_loan_data.py --customer <customer_data.json> --financial <financial_data.json>

Exit codes:
    0: 验证通过
    1: 验证失败（具体问题在 stderr 中输出）
"""

import argparse
import json
import sys


def validate_customer_data(customer_path: str) -> list[str]:
    """验证客户数据完整性。"""
    errors = []
    try:
        with open(customer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required_fields = [
            "enterprise_name",
            "credit_code",
            "credit_limit",
            "credit_balance",
            "credit_term_start",
            "credit_term_end",
            "guarantee_type",
            "risk_classification",
        ]
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"缺失必填字段: {field}")
        valid_risk_classes = ["正常", "关注", "次级", "可疑", "损失"]
        if "risk_classification" in data and data["risk_classification"] not in valid_risk_classes:
            errors.append(f"风险分类无效: {data['risk_classification']}，有效值: {valid_risk_classes}")
    except FileNotFoundError:
        errors.append(f"客户数据文件不存在: {customer_path}")
    except json.JSONDecodeError as e:
        errors.append(f"客户数据 JSON 解析失败: {e}")
    return errors


def validate_financial_data(financial_path: str) -> list[str]:
    """验证财务数据完整性和基本勾稽关系。"""
    errors = []
    try:
        with open(financial_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 检查必需表是否存在
        for statement in ["balance_sheet", "income_statement", "cash_flow_statement"]:
            if statement not in data:
                errors.append(f"缺失财务报表: {statement}")
        # 资产负债平衡检查
        if "balance_sheet" in data:
            bs = data["balance_sheet"]
            if all(k in bs for k in ["total_assets", "total_liabilities", "equity"]):
                diff = bs["total_assets"] - (bs["total_liabilities"] + bs["equity"])
                tolerance = abs(bs["total_assets"]) * 0.01  # 1% 容差
                if abs(diff) > tolerance:
                    errors.append(
                        f"资产负债表不平衡: 总资产={bs['total_assets']}, "
                        f"总负债+权益={bs['total_liabilities'] + bs['equity']}, "
                        f"差异={diff}"
                    )
        # 现金流量表期末余额与资产负债表货币资金匹配检查
        if "cash_flow_statement" in data and "balance_sheet" in data:
            cfs = data["cash_flow_statement"]
            bs = data["balance_sheet"]
            if "ending_cash" in cfs and "cash_and_equivalents" in bs:
                diff = abs(cfs["ending_cash"] - bs["cash_and_equivalents"])
                tolerance = abs(bs["cash_and_equivalents"]) * 0.01
                if diff > tolerance:
                    errors.append(
                        f"现金流量表期末余额与资产负债表货币资金不匹配: "
                        f"现金流期末={cfs['ending_cash']}, 货币资金={bs['cash_and_equivalents']}"
                    )
    except FileNotFoundError:
        errors.append(f"财务数据文件不存在: {financial_path}")
    except json.JSONDecodeError as e:
        errors.append(f"财务数据 JSON 解析失败: {e}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="贷后数据验证脚本")
    parser.add_argument("--customer", required=False, help="客户数据 JSON 文件路径")
    parser.add_argument("--financial", required=False, help="财务数据 JSON 文件路径")
    args = parser.parse_args()

    all_errors = []

    if args.customer:
        all_errors.extend(validate_customer_data(args.customer))
    if args.financial:
        all_errors.extend(validate_financial_data(args.financial))

    if all_errors:
        print("验证失败:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("验证通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
