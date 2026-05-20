#!/usr/bin/env python3
"""
资金用途合规性检查脚本
对照 fund-usage-policy.md 的禁止性清单和回流规则进行自动化违规检测。

Usage:
    python scripts/check_fund_usage.py --input <fund_flow.json> --rules <rules.md>

Exit codes:
    0: 未发现违规
    1: 发现违规（具体违规事项在 stdout 中输出）
"""

import argparse
import json
import sys


# 禁止性领域关键词
FORBIDDEN_KEYWORDS = [
    "房地产", "土地储备", "股票", "期货", "衍生品",
    "理财", "P2P", "民间借贷", "非法集资",
]

# 回流识别参数
FLOWBACK_WINDOW_7D = 7       # 7 日内
FLOWBACK_WINDOW_30D = 30     # 30 日内
FLOWBACK_THRESHOLD = 0.30    # 超过贷款金额 30%


def check_forbidden_destinations(fund_flow: dict) -> list[str]:
    """检查资金是否流入禁止性领域。"""
    violations = []
    transactions = fund_flow.get("transactions", [])
    for txn in transactions:
        recipient = txn.get("recipient", "")
        purpose = txn.get("purpose", "")
        combined = f"{recipient} {purpose}"
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword in combined:
                violations.append(
                    f"疑似违规: 交易对手='{recipient}', 用途='{purpose}', "
                    f"命中禁止词='{keyword}', 金额={txn.get('amount', 'N/A')}万元"
                )
                break
    return violations


def check_flowback(fund_flow: dict) -> list[str]:
    """检查资金回流。"""
    violations = []
    loan_amount = fund_flow.get("loan_amount", 0)
    if loan_amount <= 0:
        return violations
    transactions = fund_flow.get("transactions", [])
    # 简单回流检测：检查是否有资金转回借款人或其关联方
    borrower = fund_flow.get("borrower", "")
    related_parties = fund_flow.get("related_parties", [])
    threshold = loan_amount * FLOWBACK_THRESHOLD
    flowback_total = 0
    for txn in transactions:
        recipient = txn.get("recipient", "")
        if recipient == borrower or recipient in related_parties:
            flowback_total += txn.get("amount", 0)
    if flowback_total > threshold:
        violations.append(
            f"资金回流: 累计回流金额={flowback_total}万元, "
            f"贷款金额={loan_amount}万元, 回流比例={flowback_total/loan_amount:.1%}, "
            f"超过阈值={FLOWBACK_THRESHOLD:.0%}"
        )
    return violations


def main():
    parser = argparse.ArgumentParser(description="资金用途合规性检查")
    parser.add_argument("--input", required=True, help="资金流向 JSON 文件路径")
    parser.add_argument("--rules", required=False, help="规则文件路径（可选）")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            fund_flow = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}", file=sys.stderr)
        sys.exit(1)

    all_violations = []
    all_violations.extend(check_forbidden_destinations(fund_flow))
    all_violations.extend(check_flowback(fund_flow))

    if all_violations:
        print(f"发现 {len(all_violations)} 项疑似违规:")
        for v in all_violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("未发现资金用途违规")
        sys.exit(0)


if __name__ == "__main__":
    main()
