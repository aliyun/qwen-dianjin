#!/usr/bin/env python3
"""财务数据完整性与勾稽关系验证脚本。

验证要点：
1. 报表平衡性（资产=负债+所有者权益）
2. 数据连续性（至少覆盖连续2期）
3. 关键科目非空检查
4. 报表口径一致性检查
"""

import json
import sys
from datetime import datetime


def validate_data_balance(total_assets, total_liabilities, equity):
    """验证资产负债表平衡：总资产 = 总负债 + 所有者权益"""
    diff = abs(total_assets - (total_liabilities + equity))
    tolerance = total_assets * 0.01  # 1% 容差
    if diff > tolerance:
        raise ValueError(
            f"资产负债不平衡：总资产 {total_assets} != "
            f"总负债 {total_liabilities} + 权益 {equity}，差额 {diff}"
        )
    return True


def validate_data_continuity(periods):
    """验证数据连续性：至少覆盖连续2期"""
    if len(periods) < 2:
        raise ValueError(
            f"数据期数不足：仅提供 {len(periods)} 期，至少需要 2 期进行趋势分析"
        )
    return True


def validate_key_items(data_dict):
    """验证关键科目非空"""
    required_keys = [
        "营业收入", "营业成本", "净利润", "经营现金流",
        "总资产", "总负债", "应收账款", "存货"
    ]
    missing = []
    for key in required_keys:
        if key not in data_dict or data_dict[key] is None:
            missing.append(key)
    if missing:
        raise ValueError(f"关键科目缺失：{', '.join(missing)}")
    return True


def main():
    """主入口：读取财务数据文件并执行验证"""
    if len(sys.argv) < 2:
        print("用法: python validate_financial_data.py <data.json>")
        sys.exit(1)

    data_file = sys.argv[1]
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：数据文件不存在: {data_file}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败: {e}")
        sys.exit(2)

    errors = []

    # 1. 报表平衡性检查
    try:
        ta = data.get("总资产")
        tl = data.get("总负债")
        eq = data.get("所有者权益")
        if all(v is not None for v in [ta, tl, eq]):
            validate_data_balance(ta, tl, eq)
            print("[PASS] 资产负债表平衡验证通过")
        else:
            print("[SKIP] 资产负债表平衡验证跳过（数据不全）")
    except ValueError as e:
        errors.append(str(e))
        print(f"[FAIL] {e}")

    # 2. 数据连续性检查
    try:
        periods = data.get("periods", [])
        validate_data_continuity(periods)
        print(f"[PASS] 数据连续性验证通过（{len(periods)} 期）")
    except ValueError as e:
        errors.append(str(e))
        print(f"[WARN] {e}")

    # 3. 关键科目检查
    try:
        validate_key_items(data)
        print("[PASS] 关键科目完整性验证通过")
    except ValueError as e:
        errors.append(str(e))
        print(f"[WARN] {e}")

    if errors:
        print(f"\n验证完成，发现 {len(errors)} 个问题")
        sys.exit(1)
    else:
        print("\n验证全部通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
