#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-adjustment

校验理赔理算所需的输入数据完整性。

用法：
    python scripts/validate_input.py --input <input_file_or_data>

退出码：
    0 - 校验通过
    1 - 校验失败
    2 - 参数错误
"""
import sys
import json
import argparse

def validate(data: dict) -> list[str]:
    errors = []
    # 案件号校验
    case_number = data.get("case_number")
    if not case_number:
        errors.append("缺少必填字段: case_number（案件号）")

    # 费用明细校验
    expense_details = data.get("expense_details")
    if not expense_details:
        errors.append("缺少必填字段: expense_details（费用明细）")
    elif isinstance(expense_details, list) and len(expense_details) == 0:
        errors.append("费用明细不能为空列表")

    # 保单参数校验
    policy_params = data.get("policy_params")
    if not policy_params:
        errors.append("缺少必填字段: policy_params（保单参数）")

    return errors

def main():
    parser = argparse.ArgumentParser(description="输入数据校验")
    parser.add_argument("--input", required=True, help="输入数据文件路径或 JSON 字符串")
    args = parser.parse_args()
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"输入数据加载失败: {e}", file=sys.stderr)
        sys.exit(2)
    errors = validate(data)
    if errors:
        for err in errors:
            print(f"❌ {err}", file=sys.stderr)
        print(json.dumps({"passed": False, "error_count": len(errors)}, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps({"passed": True, "error_count": 0}, ensure_ascii=False))
    sys.exit(0)

if __name__ == "__main__":
    main()
