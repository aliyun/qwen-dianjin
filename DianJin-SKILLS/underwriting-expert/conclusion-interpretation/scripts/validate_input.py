#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-underwriting-conclusion-interpretation

校验核保结果通知书生成所需的输入数据完整性和合规性。

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
    # 核保结论类型校验
    valid_conclusions = ["标准体", "加费", "除外", "延期", "拒保"]
    conclusion = data.get("conclusion_type")
    if not conclusion:
        errors.append("缺少必填字段: conclusion_type（核保结论类型）")
    elif conclusion not in valid_conclusions:
        errors.append(f"核保结论类型无效: {conclusion}，有效值为: {valid_conclusions}")

    # 被保险人信息校验
    if not data.get("insured_name"):
        errors.append("缺少必填字段: insured_name（被保险人姓名）")
    if not data.get("insured_id_number"):
        errors.append("缺少必填字段: insured_id_number（被保险人证件号）")

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
