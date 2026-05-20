#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-adjudication-review

校验理赔案件复审所需的输入数据完整性。

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

    # 医审结果校验
    medical_review = data.get("medical_review_result")
    if not medical_review:
        errors.append("缺少必填字段: medical_review_result（医审结果）")

    # 理算结果校验
    adjustment_result = data.get("adjustment_result")
    if not adjustment_result:
        errors.append("缺少必填字段: adjustment_result（理算结果）")

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
