#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-underwriting-medical-assessment

用法：
    python scripts/validate_input.py --input <input_file_or_data>

退出码：
    0 - 校验通过
    1 - 校验失败（具体错误输出到 stderr）
    2 - 参数错误
"""
import sys
import json
import argparse


def validate(data: dict) -> list[str]:
    """校验输入数据，返回错误列表。空列表表示通过。"""
    errors = []

    # 体检报告内容
    report = data.get("体检报告内容")
    if not report:
        errors.append("缺少必填字段：体检报告内容")

    # 申请险种
    insurance_type = data.get("申请险种")
    if not insurance_type:
        errors.append("缺少必填字段：申请险种")

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
