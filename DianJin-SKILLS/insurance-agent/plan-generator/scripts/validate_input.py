#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-plan-generator

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

    # 客户姓名
    if not data.get("客户姓名"):
        errors.append("缺少必填字段：客户姓名")

    # 年龄
    age = data.get("年龄")
    if age is None:
        errors.append("缺少必填字段：年龄")
    elif not isinstance(age, (int, float)) or age < 0:
        errors.append("年龄必须为非负数")

    # 产品方案
    plan = data.get("产品方案")
    if not plan:
        errors.append("缺少必填字段：产品方案")

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
