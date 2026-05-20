#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-objection-handler

校验代理人异议处理所需的输入数据完整性。

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
    # 异议类型校验
    objection_type = data.get("objection_type")
    if not objection_type:
        errors.append("缺少必填字段: objection_type（异议类型）")

    # 客户基本信息校验
    client_info = data.get("client_info")
    if not client_info:
        errors.append("缺少必填字段: client_info（客户基本信息）")

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
