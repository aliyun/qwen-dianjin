#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-underwriting-follow-up-outreach

校验回访跟进所需的输入数据完整性。

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
    # 客户信息校验
    customer_id = data.get("customer_id")
    if not customer_id:
        errors.append("缺少必填字段: customer_id（客户标识）")

    # 联系方式校验
    contact = data.get("contact_info")
    if not contact:
        errors.append("缺少必填字段: contact_info（联系方式）")

    # 回访事由校验
    reason = data.get("visit_reason")
    if not reason:
        errors.append("缺少必填字段: visit_reason（回访事由）")
    elif reason not in ["新单送达", "核保结论", "常规回访", "续期提醒"]:
        errors.append(f"不支持的回访事由: {reason}")

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
