#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-health-disclosure-guide"""
import sys, json, argparse

def validate(data: dict) -> list[str]:
    errors = []
    if "insurance_type" not in data or not data["insurance_type"]:
        errors.append("缺少必填字段: insurance_type（险种类型）")
    if "customer_health_info" not in data or not data["customer_health_info"]:
        errors.append("缺少必填字段: customer_health_info（客户健康信息）")
    return errors

def main():
    parser = argparse.ArgumentParser(description="健康告知引导 - 输入数据校验")
    parser.add_argument("--input", required=True, help="输入数据文件路径（JSON格式）")
    args = parser.parse_args()
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"输入文件不存在: {args.input}", file=sys.stderr); sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}", file=sys.stderr); sys.exit(2)
    errors = validate(data)
    if errors:
        for err in errors: print(f"❌ {err}", file=sys.stderr)
        print(json.dumps({"passed": False, "error_count": len(errors)}, ensure_ascii=False)); sys.exit(1)
    print(json.dumps({"passed": True, "error_count": 0}, ensure_ascii=False)); sys.exit(0)

if __name__ == "__main__":
    main()
