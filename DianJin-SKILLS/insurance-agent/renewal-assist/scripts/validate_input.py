#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-renewal-assist"""
import sys, json, argparse

def validate(data: dict) -> list[str]:
    errors = []
    if "customer_name" not in data or not data["customer_name"]:
        errors.append("缺少必填字段: customer_name（客户姓名）")
    if "product_name" not in data or not data["product_name"]:
        errors.append("缺少必填字段: product_name（产品名称）")
    if "renewal_date" not in data or not data["renewal_date"]:
        errors.append("缺少必填字段: renewal_date（续期日期）")
    if "premium_amount" in data and isinstance(data["premium_amount"], (int, float)):
        if data["premium_amount"] <= 0:
            errors.append(f"保费金额必须为正数: {data['premium_amount']}")
    if "policy_status" in data and data["policy_status"]:
        valid = ["正常", "欠费", "失效"]
        if data["policy_status"] not in valid:
            errors.append(f"保单状态无效: {data['policy_status']}")
    return errors

def main():
    parser = argparse.ArgumentParser(description="续期维护 - 输入数据校验")
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
