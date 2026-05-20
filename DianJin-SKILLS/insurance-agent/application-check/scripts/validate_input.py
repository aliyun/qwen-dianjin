#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-application-check"""
import sys, json, argparse

def validate(data: dict) -> list[str]:
    errors = []
    if "insurance_type" not in data or not data["insurance_type"]:
        errors.append("缺少必填字段: insurance_type（险种类型）")
    else:
        valid = ["健康险", "寿险", "意外险", "年金险", "车险"]
        if data["insurance_type"] not in valid:
            errors.append(f"险种类型无效: {data['insurance_type']}")
    if "applicant_id" in data and data["applicant_id"]:
        if len(data["applicant_id"]) != 18:
            errors.append(f"身份证号长度异常: {len(data['applicant_id'])}，应为18位")
    if "beneficiaries" in data and isinstance(data["beneficiaries"], list):
        total_share = sum(b.get("share", 0) for b in data["beneficiaries"])
        if len(data["beneficiaries"]) > 1 and total_share != 100:
            errors.append(f"受益人份额合计为{total_share}%，必须等于100%")
    return errors

def main():
    parser = argparse.ArgumentParser(description="投保前核查 - 输入数据校验")
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
