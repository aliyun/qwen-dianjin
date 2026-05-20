#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-case-registration

校验理赔案件登记所需的输入数据完整性。

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
    # 报案人信息校验
    reporter = data.get("reporter")
    if not reporter:
        errors.append("缺少必填字段: reporter（报案人信息）")
    elif isinstance(reporter, dict):
        if not reporter.get("name"):
            errors.append("报案人信息缺少: name（姓名）")
        if not reporter.get("phone"):
            errors.append("报案人信息缺少: phone（联系电话）")

    # 出险日期校验
    incident_date = data.get("incident_date")
    if not incident_date:
        errors.append("缺少必填字段: incident_date（出险日期）")

    # 出险原因校验
    incident_reason = data.get("incident_reason")
    if not incident_reason:
        errors.append("缺少必填字段: incident_reason（出险原因）")
    elif isinstance(incident_reason, str) and len(incident_reason.strip()) < 5:
        errors.append("出险原因描述过短")

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
