#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-coverage-analysis

校验责任免除检查所需的输入数据完整性。

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
    # 理赔材料非空校验
    claim_materials = data.get("claim_materials")
    if not claim_materials:
        errors.append("缺少必填字段: claim_materials（理赔材料不能为空）")
    elif isinstance(claim_materials, list) and len(claim_materials) == 0:
        errors.append("理赔材料不能为空列表")

    # 出险场景校验
    incident_scenario = data.get("incident_scenario")
    if not incident_scenario:
        errors.append("缺少必填字段: incident_scenario（出险场景）")
    elif isinstance(incident_scenario, str) and len(incident_scenario.strip()) < 5:
        errors.append("出险场景描述过短")

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
