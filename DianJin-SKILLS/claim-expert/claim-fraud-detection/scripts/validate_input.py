#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-fraud-detection

校验理赔反欺诈检测所需的输入数据完整性。

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
    # 案件材料非空校验
    case_materials = data.get("case_materials")
    if not case_materials:
        errors.append("缺少必填字段: case_materials（案件材料不能为空）")
    elif isinstance(case_materials, list) and len(case_materials) == 0:
        errors.append("案件材料不能为空列表")

    # 案件基本信息校验
    case_info = data.get("case_info")
    if not case_info:
        errors.append("缺少必填字段: case_info（案件基本信息）")

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
