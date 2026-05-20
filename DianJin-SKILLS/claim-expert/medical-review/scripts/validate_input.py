#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-medical-review

校验理赔医学审查（费用审核+药品审核）所需的输入数据完整性。

用法：
    python scripts/validate_input.py --input <input_file_or_data>

退出码：
    0 - 校验通过
    1 - 校验失败
    2 - 参数错误
"""
import sys
import json
import os
import argparse

def validate(data: dict) -> list[str]:
    errors = []
    # 费用清单非空校验
    expense_list = data.get("expense_list")
    if not expense_list:
        errors.append("缺少必填字段: expense_list（费用清单不能为空）")
    elif isinstance(expense_list, list) and len(expense_list) == 0:
        errors.append("费用清单不能为空列表")

    # 图片路径校验
    image_path = data.get("image_path")
    if image_path and not os.path.isfile(image_path):
        errors.append(f"图片文件不存在: {image_path}")

    # 诊断信息校验
    diagnosis = data.get("diagnosis")
    if not diagnosis:
        errors.append("缺少必填字段: diagnosis（诊断信息）")
    elif isinstance(diagnosis, str) and len(diagnosis.strip()) < 2:
        errors.append("诊断信息过短")

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
