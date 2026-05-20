#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-underwriting-recording-inspection

校验双录质检所需的输入数据完整性。

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
    # 音视频文件路径校验
    media_path = data.get("media_path")
    if not media_path:
        errors.append("缺少必填字段: media_path（音视频文件路径）")
    elif not os.path.isfile(media_path):
        errors.append(f"音视频文件不存在: {media_path}")

    # 险种类型校验
    product_type = data.get("product_type")
    if product_type and product_type not in ["人寿", "健康", "意外", "年金"]:
        errors.append(f"不支持的险种类型: {product_type}")

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
