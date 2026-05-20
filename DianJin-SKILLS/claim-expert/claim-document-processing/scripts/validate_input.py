#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-claim-document-processing

校验理赔材料处理所需的输入数据完整性。

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def validate(data: dict) -> list[str]:
    errors = []
    # 输入目录存在校验
    input_dir = data.get("input_dir")
    if not input_dir:
        errors.append("缺少必填字段: input_dir（输入目录路径）")
    elif not os.path.isdir(input_dir):
        errors.append(f"输入目录不存在: {input_dir}")

    # 包含图片文件校验
    if input_dir and os.path.isdir(input_dir):
        files = os.listdir(input_dir)
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        if not image_files:
            errors.append(f"输入目录中未找到图片文件（支持的格式: {IMAGE_EXTENSIONS}）")

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
