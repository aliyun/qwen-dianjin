#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-social-media

校验代理人社交媒体内容生成所需的输入数据完整性。

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
    # 发布数量校验（1-5）
    post_count = data.get("post_count")
    if post_count is None:
        errors.append("缺少必填字段: post_count（发布数量）")
    elif not isinstance(post_count, int):
        errors.append("发布数量应为整数类型")
    elif post_count < 1 or post_count > 5:
        errors.append(f"发布数量应在 1-5 之间，当前值: {post_count}")

    # 内容主题校验
    content_theme = data.get("content_theme")
    if not content_theme:
        errors.append("缺少必填字段: content_theme（内容主题）")
    elif isinstance(content_theme, str) and len(content_theme.strip()) < 2:
        errors.append("内容主题过短")

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
