#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-product-wiki"""
import sys, json, argparse

def validate(data: dict) -> list[str]:
    errors = []
    if "work_mode" not in data or not data["work_mode"]:
        errors.append("缺少必填字段: work_mode（工作模式）")
    else:
        valid = ["产品入库", "学习问答", "模拟演练", "话术推荐"]
        if data["work_mode"] not in valid:
            errors.append(f"工作模式无效: {data['work_mode']}")
    if "output_style" in data and data["output_style"]:
        valid = ["专业型", "亲和型", "数据型", "故事型"]
        if data["output_style"] not in valid:
            errors.append(f"输出风格无效: {data['output_style']}")
    return errors

def main():
    parser = argparse.ArgumentParser(description="产品知识库 - 输入数据校验")
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
