#!/usr/bin/env python3
"""输入数据校验脚本 - insurance-agent-activity-planner

校验代理人活动规划所需的输入数据完整性。

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
    # 业绩目标 > 0 校验
    performance_target = data.get("performance_target")
    if performance_target is None:
        errors.append("缺少必填字段: performance_target（业绩目标）")
    elif not isinstance(performance_target, (int, float)):
        errors.append("业绩目标应为数值类型")
    elif performance_target <= 0:
        errors.append(f"业绩目标应大于0，当前值: {performance_target}")

    # 当前时间节点校验
    current_time = data.get("current_time")
    if not current_time:
        errors.append("缺少必填字段: current_time（当前时间节点）")

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
