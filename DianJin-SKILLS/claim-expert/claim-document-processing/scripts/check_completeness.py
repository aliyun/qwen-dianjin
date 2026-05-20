#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保险理赔材料完整性检查脚本

基于多模态大模型一次性分析所有理赔材料图片，检查：
1. 类型完整性：核心材料链是否缺失（如有发票应配套病历和费用清单）
2. 页面完整性：多页文档是否存在缺页
3. 重复材料识别：检测发票/病历/清单等重复提交

用法:
    python3 check_completeness.py --input-dir <输入目录> --api-key <API_KEY>
    python3 check_completeness.py -i <目录> -k <KEY> -m qwen3.5-plus -o <输出路径>

参数:
    --input-dir, -i    包含理赔图片的源目录 (必需)
    --api-key, -k      DashScope API Key (必需)
    --model, -m        模型名称 (可选，默认 qwen3.5-plus)
    --output, -o       JSON报告输出路径 (可选，默认自动生成)
    --claim-type, -t   案件类型 (可选，默认 "通用医疗")
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径以引用 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_directory, get_file_info_list,
    call_qwen_multimodal, extract_json_from_text,
    save_json_report, print_separator, LoadedFile,
    create_output_dir
)


# ===== 完整性检查提示词 =====
COMPLETENESS_CHECK_SYSTEM_PROMPT = """# Role
你是一名资深保险理赔智能审核专家，专注于理赔材料的完整性与重复性检测。

# Task
对用户上传的全部理赔材料图片进行完整性校验和重复材料识别。

# Constraints & Rules
- 真实性原则：所有判断必须严格基于图片可见内容，无法识别的字段填 null
- 预警优先：缺失、重复、可疑材料必须生成明确预警
- 格式规范：输出必须是合法 JSON，禁止 Markdown 标记，禁止额外解释文字

# Workflow

## Step 1: 单图识别
对每张图片识别其材料类型（从以下选项中选择）：
- medical_invoice: 医疗发票（含住院/门诊/电子发票）
  【重要】必须是正式税务发票，有发票代码、发票号码、税号等要素
  【排除】收费凭证、收据、缴费条等非正式发票归为 other
- appraisal_report: 鉴定报告（含伤残/司法/资质鉴定）
- appraisal_fee: 鉴定费用发票
- medical_record: 病历资料（含入/出院记录、诊断证明、疾病证明、检查报告）
- expense_list: 费用清单（含住院/用药明细、医保结算单）
- other: 其他无法归类的材料

同时提取关键信息：患者姓名、证件号、票据编号、日期、金额等。

## Step 2: 类型完整性检查
基于案件类型（{{claim_type}}），判断核心材料链是否完整：
- 通用医疗理赔通常需要：医疗发票 + 病历资料 + 费用清单
- 有鉴定的案件还需要：鉴定报告 + 鉴定费用发票
- 列出缺失的材料类型

## Step 3: 页面完整性检查
- 检查多页文档的页码标识
- 若无明显页码但内容中断，推测"疑似缺页"并说明
- 不直接断定"确定缺页"

## Step 4: 重复材料识别
发票类：比对"票据代码+号码+金额"，完全一致为 high 置信度，号码相同但打印时间不同为 medium
病历/证明类：OCR 文本相似度≥95% 且关键字段一致为 high
费用清单类：比对"住院号+合计+时间范围"
边界处理：
- 同一住院多次结算、病历更新版本等不判定为重复
- 同一发票的纸质版/电子版/打印件判定为重复（high）
- 连号发票或不同日期发票不判定为重复

# Output JSON Schema
{
  "completeness_check": {
    "is_complete": true/false,
    "claim_type": "string (检测到的案件类型)",
    "existing_types": ["string (已有的材料类型)"],
    "missing_types": ["string (缺失的材料类型及说明)"],
    "missing_pages_warning": "string | null (缺页预警)"
  },
  "duplicate_materials": [
    {
      "image_indices": [0, 1],
      "filenames": ["file1.jpg", "file2.jpg"],
      "reason": "string (重复原因)",
      "document_type": "string",
      "confidence": "high | medium | low"
    }
  ],
  "documents_summary": [
    {
      "image_index": 0,
      "filename": "string",
      "category": "string",
      "key_info": "string (简要关键信息)"
    }
  ],
  "risk_warnings": [
    {
      "level": "high | medium | low",
      "type": "completeness | duplicate | page_missing",
      "message": "string"
    }
  ]
}

请直接输出JSON结果。"""


def build_user_prompt(file_info: list, claim_type: str) -> str:
    """构建用户提示词"""
    prompt = f"案件类型: {claim_type}\n\n请分析以下理赔材料的完整性：\n\n"
    for i, info in enumerate(file_info):
        prompt += f"图片{i}: {info['file_name']}"
        if info['file_type'] == 'pdf':
            prompt += f" (PDF第{info['page_index'] + 1}页)"
        prompt += f", 尺寸: {info['size']}\n"
    prompt += "\n请按照系统提示中的JSON格式返回完整性检查结果。"
    return prompt


def extract_from_analysis_result(analysis_path: str) -> dict:
    """从前置解析结果中提取完整性检查信息"""
    print(f"  使用前置解析结果: {analysis_path}")
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    # 直接提取 completeness_check 部分
    completeness = analysis.get("completeness_check", {})
    documents = analysis.get("documents", [])

    # 构建 documents_summary
    docs_summary = []
    for doc in documents:
        docs_summary.append({
            "image_index": doc.get("image_index", 0),
            "filename": doc.get("filename", ""),
            "category": doc.get("category", "unknown"),
            "key_info": doc.get("sub_type", "")
        })

    # 提取完整性相关的 risk_warnings
    all_warnings = analysis.get("risk_warnings", [])
    completeness_warnings = [
        w for w in all_warnings
        if w.get("type") in ("completeness", "duplicate", "page_missing")
    ]

    return {
        "completeness_check": completeness,
        "duplicate_materials": completeness.get("duplicate_materials", []),
        "documents_summary": docs_summary,
        "risk_warnings": completeness_warnings,
        "source": "analysis_result",
        "analysis_file": analysis_path,
        "model": analysis.get("model", "unknown"),
        "file_stats": analysis.get("file_stats", {})
    }


def run_completeness_check(
    input_dir: str = None,
    api_key: str = None,
    model: str = "qwen3.5-plus",
    output_path: str = None,
    claim_type: str = "通用医疗",
    analysis_result_path: str = None
) -> dict:
    """
    执行材料完整性检查

    Args:
        input_dir: 输入目录（独立推理时需要）
        api_key: DashScope API Key（独立推理时需要）
        model: 模型名称
        output_path: 输出JSON路径
        claim_type: 案件类型
        analysis_result_path: 前置解析结果路径（若提供则跳过模型调用）

    Returns:
        完整性检查结果字典
    """
    print_separator("保险理赔材料完整性检查")

    # ===== 模式判断 =====
    if analysis_result_path:
        print(f"模式: 复用前置解析结果")
        extracted = extract_from_analysis_result(analysis_result_path)
        report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "input_dir": input_dir or "(from analysis)",
            "model": extracted.get("model", model),
            "claim_type": claim_type,
            "file_stats": extracted.get("file_stats", {}),
            "analysis_metadata": {"source": "pre_analysis", "analysis_file": analysis_result_path},
            "completeness_check": extracted.get("completeness_check", {}),
            "duplicate_materials": extracted.get("duplicate_materials", []),
            "documents_summary": extracted.get("documents_summary", []),
            "risk_warnings": extracted.get("risk_warnings", [])
        }
    else:
        # ===== 独立推理模式 =====
        if not input_dir or not api_key:
            print("[ERROR] 独立模式需要 --input-dir 和 --api-key")
            return {"success": False, "error": "缺少必要参数"}

        print(f"模式: 独立大模型推理")
        print(f"输入目录: {input_dir}")
        print(f"模型: {model}")
        print(f"案件类型: {claim_type}")

        # 1. 加载文件
        print("\n[1/3] 加载材料文件...")
        loaded_files, stats = load_directory(input_dir)
        if not loaded_files:
            print("[ERROR] 未找到可处理的材料文件")
            return {"success": False, "error": "未找到可处理的材料文件"}

        print(f"  已加载: {stats['total_files']} 个文件, "
              f"{stats['total_pages']} 页 (图片: {stats['image']}, PDF: {stats['pdf']})")

        # 2. 调用模型
        print(f"\n[2/3] 调用模型分析 ({model})...")
        file_info = get_file_info_list(loaded_files)
        images = [f.base64_data for f in loaded_files]

        system_prompt = COMPLETENESS_CHECK_SYSTEM_PROMPT.replace("{{claim_type}}", claim_type)
        user_prompt = build_user_prompt(file_info, claim_type)

        start_time = time.time()
        response_text, metadata = call_qwen_multimodal(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            max_tokens=8192
        )
        elapsed = time.time() - start_time

        if not metadata.get("success"):
            error_msg = metadata.get("error", "模型调用失败")
            print(f"  [ERROR] {error_msg}")
            return {"success": False, "error": error_msg}

        print(f"  耗时: {elapsed:.1f}s | "
              f"Token: {metadata.get('input_tokens', 0)}/{metadata.get('output_tokens', 0)}")

        # 3. 解析结果
        print("\n[3/3] 解析检查结果...")
        result = extract_json_from_text(response_text)
        if not result:
            print("  [WARN] JSON解析失败，保存原始响应")
            result = {"raw_response": response_text}

        # 组装最终报告
        report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "input_dir": input_dir,
            "model": model,
            "claim_type": claim_type,
            "file_stats": stats,
            "analysis_metadata": {
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "duration_ms": metadata.get("duration_ms", 0)
            },
            "completeness_check": result.get("completeness_check", {}),
            "duplicate_materials": result.get("duplicate_materials", []),
            "documents_summary": result.get("documents_summary", []),
            "risk_warnings": result.get("risk_warnings", [])
        }

    # 保存报告
    if not output_path:
        output_dir = create_output_dir("completeness")
        output_path = os.path.join(output_dir, "completeness.json")

    save_json_report(report, output_path)

    # 输出摘要
    print_separator("检查结果摘要")
    cc = report.get("completeness_check", {})
    is_complete = cc.get("is_complete", "未知")
    missing = cc.get("missing_types", [])
    duplicates = report.get("duplicate_materials", [])
    warnings = report.get("risk_warnings", [])

    print(f"  材料完整性: {'完整' if is_complete else '不完整'}")
    if missing:
        print(f"  缺失材料: {', '.join(missing)}")
    if duplicates:
        print(f"  重复材料: {len(duplicates)} 组")
        for dup in duplicates:
            print(f"    - [{dup.get('confidence', '?')}] {dup.get('reason', '')}")
    if warnings:
        print(f"  风险预警: {len(warnings)} 条")
        for w in warnings:
            print(f"    - [{w.get('level', '?')}] {w.get('message', '')}")

    print(f"\n报告已保存: {output_path}")
    return report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="保险理赔材料完整性检查 - 基于多模态大模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 独立推理模式:
  python3 check_completeness.py -i ./claims -k sk-xxx
  python3 check_completeness.py -i ./claims -k sk-xxx -m qwen3.5-plus -t "交通事故"

  # 复用前置解析结果（无需调用大模型）:
  python3 check_completeness.py --analysis-result ./analysis.json
"""
    )
    parser.add_argument("--input-dir", "-i", default=None, help="包含理赔图片的源目录")
    parser.add_argument("--api-key", "-k", default=None, help="DashScope API Key")
    parser.add_argument("--model", "-m", default="qwen3.5-plus",
                        help="模型名称（默认 qwen3.5-plus）")
    parser.add_argument("--output", "-o", default=None, help="JSON报告输出路径")
    parser.add_argument("--claim-type", "-t", default="通用医疗",
                        help="案件类型（默认 '通用医疗'）")
    parser.add_argument("--analysis-result", "-a", default=None,
                        help="前置解析结果JSON路径（若提供则跳过大模型调用）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 参数校验
    if not args.analysis_result and (not args.input_dir or not args.api_key):
        print("错误: 需要提供 --analysis-result 或同时提供 --input-dir 和 --api-key")
        sys.exit(1)

    input_dir = os.path.abspath(args.input_dir) if args.input_dir else None
    if input_dir and not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    result = run_completeness_check(
        input_dir=input_dir,
        api_key=args.api_key,
        model=args.model,
        output_path=args.output,
        claim_type=args.claim_type,
        analysis_result_path=args.analysis_result
    )

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
