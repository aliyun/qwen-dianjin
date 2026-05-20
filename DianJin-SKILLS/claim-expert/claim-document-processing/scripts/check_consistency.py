#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保险理赔材料一致性检查脚本

基于多模态大模型对所有理赔材料进行跨图逻辑关联校验，检查：
1. 身份一致性：所有图片中的患者姓名、身份证号是否一致
2. 时间线校验：入院→出院→发票开具时间是否符合逻辑
3. 费用逻辑校验：发票金额与费用清单总额是否匹配
4. 医疗逻辑校验：治疗行为与诊断是否相关、合理

用法:
    python3 check_consistency.py --input-dir <输入目录> --api-key <API_KEY>
    python3 check_consistency.py -i <目录> -k <KEY> -m qwen3.5-plus -o <输出路径>

参数:
    --input-dir, -i    包含理赔图片的源目录 (必需)
    --api-key, -k      DashScope API Key (必需)
    --model, -m        模型名称 (可选，默认 qwen3.5-plus)
    --output, -o       JSON报告输出路径 (可选，默认自动生成)
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


# ===== 一致性检查提示词 =====
CONSISTENCY_CHECK_SYSTEM_PROMPT = """# Role
你是一名资深保险理赔智能审核专家，专注于跨图逻辑关联校验和一致性分析。

# Task
对用户上传的全部理赔材料图片进行跨材料逻辑一致性检查，发现不一致或可疑之处。

# Constraints & Rules
- 真实性原则：所有判断必须严格基于图片可见内容，无法识别的字段填 null
- 预警优先：任何不一致或异常必须生成明确预警
- 格式规范：输出必须是合法 JSON，禁止 Markdown 标记，禁止额外解释文字

# Workflow

## Step 1: 信息提取
对每张图片独立提取关键信息：
- 患者姓名、身份证号
- 票据/报告编号、日期
- 金额（发票金额、费用清单金额）
- 医院名称
- 入出院时间
- 诊断信息

## Step 2: 身份一致性校验
- 比对所有图片中出现的患者姓名是否完全一致
- 比对身份证号是否一致
- 如果不同材料中出现不同姓名/ID，标记为高风险

## Step 3: 时间线校验
验证就诊时间逻辑：
- 入院日期 < 出院日期
- 出院日期 ≤ 发票开具日期（允许小范围偏差）
- 鉴定日期 > 出院日期
- 如果存在时间倒置，标记为高风险

## Step 4: 费用逻辑校验
- 比对医疗发票总金额与费用清单总额是否匹配
- 验证费用清单中各项明细金额之和是否等于总额
- 如有显著差异（>5%），标记为中/高风险
- 医疗行为与诊断是否相关

## Step 5: 医疗逻辑校验
- 治疗项目是否与诊断结果相匹配
- 用药是否合理（与诊断是否相关）
- 住院天数是否与病情严重程度匹配

# Output JSON Schema
{
  "consistency_result": {
    "overall_consistent": true/false,
    "consistency_score": 0.95,
    "summary": "string (整体一致性评估摘要)"
  },
  "identity_check": {
    "name_consistent": true/false,
    "id_consistent": true/false,
    "patient_names_found": ["string"],
    "id_numbers_found": ["string"],
    "inconsistency_detail": "string | null"
  },
  "timeline_check": {
    "timeline_consistent": true/false,
    "admission_date": "YYYY-MM-DD | null",
    "discharge_date": "YYYY-MM-DD | null",
    "invoice_dates": ["YYYY-MM-DD"],
    "appraisal_date": "YYYY-MM-DD | null",
    "inconsistency_detail": "string | null"
  },
  "financial_check": {
    "amount_consistent": true/false,
    "invoice_total": 0.0,
    "expense_list_total": 0.0,
    "difference": 0.0,
    "difference_percentage": "string",
    "inconsistency_detail": "string | null"
  },
  "medical_logic_check": {
    "logic_consistent": true/false,
    "diagnosis": "string",
    "treatment_match": true/false,
    "inconsistency_detail": "string | null"
  },
  "documents_extracted": [
    {
      "image_index": 0,
      "filename": "string",
      "category": "string",
      "patient_name": "string | null",
      "id_number": "string | null",
      "date": "string | null",
      "amount": 0.0,
      "hospital": "string | null"
    }
  ],
  "risk_warnings": [
    {
      "level": "high | medium | low",
      "type": "identity | timeline | financial | medical_logic",
      "message": "string"
    }
  ]
}

请直接输出JSON结果。"""


def build_user_prompt(file_info: list) -> str:
    """构建用户提示词"""
    prompt = "请对以下理赔材料进行跨图逻辑一致性检查：\n\n"
    for i, info in enumerate(file_info):
        prompt += f"图片{i}: {info['file_name']}"
        if info['file_type'] == 'pdf':
            prompt += f" (PDF第{info['page_index'] + 1}页)"
        prompt += f", 尺寸: {info['size']}\n"
    prompt += "\n请按照系统提示中的JSON格式返回一致性检查结果。"
    return prompt


def extract_consistency_from_analysis(analysis_path: str) -> dict:
    """从前置解析结果中提取一致性检查信息"""
    print(f"  使用前置解析结果: {analysis_path}")
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    cross_validation = analysis.get("cross_validation", {})
    documents = analysis.get("documents", [])

    # 构建 documents_extracted
    docs_extracted = []
    for doc in documents:
        ext = doc.get("extraction", {})
        docs_extracted.append({
            "image_index": doc.get("image_index", 0),
            "filename": doc.get("filename", ""),
            "category": doc.get("category", "unknown"),
            "patient_name": ext.get("patient_name"),
            "id_number": ext.get("id_number"),
            "date": ext.get("date"),
            "amount": ext.get("amount"),
            "hospital": ext.get("hospital_name")
        })

    # 提取一致性相关的 risk_warnings
    all_warnings = analysis.get("risk_warnings", [])
    consistency_warnings = [
        w for w in all_warnings
        if w.get("type") in ("consistency", "identity", "timeline", "financial", "medical_logic")
    ]

    # 构建一致性结果（从 cross_validation 字段映射）
    consistency_result = {
        "overall_consistent": (
            cross_validation.get("name_consistent", True) and
            cross_validation.get("timeline_consistent", True) and
            cross_validation.get("amount_consistent", True)
        ),
        "summary": cross_validation.get("logic_summary", "")
    }

    identity_check = {
        "name_consistent": cross_validation.get("name_consistent", True),
        "id_consistent": cross_validation.get("id_consistent", True),
        "inconsistency_detail": None
    }

    timeline_check = {
        "timeline_consistent": cross_validation.get("timeline_consistent", True),
        "inconsistency_detail": None
    }

    financial_check = {
        "amount_consistent": cross_validation.get("amount_consistent", True),
        "inconsistency_detail": None
    }

    # 从 inconsistency_details 分配具体信息
    for detail in cross_validation.get("inconsistency_details", []):
        detail_str = detail if isinstance(detail, str) else str(detail)
        if "姓名" in detail_str or "身份" in detail_str:
            identity_check["inconsistency_detail"] = detail_str
        elif "时间" in detail_str or "日期" in detail_str:
            timeline_check["inconsistency_detail"] = detail_str
        elif "金额" in detail_str or "费用" in detail_str:
            financial_check["inconsistency_detail"] = detail_str

    return {
        "consistency_result": consistency_result,
        "identity_check": identity_check,
        "timeline_check": timeline_check,
        "financial_check": financial_check,
        "medical_logic_check": {"logic_consistent": True},
        "documents_extracted": docs_extracted,
        "risk_warnings": consistency_warnings,
        "source": "analysis_result",
        "model": analysis.get("model", "unknown"),
        "file_stats": analysis.get("file_stats", {})
    }


def run_consistency_check(
    input_dir: str = None,
    api_key: str = None,
    model: str = "qwen3.5-plus",
    output_path: str = None,
    analysis_result_path: str = None
) -> dict:
    """
    执行材料一致性检查

    Args:
        input_dir: 输入目录（独立推理时需要）
        api_key: DashScope API Key（独立推理时需要）
        model: 模型名称
        output_path: 输出JSON路径
        analysis_result_path: 前置解析结果路径（若提供则跳过模型调用）

    Returns:
        一致性检查结果字典
    """
    print_separator("保险理赔材料一致性检查")

    # ===== 模式判断 =====
    if analysis_result_path:
        print(f"模式: 复用前置解析结果")
        extracted = extract_consistency_from_analysis(analysis_result_path)
        report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "input_dir": input_dir or "(from analysis)",
            "model": extracted.get("model", model),
            "file_stats": extracted.get("file_stats", {}),
            "analysis_metadata": {"source": "pre_analysis", "analysis_file": analysis_result_path},
            "consistency_result": extracted.get("consistency_result", {}),
            "identity_check": extracted.get("identity_check", {}),
            "timeline_check": extracted.get("timeline_check", {}),
            "financial_check": extracted.get("financial_check", {}),
            "medical_logic_check": extracted.get("medical_logic_check", {}),
            "documents_extracted": extracted.get("documents_extracted", []),
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

        # 1. 加载文件
        print("\n[1/3] 加载材料文件...")
        loaded_files, stats = load_directory(input_dir)
        if not loaded_files:
            print("[ERROR] 未找到可处理的材料文件")
            return {"success": False, "error": "未找到可处理的材料文件"}

        print(f"  已加载: {stats['total_files']} 个文件, "
              f"{stats['total_pages']} 页 (图片: {stats['image']}, PDF: {stats['pdf']})")

        # 2. 调用模型
        print(f"\n[2/3] 调用模型进行跨图分析 ({model})...")
        file_info = get_file_info_list(loaded_files)
        images = [f.base64_data for f in loaded_files]

        user_prompt = build_user_prompt(file_info)

        start_time = time.time()
        response_text, metadata = call_qwen_multimodal(
            api_key=api_key,
            model=model,
            system_prompt=CONSISTENCY_CHECK_SYSTEM_PROMPT,
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
        print("\n[3/3] 解析一致性检查结果...")
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
            "file_stats": stats,
            "analysis_metadata": {
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "duration_ms": metadata.get("duration_ms", 0)
            },
            "consistency_result": result.get("consistency_result", {}),
            "identity_check": result.get("identity_check", {}),
            "timeline_check": result.get("timeline_check", {}),
            "financial_check": result.get("financial_check", {}),
            "medical_logic_check": result.get("medical_logic_check", {}),
            "documents_extracted": result.get("documents_extracted", []),
            "risk_warnings": result.get("risk_warnings", [])
        }

    # 保存报告
    if not output_path:
        output_dir = create_output_dir("consistency")
        output_path = os.path.join(output_dir, "consistency.json")

    save_json_report(report, output_path)

    # 输出摘要
    print_separator("检查结果摘要")
    cr = report.get("consistency_result", {})
    overall = cr.get("overall_consistent", "未知")
    score = cr.get("consistency_score", "N/A")
    print(f"  整体一致性: {'一致' if overall else '存在不一致'} (评分: {score})")

    identity = report.get("identity_check", {})
    if not identity.get("name_consistent", True):
        print(f"  [!] 姓名不一致: {identity.get('inconsistency_detail', '')}")
    if not identity.get("id_consistent", True):
        print(f"  [!] 身份证号不一致: {identity.get('inconsistency_detail', '')}")

    timeline = report.get("timeline_check", {})
    if not timeline.get("timeline_consistent", True):
        print(f"  [!] 时间线不一致: {timeline.get('inconsistency_detail', '')}")

    financial = report.get("financial_check", {})
    if not financial.get("amount_consistent", True):
        diff = financial.get("difference", 0)
        pct = financial.get("difference_percentage", "?")
        print(f"  [!] 金额不一致: 差异 {diff} ({pct})")

    medical = report.get("medical_logic_check", {})
    if not medical.get("logic_consistent", True):
        print(f"  [!] 医疗逻辑不一致: {medical.get('inconsistency_detail', '')}")

    warnings = report.get("risk_warnings", [])
    if warnings:
        print(f"\n  风险预警: {len(warnings)} 条")
        for w in warnings:
            print(f"    - [{w.get('level', '?')}][{w.get('type', '?')}] {w.get('message', '')}")

    print(f"\n报告已保存: {output_path}")
    return report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="保险理赔材料一致性检查 - 跨图逻辑关联校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 独立推理模式:
  python3 check_consistency.py -i ./claims -k sk-xxx
  python3 check_consistency.py -i ./claims -k sk-xxx -m qwen3.5-plus -o ./report.json

  # 复用前置解析结果（无需调用大模型）:
  python3 check_consistency.py --analysis-result ./analysis.json
"""
    )
    parser.add_argument("--input-dir", "-i", default=None, help="包含理赔图片的源目录")
    parser.add_argument("--api-key", "-k", default=None, help="DashScope API Key")
    parser.add_argument("--model", "-m", default="qwen3.5-plus",
                        help="模型名称（默认 qwen3.5-plus）")
    parser.add_argument("--output", "-o", default=None, help="JSON报告输出路径")
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

    result = run_consistency_check(
        input_dir=input_dir,
        api_key=args.api_key,
        model=args.model,
        output_path=args.output,
        analysis_result_path=args.analysis_result
    )

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
