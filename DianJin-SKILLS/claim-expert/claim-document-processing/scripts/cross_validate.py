#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保险理赔材料大模型交叉验证脚本

使用两个独立的多模态大模型（默认 Qwen + Kimi）分别对理赔材料进行分析，
然后对比两个模型的结果，找出分歧点，提高审核可靠性。

工作流程:
1. 主模型（Qwen3.5）独立分析所有材料
2. 验证模型（Kimi-K2.5）独立分析相同材料
3. 对比两个模型的分类、提取信息、金额等关键结果
4. 输出差异报告和最终综合判定

用法:
    python3 cross_validate.py --input-dir <输入目录> --api-key <API_KEY>
    python3 cross_validate.py -i <目录> -k <KEY> --primary-model qwen3.5-plus --secondary-model kimi-k2.5

参数:
    --input-dir, -i          包含理赔图片的源目录 (必需)
    --api-key, -k            DashScope API Key (必需)
    --primary-model, -p      主模型名称 (默认 qwen3.5-plus)
    --secondary-model, -s    验证模型名称 (默认 kimi-k2.5)
    --output, -o             JSON报告输出路径 (可选)
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
    call_qwen_multimodal, call_kimi_multimodal,
    extract_json_from_text, save_json_report,
    print_separator, LoadedFile, DASHSCOPE_AVAILABLE,
    create_output_dir
)


# ===== 交叉验证用的综合分析提示词 =====
CROSS_VALIDATION_ANALYSIS_PROMPT = """# Role
你是一名资深保险理赔智能审核专家。你具备多模态分析能力，能够精准识别、分类并提取用户上传的理赔材料中的关键信息。

# CRITICAL OUTPUT RULES
1. 只输出JSON：直接输出JSON格式结果，不要输出任何分析过程、Markdown文本或说明
2. 无代码块：不要使用```json或```包裹，直接输出纯JSON
3. 真实性原则：提取信息必须严格基于图片内容，无法识别的字段填 null
4. 每张图片必须有记录：documents数组中每张输入图片均需对应一个元素

# Workflow
1. 对每张图片进行分析：清晰度检测、材料分类、关键信息提取
2. 材料分类选项：
   - medical_invoice (医疗发票)
   - appraisal_report (鉴定报告)
   - appraisal_fee (鉴定费用)
   - medical_record (病历资料)
   - expense_list (费用清单)
   - other (其他材料)
3. 完整性校验：检查材料类型是否齐全
4. 跨图校验：姓名/身份证一致性、时间线逻辑、费用匹配

# Output JSON Schema
{
  "case_summary": {
    "total_images": 0,
    "overall_quality": "high | medium | low",
    "claim_type_detected": "string"
  },
  "documents": [
    {
      "image_index": 0,
      "category": "string",
      "quality_status": "high | low",
      "extraction": {
        "patient_name": "string | null",
        "id_number": "string | null",
        "document_number": "string | null",
        "date": "YYYY-MM-DD | null",
        "amount": 0.0,
        "diagnosis": "string | null",
        "hospital_name": "string | null",
        "time_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
      }
    }
  ],
  "completeness_check": {
    "is_complete": true/false,
    "missing_types": ["string"]
  },
  "cross_validation": {
    "name_consistent": true/false,
    "timeline_consistent": true/false,
    "logic_summary": "string",
    "inconsistency_details": ["string"]
  },
  "risk_warnings": [
    {
      "level": "high | medium | low",
      "type": "quality | completeness | consistency | data_extraction",
      "message": "string"
    }
  ]
}

请开始分析用户上传的理赔材料图片，直接输出JSON结果。"""


def build_user_prompt(file_info: list) -> str:
    """构建用户提示词"""
    prompt = "请分析以下理赔材料图片：\n\n"
    for i, info in enumerate(file_info):
        prompt += f"图片{i}: {info['file_name']}"
        if info['file_type'] == 'pdf':
            prompt += f" (PDF第{info['page_index'] + 1}页)"
        prompt += f", 尺寸: {info['size']}\n"
    prompt += "\n请按照系统提示中的JSON格式返回完整的分析结果。"
    return prompt


def compare_results(primary_result: dict, secondary_result: dict) -> dict:
    """
    对比两个模型的分析结果，生成差异报告

    Args:
        primary_result: 主模型分析结果
        secondary_result: 验证模型分析结果

    Returns:
        差异对比报告
    """
    differences = []
    agreements = []

    # 1. 对比文档分类
    primary_docs = primary_result.get("documents", [])
    secondary_docs = secondary_result.get("documents", [])

    classification_diffs = []
    for i, (p_doc, s_doc) in enumerate(zip(primary_docs, secondary_docs)):
        p_cat = p_doc.get("category", "unknown")
        s_cat = s_doc.get("category", "unknown")
        if p_cat != s_cat:
            classification_diffs.append({
                "image_index": i,
                "primary_category": p_cat,
                "secondary_category": s_cat,
                "filename": p_doc.get("filename", f"image_{i}")
            })

    if classification_diffs:
        differences.append({
            "type": "classification",
            "description": f"分类不一致: {len(classification_diffs)} 个文档",
            "details": classification_diffs
        })
    else:
        agreements.append("文档分类结果完全一致")

    # 2. 对比跨图验证
    p_cv = primary_result.get("cross_validation", {})
    s_cv = secondary_result.get("cross_validation", {})

    if p_cv.get("name_consistent") != s_cv.get("name_consistent"):
        differences.append({
            "type": "identity",
            "description": "姓名一致性判定不同",
            "primary": p_cv.get("name_consistent"),
            "secondary": s_cv.get("name_consistent")
        })

    if p_cv.get("timeline_consistent") != s_cv.get("timeline_consistent"):
        differences.append({
            "type": "timeline",
            "description": "时间线一致性判定不同",
            "primary": p_cv.get("timeline_consistent"),
            "secondary": s_cv.get("timeline_consistent")
        })

    # 3. 对比完整性判定
    p_cc = primary_result.get("completeness_check", {})
    s_cc = secondary_result.get("completeness_check", {})

    if p_cc.get("is_complete") != s_cc.get("is_complete"):
        differences.append({
            "type": "completeness",
            "description": "完整性判定不同",
            "primary_complete": p_cc.get("is_complete"),
            "secondary_complete": s_cc.get("is_complete"),
            "primary_missing": p_cc.get("missing_types", []),
            "secondary_missing": s_cc.get("missing_types", [])
        })

    # 4. 对比金额提取
    amount_diffs = []
    for i, (p_doc, s_doc) in enumerate(zip(primary_docs, secondary_docs)):
        p_amount = p_doc.get("extraction", {}).get("amount")
        s_amount = s_doc.get("extraction", {}).get("amount")
        if p_amount and s_amount:
            try:
                p_val = float(p_amount) if p_amount else 0
                s_val = float(s_amount) if s_amount else 0
                if abs(p_val - s_val) > 0.01 and p_val > 0:
                    diff_pct = abs(p_val - s_val) / max(p_val, s_val) * 100
                    amount_diffs.append({
                        "image_index": i,
                        "primary_amount": p_val,
                        "secondary_amount": s_val,
                        "difference_percentage": f"{diff_pct:.1f}%"
                    })
            except (TypeError, ValueError):
                pass

    if amount_diffs:
        differences.append({
            "type": "financial",
            "description": f"金额提取差异: {len(amount_diffs)} 处",
            "details": amount_diffs
        })
    else:
        agreements.append("金额提取结果一致")

    # 5. 对比患者信息
    p_names = set()
    s_names = set()
    for doc in primary_docs:
        name = doc.get("extraction", {}).get("patient_name")
        if name:
            p_names.add(name)
    for doc in secondary_docs:
        name = doc.get("extraction", {}).get("patient_name")
        if name:
            s_names.add(name)

    if p_names != s_names and p_names and s_names:
        differences.append({
            "type": "patient_info",
            "description": "患者信息提取不一致",
            "primary_names": list(p_names),
            "secondary_names": list(s_names)
        })
    elif p_names and s_names:
        agreements.append("患者信息提取一致")

    # 生成综合判定
    is_consistent = len(differences) == 0
    confidence = 1.0 - min(len(differences) * 0.15, 0.6)

    return {
        "is_consistent": is_consistent,
        "confidence_score": round(confidence, 2),
        "total_differences": len(differences),
        "total_agreements": len(agreements),
        "differences": differences,
        "agreements": agreements,
        "recommendation": (
            "两个模型分析结果一致，可信度高" if is_consistent
            else f"存在 {len(differences)} 处差异，建议人工复核差异项"
        )
    }


def run_cross_validation(
    input_dir: str,
    api_key: str,
    primary_model: str = "qwen3.5-plus",
    secondary_model: str = "kimi-k2.5",
    output_path: str = None,
    analysis_result_path: str = None
) -> dict:
    """
    执行双模型交叉验证

    Args:
        input_dir: 输入目录
        api_key: DashScope API Key
        primary_model: 主模型
        secondary_model: 验证模型
        output_path: 输出路径
        analysis_result_path: 前置解析结果（作为主模型结果，跳过主模型调用）

    Returns:
        交叉验证结果字典
    """
    print_separator("保险理赔材料 - 大模型交叉验证")
    print(f"输入目录: {input_dir}")
    print(f"主模型: {primary_model}")
    print(f"验证模型: {secondary_model}")

    # 检查 dashscope 依赖（Kimi模型需要）
    is_kimi = "kimi" in secondary_model.lower()
    if is_kimi and not DASHSCOPE_AVAILABLE:
        print("[WARN] dashscope 未安装，Kimi模型将使用 OpenAI 兼容模式调用")

    # 1. 加载文件（交叉验证始终需要加载图片给次模型）
    print("\n[1/4] 加载材料文件...")
    loaded_files, stats = load_directory(input_dir)
    if not loaded_files:
        print("[ERROR] 未找到可处理的材料文件")
        return {"success": False, "error": "未找到可处理的材料文件"}

    print(f"  已加载: {stats['total_files']} 个文件, "
          f"{stats['total_pages']} 页 (图片: {stats['image']}, PDF: {stats['pdf']})")

    file_info = get_file_info_list(loaded_files)
    images = [f.base64_data for f in loaded_files]
    user_prompt = build_user_prompt(file_info)

    # 2. 主模型分析（可复用前置解析结果）
    if analysis_result_path:
        print(f"\n[2/4] 复用前置解析结果作为主模型分析 ({primary_model})...")
        print(f"  解析文件: {analysis_result_path}")
        with open(analysis_result_path, 'r', encoding='utf-8') as f:
            primary_result = json.load(f)
        # 提取核心分析部分
        primary_result = {
            "case_summary": primary_result.get("case_summary", {}),
            "documents": primary_result.get("documents", []),
            "completeness_check": primary_result.get("completeness_check", {}),
            "cross_validation": primary_result.get("cross_validation", {}),
            "risk_warnings": primary_result.get("risk_warnings", [])
        }
        primary_meta = {
            "source": "pre_analysis",
            "analysis_file": analysis_result_path,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_ms": 0
        }
        print(f"  已加载主模型分析结果: {len(primary_result.get('documents', []))} 个文档")
    else:
        print(f"\n[2/4] 主模型分析 ({primary_model})...")
        start_time = time.time()
        primary_text, primary_meta = call_qwen_multimodal(
            api_key=api_key,
            model=primary_model,
            system_prompt=CROSS_VALIDATION_ANALYSIS_PROMPT,
            user_prompt=user_prompt,
            images=images,
            max_tokens=8192
        )
        primary_elapsed = time.time() - start_time

        if not primary_meta.get("success"):
            error_msg = f"主模型调用失败: {primary_meta.get('error', '未知错误')}"
            print(f"  [ERROR] {error_msg}")
            return {"success": False, "error": error_msg}

        print(f"  耗时: {primary_elapsed:.1f}s | "
              f"Token: {primary_meta.get('input_tokens', 0)}/{primary_meta.get('output_tokens', 0)}")

        primary_result = extract_json_from_text(primary_text)
        if not primary_result:
            print("  [WARN] 主模型JSON解析失败")
            primary_result = {"documents": [], "raw_response": primary_text}

    # 3. 验证模型分析
    print(f"\n[3/4] 验证模型分析 ({secondary_model})...")

    # 构建验证模型的用户提示词（包含主模型的提取信息供对比）
    # 但同时要求独立分析，避免偏向
    validation_user_prompt = user_prompt + "\n\n请独立分析以上材料，直接输出JSON结果。"

    start_time = time.time()

    # 根据模型类型选择调用方式
    if is_kimi and DASHSCOPE_AVAILABLE:
        secondary_text, secondary_meta = call_kimi_multimodal(
            api_key=api_key,
            model=secondary_model,
            system_prompt=CROSS_VALIDATION_ANALYSIS_PROMPT,
            user_prompt=validation_user_prompt,
            images=images,
            max_tokens=8192
        )
    else:
        secondary_text, secondary_meta = call_qwen_multimodal(
            api_key=api_key,
            model=secondary_model,
            system_prompt=CROSS_VALIDATION_ANALYSIS_PROMPT,
            user_prompt=validation_user_prompt,
            images=images,
            max_tokens=8192
        )
    secondary_elapsed = time.time() - start_time

    if not secondary_meta.get("success"):
        error_msg = f"验证模型调用失败: {secondary_meta.get('error', '未知错误')}"
        print(f"  [ERROR] {error_msg}")
        # 验证模型失败不阻断，只是标记
        secondary_result = {"documents": [], "error": error_msg}
        secondary_meta["success"] = False
    else:
        print(f"  耗时: {secondary_elapsed:.1f}s | "
              f"Token: {secondary_meta.get('input_tokens', 0)}/{secondary_meta.get('output_tokens', 0)}")
        secondary_result = extract_json_from_text(secondary_text)
        if not secondary_result:
            print("  [WARN] 验证模型JSON解析失败")
            secondary_result = {"documents": [], "raw_response": secondary_text}

    # 4. 对比分析
    print("\n[4/4] 对比两个模型结果...")
    comparison = compare_results(primary_result, secondary_result)

    # 组装最终报告
    report = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "input_dir": input_dir,
        "models": {
            "primary": primary_model,
            "secondary": secondary_model
        },
        "file_stats": stats,
        "cross_validation_summary": comparison,
        "primary_analysis": {
            "metadata": {
                "input_tokens": primary_meta.get("input_tokens", 0),
                "output_tokens": primary_meta.get("output_tokens", 0),
                "duration_ms": primary_meta.get("duration_ms", 0)
            },
            "result": primary_result
        },
        "secondary_analysis": {
            "metadata": {
                "input_tokens": secondary_meta.get("input_tokens", 0),
                "output_tokens": secondary_meta.get("output_tokens", 0),
                "duration_ms": secondary_meta.get("duration_ms", 0)
            },
            "result": secondary_result
        }
    }

    # 保存报告
    if not output_path:
        output_dir = create_output_dir("cross_validate")
        output_path = os.path.join(output_dir, "cross_validation.json")

    save_json_report(report, output_path)

    # 输出摘要
    print_separator("交叉验证结果摘要")
    print(f"  整体一致性: {'一致' if comparison['is_consistent'] else '存在差异'}")
    print(f"  可信度评分: {comparison['confidence_score']}")
    print(f"  差异数量: {comparison['total_differences']}")
    print(f"  一致项: {comparison['total_agreements']}")

    if comparison['differences']:
        print(f"\n  差异明细:")
        for diff in comparison['differences']:
            print(f"    - [{diff['type']}] {diff['description']}")

    if comparison['agreements']:
        print(f"\n  一致项:")
        for agr in comparison['agreements']:
            print(f"    + {agr}")

    print(f"\n  建议: {comparison['recommendation']}")
    print(f"\n报告已保存: {output_path}")
    return report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="保险理赔材料大模型交叉验证 - 双模型独立分析与对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 独立双模型推理:
  python3 cross_validate.py -i ./claims -k sk-xxx
  python3 cross_validate.py -i ./claims -k sk-xxx -p qwen3.5-plus -s kimi-k2.5

  # 复用前置解析结果作为主模型（仅调用次模型）:
  python3 cross_validate.py -i ./claims -k sk-xxx --analysis-result ./analysis.json
"""
    )
    parser.add_argument("--input-dir", "-i", required=True, help="包含理赔图片的源目录")
    parser.add_argument("--api-key", "-k", required=True, help="DashScope API Key")
    parser.add_argument("--primary-model", "-p", default="qwen3.5-plus",
                        help="主模型名称（默认 qwen3.5-plus）")
    parser.add_argument("--secondary-model", "-s", default="kimi-k2.5",
                        help="验证模型名称（默认 kimi-k2.5）")
    parser.add_argument("--output", "-o", default=None, help="JSON报告输出路径")
    parser.add_argument("--analysis-result", "-a", default=None,
                        help="前置解析结果JSON路径（作为主模型结果，仅调用次模型进行验证）")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    result = run_cross_validation(
        input_dir=input_dir,
        api_key=args.api_key,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
        output_path=args.output,
        analysis_result_path=args.analysis_result
    )

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
