#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保险理赔材料前置解析脚本

通过一次多模态大模型调用，对所有理赔材料图片进行综合分析，输出包含：
- 材料分类
- 关键信息提取
- 完整性检查
- 跨图一致性校验
- 风险预警

后续的分类归档、完整性检查、一致性检查、交叉验证脚本均可复用本脚本的输出结果，
避免重复推理消耗 Token。

用法:
    python3 analyze_materials.py --input-dir <输入目录> --api-key <API_KEY>
    python3 analyze_materials.py -i <目录> -k <KEY> -m qwen3.5-plus -o <输出路径>

参数:
    --input-dir, -i    包含理赔图片的源目录 (必需)
    --api-key, -k      DashScope API Key (必需)
    --model, -m        模型名称 (可选，默认 qwen3.5-plus)
    --output, -o       JSON结果输出路径 (可选，默认自动生成)
    --claim-type, -t   案件类型 (可选，默认 "通用医疗")
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_directory, get_file_info_list,
    call_qwen_multimodal, extract_json_from_text,
    save_json_report, print_separator,
    generate_audit_log, save_audit_log,
    create_output_dir
)


# ===== 综合材料分析系统提示词（复用 AICompete 的核心逻辑）=====
MATERIAL_ANALYSIS_SYSTEM_PROMPT = """# Role Definition
你是一名资深保险理赔智能审核专家。你具备多模态分析能力，能够精准识别、分类并提取用户上传的理赔材料（如医疗发票、病历、鉴定报告等）中的关键信息。你的核心任务是执行严格的质量评估、逻辑校验和重复性检测，最终输出符合程序解析要求的结构化 JSON 审核报告。

# Constraints & Rules
在执行任务时，必须严格遵守以下原则：
真实性原则 (Grounding)：所有提取信息必须严格基于图片可见内容。若因模糊、遮挡导致无法识别，对应字段必须填 null，并在 risk_warnings 中标注，严禁猜测、臆造或产生幻觉。
预警优先 (Alert First)：对于清晰度低、核心材料缺失、逻辑冲突（如时间线倒置）、疑似重复材料等情况，必须生成明确的预警信息。
格式规范 (Strict JSON)：
    最终输出必须是合法的 JSON 格式。
    禁止包含 Markdown 代码块标记（如 ```json ... ```）。
    禁止在 JSON 前后添加任何解释性文字。
    确保输出可直接被后端程序解析。

# Workflow
请按以下步骤逐步思考并执行分析：

Step 1: 单图分析与质量评估
对每一张输入图片独立执行：
清晰度检测：判断是否存在模糊、反光、缺角。若影响关键信息读取，标记 quality_status 为 low。
材料分类：从以下标准体系中选择最匹配的一项（注意边界情况）：
    medical_invoice: 医疗发票（含住院/门诊/电子发票）
       【重要】必须是正式的税务发票，有发票代码、发票号码、税号等要素
       【排除】收费凭证、收据、缴费条、预交金凭证等非正式发票不属于此类，应归为 other
    appraisal_report: 鉴定报告（含伤残/司法/资质鉴定）
       【包含】鉴定书、鉴定意见书、司法鉴定许可证、鉴定机构资质证明等
       【注意】只要是鉴定机构出具的正式文件（包括资质证明），都应归为此类
    appraisal_fee: 鉴定费用发票
    medical_record: 病历资料（含入/出院记录、诊断证明、疾病证明、检查报告）
    expense_list: 费用清单（含住院/用药明细、医保结算单、费用汇总清单）
    other: 其他无法归类的材料（如收费凭证、收据、说明文件等）
基础信息提取：提取患者姓名、证件号、票据/报告编号、日期、金额、诊断结果、医院名称、起止时间。
详细信息提取：根据分类提取特有字段：
   发票类：票据代码、交款人、开票日期、医保类型、大小写金额等
   鉴定类：鉴定机构、结论（伤残等级等）、三期天数
   病历类：入出院诊断、主诉、治疗经过、医嘱等
   清单类：费用项目明细（每项含 item_name、amount）

Step 2: 完整性与重复性校验
类型完整性：基于 claim_type（{{claim_type}}），判断核心材料链是否缺失（例：有发票通常应配套病历和费用清单）。
页面完整性：检查多页文档页码标识。若无明显页码但内容中断，推测"疑似缺页"并在预警中说明。
重复材料识别：
    发票类：比对"票据代码+号码+金额"。完全一致判为 high 置信度重复。
    病历/证明类：OCR文本相似度≥95%且关键字段一致判为 high。
    费用清单类：比对"住院号+合计+时间范围"。
    边界处理：同一住院多次结算、连号发票等不判定为重复。同一发票纸质版/电子版判定为重复。

Step 3: 跨图逻辑关联 (Cross-Validation)
一致性校验：比对所有图片中的患者姓名、身份证号是否完全一致。
时间线校验：验证就诊逻辑（入院 < 出院 ≤ 发票开具日期）。
费用逻辑校验：比对发票总金额与费用清单总额是否匹配，医疗行为与诊断是否相关。

Step 4: 生成报告
将上述分析结果整合为指定的 JSON 结构。

# Output JSON Schema
请严格按照以下 JSON 结构输出，不要遗漏任何字段：

{
  "case_summary": {
    "total_images": 0,
    "overall_quality": "high | medium | low",
    "claim_type_detected": "string"
  },
  "documents": [
    {
      "image_index": 0,
      "filename": "string",
      "category": "string",
      "sub_type": "string (细分类型，如入院记录/出院记录/住院发票等)",
      "quality_status": "high | low",
      "extraction": {
        "patient_name": "string | null",
        "id_number": "string | null",
        "document_number": "string | null",
        "date": "YYYY-MM-DD | null",
        "amount": null,
        "diagnosis": "string | null",
        "hospital_name": "string | null",
        "time_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
      },
      "detailed_extraction": {}
    }
  ],
  "completeness_check": {
    "is_complete": true,
    "existing_types": ["string"],
    "missing_types": ["string"],
    "missing_pages_warning": "string | null",
    "duplicate_materials": [
      {
        "image_indices": [0, 1],
        "filenames": ["string"],
        "reason": "string",
        "document_type": "string",
        "confidence": "high | medium | low"
      }
    ]
  },
  "cross_validation": {
    "name_consistent": true,
    "id_consistent": true,
    "timeline_consistent": true,
    "amount_consistent": true,
    "logic_summary": "string",
    "inconsistency_details": ["string"]
  },
  "risk_warnings": [
    {
      "level": "high | medium | low",
      "type": "quality | completeness | consistency | duplicate | data_extraction",
      "message": "string"
    }
  ]
}

请开始分析，直接输出 JSON 结果。"""


def build_user_prompt(file_info: list, claim_type: str) -> str:
    """构建用户提示词"""
    prompt = f"案件类型: {claim_type}\n\n请分析以下理赔材料图片：\n\n"
    for i, info in enumerate(file_info):
        prompt += f"图片{i}: {info['file_name']}"
        if info['file_type'] == 'pdf':
            prompt += f" (PDF第{info['page_index'] + 1}页)"
        prompt += f", 尺寸: {info['size']}\n"
    prompt += "\n请按照系统提示中的JSON格式返回完整的分析结果。"
    return prompt


def run_analysis(
    input_dir: str,
    api_key: str,
    model: str = "qwen3.5-plus",
    output_path: str = None,
    claim_type: str = "通用医疗"
) -> dict:
    """
    执行前置材料综合解析

    Args:
        input_dir: 输入目录
        api_key: DashScope API Key
        model: 模型名称
        output_path: 输出JSON路径
        claim_type: 案件类型

    Returns:
        综合分析结果字典
    """
    print_separator("保险理赔材料 - 前置综合解析")
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
    print(f"\n[2/3] 调用模型进行综合分析 ({model})...")
    file_info = get_file_info_list(loaded_files)
    images = [f.base64_data for f in loaded_files]

    system_prompt = MATERIAL_ANALYSIS_SYSTEM_PROMPT.replace("{{claim_type}}", claim_type)
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
    print("\n[3/3] 解析分析结果...")
    result = extract_json_from_text(response_text)
    if not result:
        print("  [WARN] JSON解析失败，保存原始响应")
        result = {"raw_response": response_text, "documents": []}

    # 组装标准化输出
    report = {
        "success": True,
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input_dir": input_dir,
        "model": model,
        "claim_type": claim_type,
        "file_stats": stats,
        "file_list": file_info,
        "analysis_metadata": {
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "total_tokens": metadata.get("total_tokens", 0),
            "duration_ms": metadata.get("duration_ms", 0)
        },
        # 核心结果
        "case_summary": result.get("case_summary", {}),
        "documents": result.get("documents", []),
        "completeness_check": result.get("completeness_check", {}),
        "cross_validation": result.get("cross_validation", {}),
        "duplicate_materials": result.get("duplicate_materials", []),
        "risk_warnings": result.get("risk_warnings", []),
        # 审计元数据
        "_metadata": {
            "skill_version": "2.0.0",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "input_file_count": stats.get("total_files", 0),
            "token_usage": {
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "total_tokens": metadata.get("total_tokens", 0)
            },
            "duration_seconds": round(elapsed, 2)
        },
        # 风险提示
        "risk_disclosure": {
            "disclaimer": "本审核结果由AI辅助生成，仅供理赔人员参考，最终决定须由授权人员确认",
            "model_confidence": "基于单次多模态推理，建议使用交叉验证提升可信度",
            "data_freshness": datetime.now().isoformat()
        }
    }

    # 保存
    if not output_path:
        output_dir = create_output_dir("analyze")
        output_path = os.path.join(output_dir, "analysis.json")
    else:
        output_dir = os.path.dirname(output_path) or '.'

    save_json_report(report, output_path)

    # 生成审计日志
    try:
        risk_flags = [w.get("type", "") for w in report.get("risk_warnings", [])
                      if w.get("severity") in ("high", "HIGH")]
        audit_record = generate_audit_log(
            skill_version="2.0.0",
            input_dir=input_dir,
            file_count=stats.get("total_files", 0),
            model=model,
            capabilities=["analyze"],
            steps_executed=["load_files", "model_inference", "parse_result"],
            total_tokens=metadata.get("total_tokens", 0),
            duration_seconds=elapsed,
            errors=[],
            confidence_scores={"analysis": 0.9},
            risk_flags=risk_flags,
            output_path=output_path
        )
        audit_path = save_audit_log(audit_record, audit_dir=output_dir)
        print(f"  审计日志: {audit_path}")
    except Exception as e:
        print(f"  [WARN] 审计日志生成失败: {e}")

    # 输出摘要
    print_separator("解析结果摘要")
    docs = report.get("documents", [])
    print(f"  文档数量: {len(docs)}")

    # 分类统计
    cat_counts = {}
    for doc in docs:
        cat = doc.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat_counts:
        print(f"  分类分布:")
        for cat, cnt in sorted(cat_counts.items()):
            print(f"    {cat}: {cnt}")

    # 完整性
    cc = report.get("completeness_check", {})
    is_complete = cc.get("is_complete", "未知")
    missing = cc.get("missing_types", [])
    dups = cc.get("duplicate_materials", [])
    print(f"  完整性: {'完整' if is_complete else '不完整'}")
    if missing:
        print(f"  缺失: {', '.join(missing)}")
    if dups:
        print(f"  重复: {len(dups)} 组")

    # 一致性
    cv = report.get("cross_validation", {})
    name_ok = cv.get("name_consistent", True)
    time_ok = cv.get("timeline_consistent", True)
    print(f"  一致性: 姓名{'一致' if name_ok else '不一致'} | 时间线{'一致' if time_ok else '不一致'}")

    # 预警
    warnings = report.get("risk_warnings", [])
    if warnings:
        print(f"  风险预警: {len(warnings)} 条")
        for w in warnings[:5]:
            print(f"    [{w.get('level', '?')}] {w.get('message', '')}")
        if len(warnings) > 5:
            print(f"    ... 还有 {len(warnings) - 5} 条")

    print(f"\n解析结果已保存: {output_path}")
    print(f"后续脚本可通过 --analysis-result {output_path} 复用此结果")
    return report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="保险理赔材料前置综合解析 - 一次推理，多处复用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 analyze_materials.py -i ./claims -k sk-xxx
  python3 analyze_materials.py -i ./claims -k sk-xxx -m qwen3.5-plus -t "交通事故"

后续复用:
  python3 classify_claims.py -i ./claims -k sk-xxx --analysis-result ./claims_analysis_xxx.json
  python3 check_completeness.py --analysis-result ./claims_analysis_xxx.json
  python3 check_consistency.py --analysis-result ./claims_analysis_xxx.json
  python3 cross_validate.py -i ./claims -k sk-xxx --analysis-result ./claims_analysis_xxx.json
"""
    )
    parser.add_argument("--input-dir", "-i", required=True, help="包含理赔图片的源目录")
    parser.add_argument("--api-key", "-k", required=True, help="DashScope API Key")
    parser.add_argument("--model", "-m", default="qwen3.5-plus",
                        help="模型名称（默认 qwen3.5-plus）")
    parser.add_argument("--output", "-o", default=None, help="JSON结果输出路径")
    parser.add_argument("--claim-type", "-t", default="通用医疗",
                        help="案件类型（默认 '通用医疗'）")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    result = run_analysis(
        input_dir=input_dir,
        api_key=args.api_key,
        model=args.model,
        output_path=args.output,
        claim_type=args.claim_type
    )

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
