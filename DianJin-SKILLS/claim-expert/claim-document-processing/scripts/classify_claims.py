#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保险理赔材料智能分类脚本

基于阿里云百炼（DashScope）多模态大模型进行图片内容识别与自动分类。
支持独立运行，不依赖特定工程目录结构。

用法:
    python3 classify_claims.py --input-dir <输入目录> --api-key <API_KEY> --model <模型>
    python3 classify_claims.py -i <输入目录> -k <API_KEY> -m <模型> -o <输出目录>

参数:
    --input-dir, -i   包含理赔图片的源目录 (必需)
    --output-dir, -o  分类结果输出目录 (可选，默认自动生成)
    --api-key, -k     DashScope API Key (必需)
    --model, -m       模型名称 (可选，默认 qwen-vl-max)

示例:
    python3 classify_claims.py -i ./claims_images -k sk-xxx -m qwen-vl-max
"""

import os
import sys
import shutil
import re
import base64
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径以引用 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import create_output_dir

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("错误: openai 未安装，请运行: pip install openai")
    sys.exit(1)

# ===== 可用模型列表 =====
AVAILABLE_MODELS = [
    "qwen-vl-max",
    "qwen3.5-plus",
    "qwen3.5-397b-a17b",
    "kimi-k2.5",
]

# ===== 分类规则（按优先级排序）=====
CATEGORY_RULES = [
    ("02_鉴定报告", "司法鉴定意见书", ["司法鉴定意见书", "法医学鉴定", "伤残等级鉴定"]),
    ("02_鉴定报告", "伤残鉴定书", ["伤残鉴定", "鉴定意见", "伤残等级", "劳动能力鉴定"]),
    ("03_鉴定费用", "鉴定费发票", ["鉴定费", r"司法鉴定.*发票", r"鉴定.*收费"]),
    ("04_病历资料", "入院记录", ["入院记录", "入院诊断", "入院情况", "入院时间"]),
    ("04_病历资料", "出院记录", ["出院记录", "出院诊断", "出院医嘱", "出院小结"]),
    ("04_病历资料", "疾病证明书", ["疾病证明", "诊断证明", "病情证明"]),
    ("04_病历资料", "诊断报告", ["诊断报告", "检查报告", "影像报告", "CT报告", "MRI", "X线"]),
    ("05_费用清单", "住院费用清单", ["费用清单", "住院费用", "费用明细", "收费明细"]),
    ("05_费用清单", "用药清单", ["用药清单", "药品清单", "处方笺"]),
    ("05_费用清单", "医保结算单", ["医保结算", "医疗保险结算", "社保结算", "基本医疗保险"]),
    ("01_医疗发票", "住院发票", [r"住院.*发票", "住院收费", "住院票据"]),
    ("01_医疗发票", "门诊发票", [r"门诊.*发票", "门诊收费", "门诊票据"]),
    ("01_医疗发票", "电子发票", ["电子发票", r"增值税.*发票", "发票代码", "发票号码"]),
    ("01_医疗发票", "医疗收费票据", ["收费票据", "医疗收费", "门诊收费票据", "住院收费票据"]),
]

# ===== 类型映射表 =====
TYPE_MAPPING = {
    "住院发票": ("01_医疗发票", "住院发票"),
    "门诊发票": ("01_医疗发票", "门诊发票"),
    "电子发票": ("01_医疗发票", "电子发票"),
    "医疗发票": ("01_医疗发票", "医疗发票"),
    "收费票据": ("01_医疗发票", "医疗收费票据"),
    "伤残鉴定": ("02_鉴定报告", "伤残鉴定书"),
    "伤残鉴定书": ("02_鉴定报告", "伤残鉴定书"),
    "司法鉴定": ("02_鉴定报告", "司法鉴定意见书"),
    "司法鉴定意见书": ("02_鉴定报告", "司法鉴定意见书"),
    "鉴定报告": ("02_鉴定报告", "鉴定报告"),
    "鉴定费发票": ("03_鉴定费用", "鉴定费发票"),
    "鉴定费用": ("03_鉴定费用", "鉴定费发票"),
    "入院记录": ("04_病历资料", "入院记录"),
    "出院记录": ("04_病历资料", "出院记录"),
    "诊断报告": ("04_病历资料", "诊断报告"),
    "检查报告": ("04_病历资料", "检查报告"),
    "影像报告": ("04_病历资料", "影像报告"),
    "CT报告": ("04_病历资料", "CT报告"),
    "CT影像": ("04_病历资料", "CT影像"),
    "X线报告": ("04_病历资料", "X线报告"),
    "MRI报告": ("04_病历资料", "MRI报告"),
    "疾病证明": ("04_病历资料", "疾病证明书"),
    "疾病证明书": ("04_病历资料", "疾病证明书"),
    "病历资料": ("04_病历资料", "病历资料"),
    "病历": ("04_病历资料", "病历资料"),
    "住院费用清单": ("05_费用清单", "住院费用清单"),
    "费用清单": ("05_费用清单", "住院费用清单"),
    "用药清单": ("05_费用清单", "用药清单"),
    "医保结算单": ("05_费用清单", "医保结算单"),
    "医保结算": ("05_费用清单", "医保结算单"),
}

# ===== LLM Prompt =====
CLASSIFICATION_PROMPT = """# Role
你是一名专业的保险理赔文档分析专家，擅长 OCR 文字识别与文档分类。

# Task
请分析上传的图片，完成以下任务：
1. **文档分类**：从以下列表中选择最匹配的一项：
   - 医疗发票（含住院/门诊/电子发票/其他发票）
   - 鉴定报告（含伤残鉴定/司法鉴定/资质证明/伤残会审表等）
   - 鉴定费用（鉴定费发票）
   - 病历资料（含入院记录/出院记录/诊断证明/疾病证明）
   - 费用清单（含住院清单/用药清单/医保结算单）
   - 其他材料（非上述材料分类，如辅助说明类材料）
2. **关键信息提取**：提取对理赔审核至关重要的文字内容（如：金额、日期、姓名、诊断结果、发票代码等）。

# Constraints
- 必须且仅输出合法的 JSON 格式，不要包含 markdown 标记（如 ```json）。
- 如果图片模糊无法识别，document_type 返回 "无法识别"，key_text 说明原因。
- 语言统一为简体中文。

# Output Format
{
    "document_type": "文档类型",
    "key_text": "关键文字内容摘要（请包含：总金额、开票日期、患者姓名、医疗机构名称等核心字段）"
}"""


def get_image_mime_type(image_path):
    """根据文件扩展名获取MIME类型"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def encode_image_to_base64(image_path):
    """将本地图片编码为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_image(client, model_name, image_path):
    """使用多模态大模型识别图片内容，返回(text, token_usage)"""
    try:
        image_base64 = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": CLASSIFICATION_PROMPT
                    }
                ]
            }],
            max_tokens=500
        )

        result = completion.choices[0].message.content
        token_usage = {
            "input": completion.usage.prompt_tokens if completion.usage else 0,
            "output": completion.usage.completion_tokens if completion.usage else 0,
            "total": completion.usage.total_tokens if completion.usage else 0,
        }
        return result, token_usage

    except Exception as e:
        print(f"  [ERROR] 大模型识别失败: {e}")
        return "", {"input": 0, "output": 0, "total": 0}


def refine_medical_record(key_text):
    """对病历资料进行细分（仅使用高置信度关键词，避免模型输出非确定性导致误判）"""
    # 强特征关键词检测
    has_certificate = re.search(r'疾病证明|诊断证明|病情证明', key_text)
    has_discharge = re.search(r'出院记录|出院诊断|出院医嘱|出院小结|出院情况', key_text)
    has_admission = re.search(r'入院记录|入院诊断|入院情况|入院时间|主诉.*入院', key_text)

    # 优先级：疾病证明 > 出院 > 入院（疾病证明书通常也提及入院信息，需优先匹配）
    if has_certificate:
        return "04_病历资料", "疾病证明书"
    if has_discharge:
        return "04_病历资料", "出院记录"
    if has_admission:
        return "04_病历资料", "入院记录"
    # 影像和检查报告
    if re.search(r'CT.*报告|CT.*检查|CT.*影像|CT.*胶片', key_text, re.IGNORECASE):
        return "04_病历资料", "CT影像报告"
    elif re.search(r'MRI|磁共振', key_text, re.IGNORECASE):
        return "04_病历资料", "MRI报告"
    elif re.search(r'X线|X光|DR报告', key_text, re.IGNORECASE):
        return "04_病历资料", "X线报告"
    elif re.search(r'影像|胶片', key_text):
        return "04_病历资料", "影像资料"
    elif re.search(r'检查报告|检验报告|化验', key_text):
        return "04_病历资料", "检查报告"
    elif re.search(r'诊断报告', key_text):
        return "04_病历资料", "诊断报告"
    elif re.search(r'手术记录|手术', key_text):
        return "04_病历资料", "手术记录"
    elif re.search(r'病程记录|病程', key_text):
        return "04_病历资料", "病程记录"
    return None


def classify_by_content(text):
    """根据大模型返回内容进行分类（三层判定）"""
    if not text:
        return "06_其他材料", "未识别文件"

    combined_text = text

    # 第一层：JSON解析 + 类型映射
    try:
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            result = json.loads(json_match.group())
            doc_type = result.get("document_type", "")
            key_text = result.get("key_text", "")

            # 检测 doc_type 是否为通用分类描述（模型回传了 Prompt 中的类别说明）
            is_generic_echo = "（含" in doc_type

            # 病历资料细分
            if "04_病历资料" in str(TYPE_MAPPING.get(doc_type, "")) or "病历" in doc_type:
                refined = refine_medical_record(key_text)
                if refined:
                    return refined
                # 如果细分失败且是通用描述，返回病历资料大类（避免从通用描述中误匹配子类型）
                if is_generic_echo:
                    return "04_病历资料", "病历资料"

            # 类型映射匹配（病历资料已在上方返回，此处处理其他分类）
            for key, value in TYPE_MAPPING.items():
                if key in doc_type:
                    return value

            combined_text = f"{doc_type} {key_text}"
    except (json.JSONDecodeError, AttributeError):
        combined_text = text

    # 第二层：CATEGORY_RULES 正则匹配
    for category, desc_name, keywords in CATEGORY_RULES:
        for keyword in keywords:
            if re.search(keyword, combined_text, re.IGNORECASE):
                if category == "04_病历资料":
                    if re.search(r'疾病证明|诊断证明|病情证明', combined_text):
                        return "04_病历资料", "疾病证明书"
                    elif re.search(r'入院记录|入院诊断|入院情况', combined_text):
                        return "04_病历资料", "入院记录"
                    elif re.search(r'出院记录|出院诊断|出院医嘱|出院小结', combined_text):
                        return "04_病历资料", "出院记录"
                    elif re.search(r'CT', combined_text, re.IGNORECASE):
                        return "04_病历资料", "CT影像报告"
                return category, desc_name

    # 第三层：通用兜底
    if re.search(r'发票|票据|收费', combined_text):
        return "01_医疗发票", "医疗票据"
    if re.search(r'鉴定|评定', combined_text):
        return "02_鉴定报告", "鉴定材料"
    if re.search(r'病历|诊断|医院|科室', combined_text):
        if re.search(r'入院', combined_text):
            return "04_病历资料", "入院记录"
        elif re.search(r'出院', combined_text):
            return "04_病历资料", "出院记录"
        elif re.search(r'CT', combined_text, re.IGNORECASE):
            return "04_病历资料", "CT影像报告"
        return "04_病历资料", "病历资料"
    if re.search(r'清单|明细|费用', combined_text):
        return "05_费用清单", "费用材料"

    return "06_其他材料", "其他材料"


def process_directory(client, model_name, base_src, base_dst, dir_name):
    """处理单个目录下的所有图片文件"""
    if dir_name == "":
        src_dir = base_src
        dst_base = base_dst
    else:
        src_dir = os.path.join(base_src, dir_name)
        dst_base = os.path.join(base_dst, dir_name)

    category_counters = {}
    results = []

    # 支持的图片格式
    supported_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    files = sorted([
        f for f in os.listdir(src_dir)
        if not f.startswith('.') and os.path.isfile(os.path.join(src_dir, f))
        and os.path.splitext(f)[1].lower() in supported_ext
    ])

    total_files = len(files)
    if total_files == 0:
        print(f"  [WARN] 目录中无可处理的图片文件: {src_dir}")
        return results

    for idx, filename in enumerate(files, 1):
        src_path = os.path.join(src_dir, filename)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        print(f"  [{idx}/{total_files}] 分析: {filename}")
        start_time = time.time()

        # 调用大模型识别
        text_content, token_usage = extract_text_from_image(client, model_name, src_path)
        elapsed_time = time.time() - start_time

        if text_content:
            preview = text_content[:80].replace('\n', ' ')
            print(f"    识别: {preview}...")
        else:
            print(f"    识别: (无法识别)")

        print(f"    耗时: {elapsed_time:.2f}s | Token: {token_usage['input']}/{token_usage['output']}/{token_usage['total']}")

        # 分类
        category, desc_name = classify_by_content(text_content)
        print(f"    分类: {category} -> {desc_name}")

        # 计数与命名
        key = (category, desc_name)
        category_counters[key] = category_counters.get(key, 0) + 1
        seq = category_counters[key]
        new_filename = f"{category.split('_')[0]}_{desc_name}_{seq:03d}{ext}"

        # 复制文件到目标目录
        dst_dir = os.path.join(dst_base, category)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, new_filename)
        shutil.copy2(src_path, dst_path)

        results.append({
            "dir_name": dir_name,
            "original": filename,
            "category": category,
            "desc_name": desc_name,
            "new_name": new_filename,
            "elapsed_time": elapsed_time,
            "token_input": token_usage["input"],
            "token_output": token_usage["output"],
            "token_total": token_usage["total"],
            "model_raw_result": text_content or "",
        })

    return results


def generate_report(all_results, report_path, model_name, timestamp):
    """生成Markdown分析报告"""
    total_time_ms = sum(r["elapsed_time"] * 1000 for r in all_results)
    avg_time_ms = total_time_ms / len(all_results) if all_results else 0
    total_input = sum(r["token_input"] for r in all_results)
    total_output = sum(r["token_output"] for r in all_results)
    total_tokens = sum(r["token_total"] for r in all_results)

    # 统计各分类数量
    category_stats = {}
    for r in all_results:
        cat = r["category"]
        category_stats[cat] = category_stats.get(cat, 0) + 1

    md = f"""# 保险理赔材料分类分析报告

## 基本信息

| 项目 | 值 |
|------|----|
| 分析模型 | {model_name} |
| 执行时间 | {timestamp} |
| 文件总数 | {len(all_results)} |
| 总耗时 | {total_time_ms:.0f} ms |
| 平均耗时 | {avg_time_ms:.0f} ms/文件 |

## Token消耗统计

| 类型 | 数量 |
|------|------|
| Input Tokens | {total_input} |
| Output Tokens | {total_output} |
| Total Tokens | {total_tokens} |
| 平均/文件 | {total_tokens // len(all_results) if all_results else 0} |

## 分类统计

| 分类 | 文件数量 |
|------|----------|
"""
    for cat, count in sorted(category_stats.items()):
        md += f"| {cat} | {count} |\n"

    md += f"""
## 文件分析明细

| 序号 | 目录 | 原文件名 | 目标文件名 | 分类 | 文件类型 | 耗时(ms) | Input | Output | Total | 模型识别结果 |
|------|------|----------|------------|------|----------|----------|-------|--------|-------|------------|
"""
    for idx, r in enumerate(all_results, 1):
        model_result = r["model_raw_result"].replace("\n", " ").replace("|", "\\|")[:200]
        md += (
            f"| {idx} | {r['dir_name']} | {r['original']} | {r['new_name']} "
            f"| {r['category']} | {r['desc_name']} | {r['elapsed_time']*1000:.0f} "
            f"| {r['token_input']} | {r['token_output']} | {r['token_total']} "
            f"| {model_result} |\n"
        )

    md += f"\n---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n报告已保存: {report_path}")
    return report_path


def classify_from_analysis_result(analysis_path, input_dir, output_dir):
    """
    基于前置解析结果进行文件归档（无需再次调用大模型）

    Args:
        analysis_path: analyze_materials.py 输出的 JSON 路径
        input_dir: 原始材料目录
        output_dir: 归档输出目录
    """
    print(f"  使用前置解析结果: {analysis_path}")

    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    documents = analysis.get("documents", [])
    if not documents:
        print("  [ERROR] 前置解析结果中无文档信息")
        return []

    # 从 file_list 获取文件名映射（file_list 中是实际文件系统名称，模型可能修改文件名）
    file_list = analysis.get("file_list", [])

    # 支持的图片格式
    supported_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    src_files = sorted([
        f for f in os.listdir(input_dir)
        if not f.startswith('.') and os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in supported_ext
    ])

    # 类别映射：将 analysis 中的英文 category 映射回中文目录名
    CATEGORY_DIR_MAP = {
        "medical_invoice": "01_医疗发票",
        "appraisal_report": "02_鉴定报告",
        "appraisal_fee": "03_鉴定费用",
        "medical_record": "04_病历资料",
        "expense_list": "05_费用清单",
        "other": "06_其他材料",
    }

    category_counters = {}
    results = []

    for doc in documents:
        idx = doc.get("image_index", 0)
        category_en = doc.get("category", "other")
        sub_type = doc.get("sub_type", "")

        # 获取实际文件名：优先从 file_list（实际文件系统名称），再 fallback 到 src_files
        filename = ""
        if idx < len(file_list):
            filename = file_list[idx].get("file_name", "")
        if not filename and idx < len(src_files):
            filename = src_files[idx]

        if not filename:
            continue

        src_path = os.path.join(input_dir, filename)
        if not os.path.isfile(src_path):
            print(f"    [WARN] 文件不存在: {src_path}")
            continue

        # 确定分类目录和描述名
        category_dir = CATEGORY_DIR_MAP.get(category_en, "06_其他材料")
        desc_name = sub_type if sub_type else category_en

        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # 计数与命名
        key = (category_dir, desc_name)
        category_counters[key] = category_counters.get(key, 0) + 1
        seq = category_counters[key]
        new_filename = f"{category_dir.split('_')[0]}_{desc_name}_{seq:03d}{ext}"

        # 复制文件
        dst_dir = os.path.join(output_dir, category_dir)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, new_filename)
        shutil.copy2(src_path, dst_path)

        print(f"    [{idx}] {filename} -> {category_dir}/{new_filename}")

        results.append({
            "dir_name": "",
            "original": filename,
            "category": category_dir,
            "desc_name": desc_name,
            "new_name": new_filename,
            "elapsed_time": 0,
            "token_input": 0,
            "token_output": 0,
            "token_total": 0,
            "model_raw_result": f"[from analysis] category={category_en}, sub_type={sub_type}",
        })

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="保险理赔材料智能分类 - 基于多模态大模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python3 classify_claims.py -i ./claims -k sk-xxx
  python3 classify_claims.py -i ./claims -k sk-xxx -m qwen3.5-plus -o ./output

  # 复用前置解析结果（无需再次调用大模型）:
  python3 classify_claims.py -i ./claims -k sk-xxx --analysis-result ./analysis.json

可用模型: {', '.join(AVAILABLE_MODELS)}
"""
    )
    parser.add_argument("--input-dir", "-i", required=True, help="包含理赔图片的源目录")
    parser.add_argument("--output-dir", "-o", default=None, help="输出目录（默认自动生成）")
    parser.add_argument("--api-key", "-k", required=True, help="DashScope API Key")
    parser.add_argument("--model", "-m", default="qwen-vl-max", help="模型名称（默认 qwen-vl-max）")
    parser.add_argument("--analysis-result", "-a", default=None,
                        help="前置解析结果JSON路径（若提供则跳过大模型调用，直接归档）")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    model_name = args.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = create_output_dir("classify")

    # 验证输入目录
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("保险理赔材料智能分类")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_start = time.time()

    # ===== 模式判断：复用前置解析 vs 独立推理 =====
    if args.analysis_result:
        # 复用前置解析结果，无需调用大模型
        print(f"模式: 复用前置解析结果")
        print(f"解析文件: {args.analysis_result}")
        print("=" * 60)

        results = classify_from_analysis_result(args.analysis_result, input_dir, output_dir)
        all_results.extend(results)
    else:
        # 原有逻辑：逐文件调用大模型
        print(f"模式: 独立大模型推理")
        print(f"模型: {model_name}")
        print(f"API Key: {args.api_key[:8]}...{args.api_key[-4:]}")
        print("=" * 60)

        # 验证模型
        if model_name not in AVAILABLE_MODELS:
            print(f"警告: 模型 '{model_name}' 不在预定义列表中，仍将尝试运行")

        client = OpenAI(
            api_key=args.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print("大模型客户端初始化完成\n")

        # 检测目录结构
        subdirs = sorted([
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
        ])

        supported_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        root_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and not f.startswith(".")
            and os.path.splitext(f)[1].lower() in supported_ext
        ]

        if not subdirs and root_files:
            print(f"检测到 {len(root_files)} 个图片文件（无子目录）\n")
            results = process_directory(client, model_name, input_dir, output_dir, "")
            all_results.extend(results)
        elif subdirs:
            print(f"检测到 {len(subdirs)} 个子目录:\n")
            for d in subdirs:
                print(f"{'='*60}")
                print(f"处理目录: {d}")
                print(f"{'='*60}")
                results = process_directory(client, model_name, input_dir, output_dir, d)
                all_results.extend(results)
        else:
            print("错误: 输入目录中未找到可处理的图片文件")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    # 生成报告
    report_path = os.path.join(output_dir, "report.md")
    generate_report(all_results, report_path, model_name, timestamp)

    # 汇总
    print(f"\n{'='*60}")
    print("分类完成")
    print(f"{'='*60}")
    print(f"处理文件: {len(all_results)} 个")
    print(f"总耗时: {total_elapsed:.1f} 秒")
    if all_results:
        print(f"平均: {total_elapsed/len(all_results):.1f} 秒/文件")
        total_tokens = sum(r["token_total"] for r in all_results)
        if total_tokens > 0:
            print(f"Token消耗: {total_tokens}")

    # 分类统计
    category_stats = {}
    for r in all_results:
        cat = r["category"]
        category_stats[cat] = category_stats.get(cat, 0) + 1

    print(f"\n分类分布:")
    for cat, count in sorted(category_stats.items()):
        print(f"  {cat}: {count} 个")

    print(f"\n输出目录: {output_dir}")
    print(f"分析报告: {report_path}")


if __name__ == "__main__":
    main()
