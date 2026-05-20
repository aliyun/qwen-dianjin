# -*- coding: utf-8 -*-
"""Skill: Drug indication check for insurance claims.

自包含技能 - 可独立下载和使用，无需其他依赖技能。

流程：
1. 先分类所有图片
2. 只对处方、病历、诊断证明类文档做 OCR
3. 用 qwen3.5-plus 推理药品与诊断匹配度
"""

# 支持两种导入方式
try:
    from . import (
        validate_image_paths,
        call_vision_model,
        call_text_model,
        parse_json_response,
        DRUG_RELATED_TYPES,
        CLASSIFY_DOCUMENT_PROMPT,
        EXTRACT_CONTENT_PROMPT,
    )
    DRUG_INDICATION_PROMPT = """请检查以下药品是否与诊断匹配。

药品和诊断信息：
{context}

请输出：
1. 诊断列表
2. 每种药品的匹配分析
3. 综合评估
"""
except ImportError:
    from ...core.image_utils import validate_image_paths
    from ...core.llm_client import call_vision_model, call_text_model, parse_json_response
    from ...knowledge.document_types import DRUG_RELATED_TYPES
    from ...prompts.templates import CLASSIFY_DOCUMENT_PROMPT, EXTRACT_CONTENT_PROMPT, DRUG_INDICATION_PROMPT

from agentscope.service import ServiceResponse, ServiceExecStatus

BATCH_SIZE = 3


def _check_drug_from_data(
    extractions: list[dict],
    api_key: str,
    text_model: str,
    verbose: bool = True,
) -> ServiceResponse:
    """Internal: check drug indications from pre-computed OCR data."""
    context_parts = []
    for item in extractions:
        fields = item.get("fields", {})
        raw_text = item.get("raw_text", "")
        doc_type = item.get("doc_type", "未知")
        part = f"【{doc_type}】\n"
        for k, v in fields.items():
            if v:
                part += f"  {k}: {v}\n"
        if raw_text:
            part += f"  摘要: {raw_text}\n"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)

    if verbose:
        print("  [3/3] 分析药品与诊断匹配度...")

    try:
        result = call_text_model(
            prompt=DRUG_INDICATION_PROMPT.format(context=context),
            api_key=api_key,
            model=text_model,
        )
    except RuntimeError as e:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"药品适用症推理失败: {e}",
        )

    try:
        parsed = parse_json_response(result)
    except ValueError:
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=f"药品适用症检查结果:\n{result}",
        )

    lines = ["药品适用症检查结果：", ""]
    diagnosis = parsed.get("diagnosis", [])
    lines.append(f"诊断: {', '.join(diagnosis)}")
    lines.append("")
    for drug in parsed.get("drugs", []):
        match_str = "匹配" if drug.get("match") else "不匹配"
        lines.append(f"  - {drug.get('name', '未知')}: {match_str}")
        lines.append(f"    理由: {drug.get('reason', '')}")
    overall = parsed.get("overall_assessment", "")
    if overall:
        lines.append(f"\n综合评估: {overall}")

    return ServiceResponse(
        status=ServiceExecStatus.SUCCESS,
        content="\n".join(lines),
    )


def check_drug_indication(
    image_paths: list[str],
    api_key: str = "",
    model: str = "qwen-vl-plus",
    text_model: str = "qwen3.5-plus",
    verbose: bool = True,
) -> ServiceResponse:
    """检查理赔材料中处方药品是否与诊断匹配，识别超范围用药。

    流程：先分类 → 筛选药品相关文档 → OCR → 推理

    Args:
        image_paths (list[str]): 图片文件路径列表。
        api_key (str): DashScope API 密钥。
        model (str): 视觉模型名称。
        text_model (str): 文本推理模型名称。
        verbose (bool): 是否打印进度。

    Returns:
        ServiceResponse: 包含药品匹配结果。
    """
    try:
        validated_paths = validate_image_paths(image_paths)
    except (FileNotFoundError, ValueError) as e:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"图片校验失败: {e}",
        )

    # Step 1: 分类所有图片
    if verbose:
        print(f"  [1/3] 单证分类 (共 {len(validated_paths)} 张)...")

    all_classifications = []
    for i in range(0, len(validated_paths), BATCH_SIZE):
        batch = validated_paths[i:i + BATCH_SIZE]
        try:
            classify_raw = call_vision_model(
                image_paths=batch,
                prompt=CLASSIFY_DOCUMENT_PROMPT,
                api_key=api_key,
                model=model,
            )
            batch_classifications = parse_json_response(classify_raw)
            for item in batch_classifications:
                item["image_index"] = item.get("image_index", 0) + i
            all_classifications.extend(batch_classifications)
        except (RuntimeError, ValueError) as e:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"单证分类失败: {e}",
            )

    # Step 2: 筛选药品相关文档并 OCR
    drug_related_indices = []
    drug_related_paths = []
    for item in all_classifications:
        if item.get("type") in DRUG_RELATED_TYPES:
            idx = item.get("image_index", 0)
            if idx < len(validated_paths):
                drug_related_indices.append(idx)
                drug_related_paths.append(validated_paths[idx])

    if not drug_related_paths:
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content="未找到药品相关文档（处方、病历、诊断证明），无法进行药品适用症检查。",
        )

    if verbose:
        print(f"  [2/3] OCR提取药品相关文档 (共 {len(drug_related_paths)} 张)...")

    all_extractions = []
    for i in range(0, len(drug_related_paths), BATCH_SIZE):
        batch = drug_related_paths[i:i + BATCH_SIZE]
        batch_indices = drug_related_indices[i:i + BATCH_SIZE]
        try:
            ocr_raw = call_vision_model(
                image_paths=batch,
                prompt=EXTRACT_CONTENT_PROMPT,
                api_key=api_key,
                model=model,
            )
            batch_extractions = parse_json_response(ocr_raw)
            for j, item in enumerate(batch_extractions):
                item["image_index"] = batch_indices[j] if j < len(batch_indices) else i + j
                for c in all_classifications:
                    if c.get("image_index") == item["image_index"]:
                        item["doc_type"] = c.get("type", "未知")
                        break
            all_extractions.extend(batch_extractions)
        except (RuntimeError, ValueError) as e:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"OCR提取失败: {e}",
            )

    # Step 3: 推理
    return _check_drug_from_data(
        extractions=all_extractions,
        api_key=api_key,
        text_model=text_model,
        verbose=verbose,
    )
