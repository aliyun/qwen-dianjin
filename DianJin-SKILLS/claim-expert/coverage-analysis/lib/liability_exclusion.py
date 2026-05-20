# -*- coding: utf-8 -*-
"""Skill: Liability exclusion check for insurance claims.

自包含技能 - 可独立下载和使用，无需其他依赖技能。

流程：
1. 先分类所有图片
2. 只对病历、诊断证明、检查报告类文档做 OCR
3. 用 qwen3.5-plus 推理责免风险
"""

# 支持两种导入方式
try:
    from . import (
        validate_image_paths,
        call_vision_model,
        call_text_model,
        parse_json_response,
        LIABILITY_RELATED_TYPES,
        CLASSIFY_DOCUMENT_PROMPT,
        EXTRACT_CONTENT_PROMPT,
    )
    # Default exclusion clauses
    DEFAULT_EXCLUSION_CLAUSES = """
常见免责条款：
1. 既往症免责
2. 等待期免责
3. 免赔额条款
4. 自费药品/项目免责
5. 酒驾/醉驾免责
6. 故意行为免责
7. 非定点医院免责
8. 未如实告知免责
"""
    LIABILITY_EXCLUSION_PROMPT = """请检查以下内容是否触发免责条款。

病历内容：
{context}

保单条款：
{policy_terms}

请输出：
1. 风险等级（高/中/低）
2. 触发的免责条款
3. 处理建议
"""
except ImportError:
    from ...core.image_utils import validate_image_paths
    from ...core.llm_client import call_vision_model, call_text_model, parse_json_response
    from ...knowledge.document_types import LIABILITY_RELATED_TYPES
    from ...knowledge.policy_rules import DEFAULT_EXCLUSION_CLAUSES
    from ...prompts.templates import CLASSIFY_DOCUMENT_PROMPT, EXTRACT_CONTENT_PROMPT, LIABILITY_EXCLUSION_PROMPT

from agentscope.service import ServiceResponse, ServiceExecStatus

BATCH_SIZE = 3


def _check_exclusion_from_data(
    extractions: list[dict],
    policy_terms: str,
    api_key: str,
    text_model: str,
    verbose: bool = True,
) -> ServiceResponse:
    """Internal: check liability exclusions from pre-computed OCR data."""
    terms = policy_terms if policy_terms else DEFAULT_EXCLUSION_CLAUSES

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
        print("  [3/3] 分析免责条款风险...")

    try:
        result = call_text_model(
            prompt=LIABILITY_EXCLUSION_PROMPT.format(
                context=context, policy_terms=terms,
            ),
            api_key=api_key,
            model=text_model,
        )
    except RuntimeError as e:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"责免事项推理失败: {e}",
        )

    try:
        parsed = parse_json_response(result)
    except ValueError:
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=f"责免事项检查结果:\n{result}",
        )

    lines = ["责免事项检查结果：", ""]
    risk_level = parsed.get("risk_level", "unknown")
    risk_map = {"high": "高风险", "medium": "中风险", "low": "低风险"}
    lines.append(f"风险等级: {risk_map.get(risk_level, risk_level)}")
    lines.append("")
    for finding in parsed.get("findings", []):
        status = "触发" if finding.get("triggered") else "未触发"
        lines.append(f"  - [{status}] {finding.get('clause', '')}")
        lines.append(f"    {finding.get('detail', '')}")
    recommendation = parsed.get("recommendation", "")
    if recommendation:
        lines.append(f"\n建议: {recommendation}")

    return ServiceResponse(
        status=ServiceExecStatus.SUCCESS,
        content="\n".join(lines),
    )


def check_liability_exclusion(
    image_paths: list[str],
    policy_terms: str = "",
    api_key: str = "",
    model: str = "qwen-vl-plus",
    text_model: str = "qwen3.5-plus",
    verbose: bool = True,
) -> ServiceResponse:
    """识别理赔案件中可能触发的保险免责条款。

    流程：先分类 → 筛选病历诊断类文档 → OCR → 推理

    Args:
        image_paths (list[str]): 图片文件路径列表。
        policy_terms (str): 保单条款文本。为空则使用默认通用条款。
        api_key (str): DashScope API 密钥。
        model (str): 视觉模型名称。
        text_model (str): 文本推理模型名称。
        verbose (bool): 是否打印进度。

    Returns:
        ServiceResponse: 包含免责条款检查结果。
    """
    try:
        validated_paths = validate_image_paths(image_paths)
    except (FileNotFoundError, ValueError) as e:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"图片校验失败: {e}",
        )

    # Step 1: 分类
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

    # Step 2: 筛选责免相关文档并 OCR
    liability_indices = []
    liability_paths = []
    for item in all_classifications:
        if item.get("type") in LIABILITY_RELATED_TYPES:
            idx = item.get("image_index", 0)
            if idx < len(validated_paths):
                liability_indices.append(idx)
                liability_paths.append(validated_paths[idx])

    if not liability_paths:
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content="未找到病历诊断类文档，无法进行责免事项检查。",
        )

    if verbose:
        print(f"  [2/3] OCR提取病历诊断类文档 (共 {len(liability_paths)} 张)...")

    all_extractions = []
    for i in range(0, len(liability_paths), BATCH_SIZE):
        batch = liability_paths[i:i + BATCH_SIZE]
        batch_indices = liability_indices[i:i + BATCH_SIZE]
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
    return _check_exclusion_from_data(
        extractions=all_extractions,
        policy_terms=policy_terms,
        api_key=api_key,
        text_model=text_model,
        verbose=verbose,
    )
