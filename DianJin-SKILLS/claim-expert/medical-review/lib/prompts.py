# -*- coding: utf-8 -*-
"""Prompt templates for claim processing skills."""

# Document classification prompt
CLASSIFY_DOCUMENT_PROMPT = """请分析以下图片，识别每张图片属于哪种单证类型。

支持的单证类型：
- 门诊病历
- 住院病历
- 诊断证明
- 费用清单
- 发票
- 身份证
- 银行卡
- 出院小结
- 手术记录
- 检查报告
- 处方
- 检验报告
- 住院费用汇总
- 门诊收费票据
- 病理报告
- 影像报告

请以JSON格式输出，格式如下：
```json
[
  {"image_index": 0, "type": "单证类型", "confidence": 0.95},
  {"image_index": 1, "type": "单证类型", "confidence": 0.92}
]
```

注意：
1. image_index 从0开始，按图片顺序编号
2. confidence 为置信度，范围0-1
3. 如果无法识别类型，使用"其他"
"""

# Content extraction prompt
EXTRACT_CONTENT_PROMPT = """请提取以下图片中的文字内容，并结构化输出关键信息。

请以JSON格式输出，格式如下：
```json
[
  {
    "image_index": 0,
    "fields": {
      "患者姓名": "",
      "性别": "",
      "年龄": "",
      "就诊日期": "",
      "医院名称": "",
      "科室": "",
      "诊断": "",
      "主诉": "",
      "治疗方案": "",
      "药品清单": "",
      "费用总额": "",
      "费用明细": ""
    },
    "raw_text": "完整的OCR识别文本"
  }
]
```

注意：
1. 根据文档类型提取相关字段，不存在的字段留空
2. raw_text 保留完整的OCR识别结果
3. 尽量保持原始格式和内容
"""

# Material completeness check prompt (used internally)
MATERIAL_CHECK_PROMPT = """根据以下分类结果，检查理赔材料是否完整。

已有材料：
{materials}

理赔类型：{claim_type}

必需材料：
{required_materials}

请输出：
1. 是否材料齐全
2. 缺失的材料列表
3. 完整度百分比
"""


def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name.
    
    Args:
        prompt_name: Name of the prompt template.
    
    Returns:
        Prompt template string.
    
    Raises:
        ValueError: If prompt name not found.
    """
    prompts = {
        "classify": CLASSIFY_DOCUMENT_PROMPT,
        "extract": EXTRACT_CONTENT_PROMPT,
        "material_check": MATERIAL_CHECK_PROMPT,
    }
    
    if prompt_name not in prompts:
        raise ValueError(f"Unknown prompt: {prompt_name}")
    
    return prompts[prompt_name]
