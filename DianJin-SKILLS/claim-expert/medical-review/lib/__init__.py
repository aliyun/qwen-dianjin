# -*- coding: utf-8 -*-
"""Self-contained library for material_check skill."""

from .image_utils import validate_image_path, validate_image_paths
from .llm_client import (
    call_vision_model,
    call_text_model,
    parse_json_response,
    LLMClientError,
)
from .knowledge import (
    DOCUMENT_TYPES,
    MEDICAL_RECORD_TYPES,
    DRUG_RELATED_TYPES,
    LIABILITY_RELATED_TYPES,
    CALCULATION_RELATED_TYPES,
    REQUIRED_DOCUMENTS,
)
from .prompts import (
    CLASSIFY_DOCUMENT_PROMPT,
    EXTRACT_CONTENT_PROMPT,
    MATERIAL_CHECK_PROMPT,
    get_prompt,
)

__all__ = [
    # Image utilities
    "validate_image_path",
    "validate_image_paths",
    # LLM client
    "call_vision_model",
    "call_text_model", 
    "parse_json_response",
    "LLMClientError",
    # Knowledge
    "DOCUMENT_TYPES",
    "MEDICAL_RECORD_TYPES",
    "DRUG_RELATED_TYPES",
    "LIABILITY_RELATED_TYPES",
    "CALCULATION_RELATED_TYPES",
    "REQUIRED_DOCUMENTS",
    # Prompts
    "CLASSIFY_DOCUMENT_PROMPT",
    "EXTRACT_CONTENT_PROMPT",
    "MATERIAL_CHECK_PROMPT",
    "get_prompt",
]
