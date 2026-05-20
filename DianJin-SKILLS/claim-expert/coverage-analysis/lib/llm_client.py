# -*- coding: utf-8 -*-
"""LLM client utilities for claim processing skills.

This module provides a unified interface for calling LLM APIs.
Supports DashScope (Alibaba Cloud) vision and text models.
"""

import json
import os
from pathlib import Path
from typing import Any

# Try to import dashscope, provide helpful error if not installed
try:
    import dashscope
    from dashscope import MultiModalConversation, Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False


class LLMClientError(RuntimeError):
    """Error from LLM client operations."""
    pass


def _read_api_key_from_env_file() -> str:
    """Read DASHSCOPE_API_KEY from .env files in common locations."""
    # Search paths: current working dir, then skill root (parent of lib/)
    search_paths = [
        Path(".env"),
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in search_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DASHSCOPE_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def get_api_key(api_key: str = "") -> str:
    """Get API key from parameter, environment, or .env file.
    
    Priority:
        1. api_key parameter
        2. DASHSCOPE_API_KEY environment variable
        3. .env file in current directory or skill root
    
    Args:
        api_key: API key parameter. If empty, uses DASHSCOPE_API_KEY env var.
    
    Returns:
        API key string.
    
    Raises:
        ValueError: If no API key available.
    """
    if api_key:
        return api_key
    
    env_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if env_key:
        return env_key
    
    file_key = _read_api_key_from_env_file()
    if file_key:
        return file_key
    
    raise ValueError(
        "No API key provided. Set DASHSCOPE_API_KEY environment variable, "
        "create a .env file with DASHSCOPE_API_KEY=xxx, or pass api_key parameter."
    )


def call_vision_model(
    image_paths: list[str],
    prompt: str,
    api_key: str = "",
    model: str = "qwen-vl-plus",
) -> str:
    """Call a vision model with images and prompt.
    
    Args:
        image_paths: List of image file paths.
        prompt: Text prompt for the model.
        api_key: DashScope API key.
        model: Model name (e.g., "qwen-vl-plus", "qwen-vl-ocr").
    
    Returns:
        Model response text.
    
    Raises:
        LLMClientError: If API call fails.
    """
    if not DASHSCOPE_AVAILABLE:
        raise LLMClientError(
            "dashscope package not installed. "
            "Install with: pip install dashscope"
        )
    
    key = get_api_key(api_key)
    
    # Build messages with images
    messages = [{"role": "user", "content": []}]
    
    # Add images
    for path in image_paths:
        messages[0]["content"].append({"image": f"file://{path}"})
    
    # Add text prompt
    messages[0]["content"].append({"text": prompt})
    
    try:
        response = MultiModalConversation.call(
            model=model,
            messages=messages,
            api_key=key,
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            raise LLMClientError(
                f"Vision model call failed: {response.code} - {response.message}"
            )
    except Exception as e:
        raise LLMClientError(f"Vision model call error: {e}")


def call_vision_model_with_usage(
    image_paths: list[str],
    prompt: str,
    api_key: str = "",
    model: str = "qwen-vl-plus",
) -> dict:
    """Call a vision model and return detailed result with token usage.
    
    Compatible with qwen_ocr.py return format for unified LLM access.
    
    Args:
        image_paths: List of image file paths.
        prompt: Text prompt for the model.
        api_key: DashScope API key.
        model: Model name (e.g., "qwen-vl-plus", "qwen-vl-ocr-latest").
    
    Returns:
        Dict with keys: status ('success'|'error'), text, input_tokens, output_tokens.
        On error, key 'error' contains the message.
    """
    if not DASHSCOPE_AVAILABLE:
        return {
            'status': 'error',
            'error': (
                "dashscope package not installed. "
                "Install with: pip install dashscope"
            ),
        }
    
    try:
        key = get_api_key(api_key)
    except ValueError as e:
        return {'status': 'error', 'error': str(e)}
    
    messages = [{"role": "user", "content": []}]
    for path in image_paths:
        messages[0]["content"].append({"image": f"file://{path}"})
    messages[0]["content"].append({"text": prompt})
    
    try:
        response = MultiModalConversation.call(
            model=model,
            messages=messages,
            api_key=key,
        )
        
        if response.status_code == 200:
            text = response.output.choices[0].message.content[0]["text"]
            usage = getattr(response, 'usage', None) or {}
            # usage may be an object or dict depending on dashscope version
            input_tokens = getattr(usage, 'input_tokens', None) or usage.get('input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', None) or usage.get('output_tokens', 0)
            return {
                'status': 'success',
                'text': text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            }
        else:
            return {
                'status': 'error',
                'error': f"Vision model call failed: {response.code} - {response.message}",
            }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def call_text_model(
    prompt: str,
    api_key: str = "",
    model: str = "qwen3.5-plus",
) -> str:
    """Call a text model with prompt.
    
    Args:
        prompt: Text prompt for the model.
        api_key: DashScope API key.
        model: Model name (e.g., "qwen3.5-plus").
    
    Returns:
        Model response text.
    
    Raises:
        LLMClientError: If API call fails.
    """
    if not DASHSCOPE_AVAILABLE:
        raise LLMClientError(
            "dashscope package not installed. "
            "Install with: pip install dashscope"
        )
    
    key = get_api_key(api_key)
    
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            api_key=key,
        )
        
        if response.status_code == 200:
            return response.output.text
        else:
            raise LLMClientError(
                f"Text model call failed: {response.code} - {response.message}"
            )
    except Exception as e:
        raise LLMClientError(f"Text model call error: {e}")


def parse_json_response(response: str) -> Any:
    """Parse JSON from model response.
    
    Handles responses that may contain markdown code blocks or extra text.
    
    Args:
        response: Raw model response text.
    
    Returns:
        Parsed JSON object (usually list or dict).
    
    Raises:
        ValueError: If JSON cannot be parsed.
    """
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    if "```" in response:
        # Find content between ``` markers
        start = response.find("```")
        # Skip language identifier if present
        start = response.find("\n", start) + 1
        end = response.find("```", start)
        
        if start > 0 and end > start:
            json_str = response[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Try finding JSON array or object
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = response.find(start_char)
        end = response.rfind(end_char)
        
        if start >= 0 and end > start:
            json_str = response[start:end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    raise ValueError(f"Could not parse JSON from response: {response[:200]}...")
