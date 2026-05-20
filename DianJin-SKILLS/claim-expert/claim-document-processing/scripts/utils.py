#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保险理赔材料分析 - 公共工具模块

提供文件加载、图片编码、模型调用、JSON提取等通用能力。
被 check_completeness.py、check_consistency.py、cross_validate.py 等脚本引用。
"""

import os
import io
import re
import json
import base64
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False


# ===== 数据结构 =====

@dataclass
class LoadedFile:
    """加载后的文件信息"""
    original_path: str
    file_name: str
    file_type: str  # "pdf" or "image"
    page_index: int  # PDF页码(0-based)，图片为0
    base64_data: str
    width: int
    height: int


# ===== 文件加载 =====

SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
SUPPORTED_PDF_FORMATS = {'.pdf'}
MAX_IMAGE_SIZE = 2048


def resize_image(image: "Image.Image", max_size: int = MAX_IMAGE_SIZE) -> "Image.Image":
    """等比例缩放图片"""
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_base64(image: "Image.Image", quality: int = 85) -> str:
    """将PIL Image转换为base64字符串"""
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def load_image_file(file_path: str) -> List[LoadedFile]:
    """加载单个图片文件"""
    if not PIL_AVAILABLE:
        print(f"[ERROR] Pillow 未安装，无法加载图片: {file_path}")
        return []
    try:
        image = Image.open(file_path)
        image = resize_image(image)
        b64 = image_to_base64(image)
        return [LoadedFile(
            original_path=file_path,
            file_name=os.path.basename(file_path),
            file_type="image",
            page_index=0,
            base64_data=b64,
            width=image.width,
            height=image.height
        )]
    except Exception as e:
        print(f"[ERROR] 加载图片失败: {file_path} - {e}")
        return []


def load_pdf_file(file_path: str, dpi: int = 150) -> List[LoadedFile]:
    """加载PDF文件并转换为图片"""
    if not PDF2IMAGE_AVAILABLE:
        print(f"[ERROR] pdf2image 未安装，无法加载PDF: {file_path}")
        return []
    try:
        images = convert_from_path(file_path, dpi=dpi)
        loaded = []
        for i, img in enumerate(images):
            img = resize_image(img)
            b64 = image_to_base64(img)
            loaded.append(LoadedFile(
                original_path=file_path,
                file_name=os.path.basename(file_path),
                file_type="pdf",
                page_index=i,
                base64_data=b64,
                width=img.width,
                height=img.height
            ))
        return loaded
    except Exception as e:
        print(f"[ERROR] 加载PDF失败: {file_path} - {e}")
        return []


def load_directory(directory: str) -> Tuple[List[LoadedFile], Dict[str, int]]:
    """
    加载目录下所有支持的文件

    Returns:
        (已加载文件列表, 统计信息字典)
    """
    all_files = []
    stats = {"total_files": 0, "pdf": 0, "image": 0, "total_pages": 0}

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[ERROR] 目录不存在: {directory}")
        return [], stats

    supported = SUPPORTED_IMAGE_FORMATS | SUPPORTED_PDF_FORMATS

    for file_path in sorted(dir_path.iterdir()):
        if not file_path.is_file() or file_path.name.startswith('.'):
            continue
        suffix = file_path.suffix.lower()
        if suffix not in supported:
            continue

        if suffix in SUPPORTED_PDF_FORMATS:
            loaded = load_pdf_file(str(file_path))
            if loaded:
                stats["pdf"] += 1
        else:
            loaded = load_image_file(str(file_path))
            if loaded:
                stats["image"] += 1

        if loaded:
            all_files.extend(loaded)
            stats["total_files"] += 1
            stats["total_pages"] += len(loaded)

    return all_files, stats


def get_file_info_list(loaded_files: List[LoadedFile]) -> List[Dict[str, str]]:
    """获取文件信息摘要列表"""
    return [
        {
            "file_name": f.file_name,
            "file_type": f.file_type,
            "page_index": f.page_index,
            "size": f"{f.width}x{f.height}"
        }
        for f in loaded_files
    ]


# ===== 模型调用 =====

def call_qwen_multimodal(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    max_tokens: int = 8192
) -> Tuple[str, Dict[str, Any]]:
    """
    通过 OpenAI 兼容模式调用 DashScope 多模态模型 (Qwen)

    Args:
        api_key: DashScope API Key
        model: 模型名称
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        images: base64图片数据列表
        max_tokens: 最大输出token数

    Returns:
        (响应文本, 元数据字典)
    """
    if not OPENAI_AVAILABLE:
        return "", {"success": False, "error": "openai 未安装"}

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 构建用户消息内容
    user_content = []
    for img_b64 in images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    user_content.append({"type": "text", "text": user_prompt})

    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=max_tokens
        )

        duration_ms = (time.time() - start_time) * 1000
        result_text = completion.choices[0].message.content if completion.choices else ""
        usage = completion.usage
        metadata = {
            "success": True,
            "model": model,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "duration_ms": duration_ms
        }
        return result_text, metadata

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return "", {"success": False, "error": str(e), "duration_ms": duration_ms}


def call_kimi_multimodal(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    max_tokens: int = 8192,
    temperature: float = 0.1
) -> Tuple[str, Dict[str, Any]]:
    """
    通过 DashScope SDK 调用 Kimi 多模态模型

    Args:
        api_key: DashScope API Key
        model: 模型名称 (如 kimi-k2.5)
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        images: base64图片数据列表
        max_tokens: 最大输出token数
        temperature: 生成温度

    Returns:
        (响应文本, 元数据字典)
    """
    if not DASHSCOPE_AVAILABLE:
        return "", {"success": False, "error": "dashscope 未安装"}

    messages = [
        {"role": "system", "content": [{"text": system_prompt}]}
    ]

    user_content = []
    for img_b64 in images:
        user_content.append({"image": f"data:image/jpeg;base64,{img_b64}"})
    user_content.append({"text": user_prompt})
    messages.append({"role": "user", "content": user_content})

    start_time = time.time()
    try:
        response = MultiModalConversation.call(
            model=model,
            messages=messages,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            result_format="message"
        )

        duration_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            output = response.output
            choices = output.get("choices", [])
            text_content = ""
            if choices:
                content = choices[0].get("message", {}).get("content", [])
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type not in ("thinking", "reasoning") and "text" in item:
                            text_content += item["text"]
                    elif isinstance(item, str):
                        text_content += item

            usage = response.usage or {}
            metadata = {
                "success": True,
                "model": model,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "duration_ms": duration_ms
            }
            return text_content, metadata
        else:
            error_msg = f"API错误: {response.code} - {response.message}"
            return "", {"success": False, "error": error_msg, "duration_ms": duration_ms}

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return "", {"success": False, "error": str(e), "duration_ms": duration_ms}


# ===== JSON 提取 =====

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型响应文本中提取JSON（5级提取策略）

    优先提取包含 'documents' 键的完整结构化结果。
    """
    if not text:
        return None

    # 方法1: 直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 方法2: ```json ... ``` 代码块
    matches = re.findall(r'```json\s*([\s\S]*?)```', text)
    for match in matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    # 方法3: ``` ... ``` 代码块
    matches = re.findall(r'```\s*([\s\S]*?)```', text)
    for match in matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    # 方法4: 查找最大的完整JSON对象（优先含documents）
    brace_start = text.find('{')
    candidates = []
    while brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[brace_start:i + 1]
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, dict):
                            if 'documents' in result:
                                return result
                            candidates.append(result)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        brace_start = text.find('{', brace_start + 1)

    # 方法5: 返回最大的候选对象
    if candidates:
        return max(candidates, key=lambda x: len(json.dumps(x)))

    return None


# ===== 报告生成 =====

def get_skill_root() -> str:
    """获取Skill根目录路径"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_output_dir(scenario: str) -> str:
    """
    在 skill 根目录下的 output/ 中创建以 时间戳_场景 命名的输出目录。
    
    Args:
        scenario: 场景名称，如 "classify", "full_audit", "completeness_consistency" 等
    
    Returns:
        创建好的输出目录绝对路径
    """
    skill_root = get_skill_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(skill_root, "output", f"{timestamp}_{scenario}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_json_report(report: Dict[str, Any], output_path: str) -> str:
    """保存JSON格式报告"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path


def generate_audit_log(
    skill_version: str,
    input_dir: str,
    file_count: int,
    model: str,
    capabilities: List[str],
    steps_executed: List[str],
    total_tokens: int,
    duration_seconds: float,
    errors: List[str],
    confidence_scores: Dict[str, Optional[float]],
    risk_flags: List[str],
    output_path: str
) -> Dict[str, Any]:
    """
    生成金融级审计日志记录
    
    满足审计追溯要求：记录每步操作的输入、输出、时间戳，支持监管回溯。
    """
    import hashlib
    import uuid
    
    # 计算输入目录hash（用于追溯）
    input_hash = ""
    try:
        dir_content = sorted(os.listdir(input_dir)) if os.path.isdir(input_dir) else []
        input_hash = hashlib.sha256(
            json.dumps(dir_content, sort_keys=True).encode()
        ).hexdigest()[:16]
    except Exception:
        input_hash = "unknown"
    
    # 计算输出hash
    output_hash = ""
    try:
        if os.path.isfile(output_path):
            with open(output_path, 'rb') as f:
                output_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        output_hash = "unknown"
    
    audit_record = {
        "audit_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "skill_version": skill_version,
        "input": {
            "file_count": file_count,
            "input_dir_hash": input_hash,
            "model": model,
            "capabilities_requested": capabilities
        },
        "execution": {
            "steps_executed": steps_executed,
            "total_tokens": total_tokens,
            "total_duration_seconds": round(duration_seconds, 2),
            "errors": errors
        },
        "output": {
            "output_path": output_path,
            "report_hash": output_hash,
            "risk_flags": risk_flags,
            "confidence_scores": confidence_scores
        },
        "risk_disclosure": {
            "confidence": min(
                (v for v in confidence_scores.values() if v is not None),
                default=0.0
            ),
            "data_freshness": "实时处理",
            "disclaimer": "本审核结果由AI辅助生成，仅供理赔人员参考，最终决定须由授权人员确认"
        }
    }
    return audit_record


def save_audit_log(audit_record: Dict[str, Any], audit_dir: str = None) -> str:
    """
    保存审计日志（追加模式，不可篡改）
    
    日志以 JSONL 格式追加到 audit_log.jsonl 文件中。
    同时在指定的 audit_dir (通常为本次执行的输出目录) 中保存一份副本。
    """
    skill_root = get_skill_root()
    
    # 全局审计日志（追加）
    global_audit_path = os.path.join(skill_root, "output", "audit_log.jsonl")
    os.makedirs(os.path.dirname(global_audit_path), exist_ok=True)
    with open(global_audit_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(audit_record, ensure_ascii=False) + '\n')
    
    # 如果指定了输出目录，也在该目录中保存一份本次的审计日志
    if audit_dir and audit_dir != os.path.dirname(global_audit_path):
        local_audit_path = os.path.join(audit_dir, "audit_log.json")
        os.makedirs(audit_dir, exist_ok=True)
        with open(local_audit_path, 'w', encoding='utf-8') as f:
            json.dump(audit_record, f, ensure_ascii=False, indent=2)
    
    return global_audit_path


def print_separator(title: str = "", char: str = "=", width: int = 60):
    """打印分隔符"""
    if title:
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")
    else:
        print(char * width)
