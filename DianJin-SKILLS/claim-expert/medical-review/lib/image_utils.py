# -*- coding: utf-8 -*-
"""Image validation utilities for claim processing skills."""

import os
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image_path(image_path: str) -> str:
    """Validate a single image path.
    
    Args:
        image_path: Path to image file.
    
    Returns:
        Validated absolute path.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not a valid image.
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")
    
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    return str(path.absolute())


def validate_image_paths(image_paths: list[str]) -> list[str]:
    """Validate a list of image paths.
    
    Args:
        image_paths: List of image file paths.
    
    Returns:
        List of validated absolute paths.
    
    Raises:
        FileNotFoundError: If any file doesn't exist.
        ValueError: If any file is not a valid image.
    """
    if not image_paths:
        raise ValueError("Image path list is empty")
    
    validated = []
    for path in image_paths:
        validated.append(validate_image_path(path))
    
    return validated
