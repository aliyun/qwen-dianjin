
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .types import Step, Trajectory

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_BARE_IMAGE_RE = re.compile(r"<image>")
_TAG_RE = re.compile(r"<image(\d+)>")
_EXPLICIT_RE = re.compile(r"#\s*Explicit\s+Prompt:\s*", re.IGNORECASE)
_SAVED_RE = re.compile(r"#\s*PROMPT\s*\n(.*?)\n\s*#\s*OUTPUT\s*\n(.*)", re.DOTALL)


def _clean(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def _split_caption(seg: str) -> tuple[str, Optional[str]]:
    parts = _EXPLICIT_RE.split(seg, maxsplit=1)
    if len(parts) == 1: 
        reasoning = _TAG_RE.sub("", seg)
        return _clean(reasoning), None
    reasoning = parts[0]
    caption = _TAG_RE.split(parts[1])[0]
    return _clean(reasoning), _clean(caption).strip() or None


def _parse_output(
    prompt: str,
    output_text: str,
    images: Optional[Sequence[Any]] = None,
) -> Trajectory:
    m = _THINK_RE.search(output_text)
    if m:
        think_text = m.group(1)
        final_answer = output_text[m.end():].strip()
    else:
        think_text = ""
        final_answer = output_text.strip()

    segments = _BARE_IMAGE_RE.split(output_text)
    n_markers = len(segments) - 1
    parse_mismatch = 0
    if images is not None:
        num_images = len(images)
        parse_mismatch = n_markers - num_images
    else:
        num_images = n_markers

    steps: list[Step] = []
    for i in range(num_images):
        seg = segments[i] if i < len(segments) else ""
        reasoning, caption = _split_caption(seg)
        tag_matches = _TAG_RE.findall(seg)
        tag = f"<image{tag_matches[-1]}>" if tag_matches else None
        steps.append(Step(text=reasoning, caption=caption, image_index=i, image_tag=tag))

    if images is None:
        images_list: list[Any] = [None] * num_images
    else:
        images_list = list(images)

    return Trajectory(
        prompt=prompt.strip(),
        raw_output=output_text,
        think_text=think_text.strip(),
        final_answer=final_answer,
        steps=steps,
        images=images_list,
        parse_mismatch=parse_mismatch,
    )


def from_generation(
    prompt: str,
    output_text: str,
    images: Optional[Sequence[Any]] = None,
    input_images: Optional[Sequence[Any]] = None,
    think_mode: bool = True,
) -> Trajectory:
    traj = _parse_output(prompt, output_text, images=images)
    traj.think_mode = think_mode
    if input_images:
        traj.input_images = list(input_images)
    return traj


def parse_saved_txt(path: str | Path, load_images: bool = False) -> Trajectory:
    path = Path(path)
    body = path.read_text(encoding="utf-8")
    m = _SAVED_RE.search(body)
    if not m:
        raise ValueError(f"{path} is not in '# PROMPT / # OUTPUT' format")
    prompt, output_text = m.group(1).strip(), m.group(2)

    traj = _parse_output(prompt, output_text, images=None)

    stem = path.stem
    resolved: list[Any] = []
    for i in range(traj.num_images):
        img_path = path.with_name(f"{stem}_image_{i}.png")
        if load_images:
            from PIL import Image

            resolved.append(Image.open(img_path).convert("RGB"))
        else:
            resolved.append(str(img_path))
    traj.images = resolved

    inputs: list[Any] = []
    i = 0
    while True:
        in_path = path.with_name(f"{stem}_input_{i}.png")
        if not in_path.exists():
            break
        inputs.append(str(in_path))
        i += 1
    traj.input_images = inputs
    return traj
