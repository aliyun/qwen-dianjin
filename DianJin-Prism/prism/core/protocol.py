
from __future__ import annotations

import re

_NUMBERED_IMAGE_RE = re.compile(r"<image(\d+)>")
_MARKER_ONLY_LINE_RE = re.compile(r"^[ \t]*<image\d*>[ \t]*$")
_INLINE_MARKER_RE = re.compile(r"[ \t]*<image\d*>[ \t]*")


def _clear_markers(segment: str) -> str:
    s = _INLINE_MARKER_RE.sub(" ", segment or "")
    s = re.sub(r" +([,.;:!?])", r"\1", s)
    return re.sub(r"^[ \t]+|[ \t]+$", "", s, flags=re.MULTILINE)


def render_think_interleaved(segments: list[str], summary: str = "") -> str:
    body = "\n".join(
        f"{_clear_markers(seg).strip()}\n<image{i + 1}>\n<image>"
        for i, seg in enumerate(segments)
    )
    tail = ("\n\n" + summary.strip()) if summary and summary.strip() else ""
    return f"<think>\n{body}\n</think>{tail}"


def render_interleaved(segments: list[str], summary: str = "") -> str:
    body = "\n".join(
        f"{_clear_markers(seg).strip()}\n<image>"
        for seg in segments
    )
    summary_clean = strip_image_references(summary).strip() if summary else ""
    tail = ("\n\n" + summary_clean) if summary_clean else ""
    return f"{body}{tail}"


def strip_all_image_markers(text: str) -> str:
    text = strip_image_references(text)
    text = text.replace("<image>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_user_images(n_images: int) -> str:
    if n_images <= 0:
        return ""
    return "\n".join(f"Image-{i + 1}: <image>" for i in range(n_images))


def refs_valid(text: str, n_images: int) -> bool:
    nums = [int(x) for x in _NUMBERED_IMAGE_RE.findall(text or "")]
    if not nums:
        return True
    if n_images <= 0:
        return False
    return len(nums) == n_images and sorted(nums) == list(range(1, n_images + 1))


def strip_image_references(text: str) -> str:
    out: list[str] = []
    for line in (text or "").split("\n"):
        if _MARKER_ONLY_LINE_RE.match(line):
            continue
        if "<image" in line:
            line = _INLINE_MARKER_RE.sub(" ", line)
            line = re.sub(r" +([,.;:!?])", r"\1", line).rstrip()
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))
