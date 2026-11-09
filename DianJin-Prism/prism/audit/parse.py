
from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..core.types import AuditResult

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_DESC_TAG_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL | re.IGNORECASE)


def extract_description(text: str) -> Optional[str]:
    m = _DESC_TAG_RE.search(text or "")
    return m.group(1).strip() if m else None


def description_truncated(text: str) -> bool:
    t = text or ""
    return "<description>" in t.lower() and extract_description(t) is None


def _coerce_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1", "valid", "correct", "pass", "passed"}:
            return True
        if s in {"false", "no", "0", "invalid", "incorrect", "wrong", "fail", "failed"}:
            return False
    return None


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_json(text: str) -> Optional[dict]:
    for candidate in _iter_json_candidates(text):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and "valid" in obj:
            return obj
    return None


def _iter_json_candidates(text: str):
    b = _first_balanced_object(text)
    if b:
        yield b
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        yield m.group(1)
    m = _JSON_OBJ_RE.search(text)
    if m:
        yield m.group(0)


def _strip_own_think(raw: str) -> str:
    t = raw.lstrip()
    if not t.startswith("<think>"):
        return raw
    if "</think>" not in t:
        return ""
    return t.split("</think>", 1)[1]


def trim_to_last_image(chain: str) -> str:
    if not chain:
        return chain
    idx = chain.rfind("<image>")
    if idx == -1:
        return chain
    return chain[: idx + len("<image>")]


_SLOT_RE = re.compile(r"<slot\s+(\d+)>\s*(.*?)\s*</slot\s+\1>", re.I | re.S)
_SLOT_OPEN_RE = re.compile(r"<slot\s+(\d+)>", re.I)
_SLOT_CLOSE_RE = re.compile(r"</slot\s*\d*>", re.I)


def parse_slots_lenient(desc: str):
    if not desc:
        return []
    ms = list(_SLOT_OPEN_RE.finditer(desc))
    out = []
    seen = set()
    for i, m in enumerate(ms):
        k = int(m.group(1))
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(desc)
        body = desc[start:end]
        body = _SLOT_CLOSE_RE.sub("", body)
        body = re.sub(r"</?description>|<image\d*>", "", body, flags=re.I)
        body = body.strip()
        if not re.search(r"[A-Za-z]", body):
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append((k, body))
    return out


def slots_to_interleaved(desc: str) -> str:
    slots = parse_slots_lenient(desc)
    if not slots:
        return ""
    return "\n".join(f"{body}\n<image>" for _, body in slots)


_IMG_HEAD_RE = re.compile(r"^\s{0,3}#{1,4}\s*\**\s*(?:image|scene|step)\s*#?\s*(\d+)\s*[:.)\-\*]*[ \t]*", re.IGNORECASE | re.MULTILINE)
_SUMMARY_HEAD_RE = re.compile(r"^\s{0,3}#{0,4}\s*\**\s*(?:summary|final answer|conclusion|final|answer)\b\s*[:.)\-\*]*[ \t]*", re.IGNORECASE | re.MULTILINE)


def image_desc_to_tplus(raw: str, n_images: int, *, think: bool = True) -> str:
    from ..core.protocol import render_interleaved, render_think_interleaved

    body = _strip_own_think(raw or "")
    ms = _SUMMARY_HEAD_RE.search(body)
    if ms:
        pre, summary = body[:ms.start()], body[ms.end():].strip()
    else:
        pre, summary = body, ""
    heads = list(_IMG_HEAD_RE.finditer(pre))
    if n_images and len(heads) != n_images:
        return ""
    if not heads:
        return ""
    segs = []
    for i, h in enumerate(heads):
        start = h.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(pre)
        seg = pre[start:end].strip()
        if not re.search(r"[A-Za-z]", seg):
            return ""
        segs.append(seg)
    if think:
        return render_think_interleaved(segs, summary)
    return render_interleaved(segs, summary)


_NO_QUOTE = {"", "null", "none", "n/a", "na", "-"}


def _clean_quote(value: Any) -> str:
    s = "" if value is None else str(value).strip()
    return "" if s.lower() in _NO_QUOTE else s


def parse_audit(raw: str) -> AuditResult:
    raw = raw or ""
    body = _strip_own_think(raw)
    tag_desc = extract_description(body)
    obj = _try_json(body)
    if obj is not None:
        valid = _coerce_bool(obj.get("valid"))
        if valid is None:
            valid = True
        reason = str(obj.get("reason", "")).strip()
        checklist = obj.get("checklist")
        if not reason and isinstance(checklist, dict):
            problems = [f"{k}: {v}" for k, v in checklist.items() if str(v).strip().upper() != "OK"]
            reason = "; ".join(problems)
        return AuditResult(
            valid=valid,
            reason=reason,
            description=(tag_desc if tag_desc is not None else str(obj.get("description", "")).strip()),
            quote=_clean_quote(obj.get("quote")),
            raw=raw,
        )

    if body.count("{") > body.count("}"):
        return AuditResult(valid=True, description=(tag_desc or ""), raw=raw)
    valid = _labelled_bool(body)
    reason = _labelled_field(body, "reason")
    description = tag_desc if tag_desc is not None else _labelled_field(body, "description")
    if valid is None:
        valid = True
    return AuditResult(valid=valid, reason=reason, description=description, raw=raw)


def _labelled_bool(text: str) -> Optional[bool]:
    m = re.search(r"valid\s*\"?\s*[:=]\s*([A-Za-z01]+)", text, re.IGNORECASE)
    if m:
        return _coerce_bool(m.group(1))
    return None


def _labelled_field(text: str, name: str) -> str:
    m = re.search(
        rf"{name}\s*\"?\s*[:=]\s*(.+?)(?:\n\s*\"?\w+\"?\s*[:=]|\"\s*,\s*\"\w+\"\s*[:=]|\Z)",
        text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().strip('"') if m else ""
