from __future__ import annotations

import re
from typing import Optional, Sequence

MIN_WORDS = 3
MIN_CHARS = 12


def _snap(text: str, start: int, end: int) -> tuple:
    lo = text.rfind("\n", 0, start) + 1
    hi = text.find("\n", end)
    return lo, (len(text) if hi < 0 else hi)


def _pattern(words: Sequence[str]) -> re.Pattern:
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def _longest_prefix_matches(words: list, text: str) -> tuple:
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _pattern(words[:mid]).search(text):
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return 0, []
    return lo, [m.span() for m in _pattern(words[:lo]).finditer(text)]


def locate_quote(quote: str, text: str, *, forbid: Optional[str] = None) -> Optional[tuple]:
    words = (quote or "").split()
    if len(words) < MIN_WORDS or not text:
        return None
    n, spans = _longest_prefix_matches(words, text)
    if not spans:
        return None
    start, end = spans[-1]
    if end - start < MIN_CHARS:
        return None
    if forbid and _pattern(words[:n]).search(forbid):
        return None
    return _snap(text, start, end)
