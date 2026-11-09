
from __future__ import annotations

from typing import Optional

TAIL_CHARS = 4000
MAX_PERIOD = 800
MIN_REPEATS = 5.0


def _prefix_function(s: str) -> list:
    pi = [0] * len(s)
    k = 0
    for i in range(1, len(s)):
        while k and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi


def min_period(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    return n - _prefix_function(s)[-1]


def detect_repetition(text: str, *, tail_chars: int = TAIL_CHARS,
                      max_period: int = MAX_PERIOD,
                      min_repeats: float = MIN_REPEATS) -> Optional[dict]:
    tail = (text or "")[-tail_chars:]
    n = len(tail)
    if n < 200:
        return None
    pi = _prefix_function(tail[::-1])
    for L in range(n, 199, -1):
        p = L - pi[L - 1]
        if p <= max_period and L / p >= min_repeats:
            return {"period": p, "repeats": round(L / p, 1), "unit": tail[-p:][:200]}
    return None


def truncate_repetition(text: str, *, keep_periods: int = 3, **kw) -> tuple[str, Optional[dict]]:
    det = detect_repetition(text, **kw)
    if not det:
        return text, None
    p = det["period"]
    i = len(text) - p - 1
    while i >= 0 and text[i] == text[i + p]:
        i -= 1
    periodic_len = len(text) - (i + 1)
    cut = periodic_len - keep_periods * p
    if cut > 0:
        det = dict(det, cut_chars=cut)
        return text[: len(text) - cut], det
    return text, det
