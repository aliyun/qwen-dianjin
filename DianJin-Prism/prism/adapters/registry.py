
from __future__ import annotations

from typing import Any, Callable

from .base import UMMAdapter

_FACTORIES: dict[str, Callable[..., UMMAdapter]] = {}


def register(name: str) -> Callable[[Callable[..., UMMAdapter]], Callable[..., UMMAdapter]]:
    def deco(fn: Callable[..., UMMAdapter]) -> Callable[..., UMMAdapter]:
        _FACTORIES[name] = fn
        return fn

    return deco


def build_adapter(name: str, **kwargs: Any) -> UMMAdapter:
    if name not in _FACTORIES:
        if name == "sensenova_u1":
            from . import sensenova_u1
        elif name == "bagel":
            from . import bagel
        elif name == "janus_pro":
            from . import janus_pro
    if name not in _FACTORIES:
        raise KeyError(f"unknown adapter '{name}'; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**kwargs)


def available() -> list[str]:
    return sorted(_FACTORIES)
