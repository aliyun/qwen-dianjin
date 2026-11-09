
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from ..core.types import Trajectory


class UMMAdapter(ABC):
    @abstractmethod
    def generate_interleaved(
        self,
        prompt: str,
        images: Sequence[Any] = (),
        *,
        think: bool = True,
        **gen_cfg: Any,
    ) -> Trajectory:
        pass

    @abstractmethod
    def regenerate_images(
        self,
        prompt: str,
        gt_text: str,
        images: Sequence[Any] = (),
        *,
        captions: Optional[Sequence[str]] = None,
        **gen_cfg: Any,
    ) -> list[Any]:
        pass

    @abstractmethod
    def understand(
        self,
        images: Sequence[Any],
        question: str,
        *,
        history: Optional[list] = None,
        **gen_cfg: Any,
    ) -> str:
        pass

    @abstractmethod
    def regenerate_text(
        self,
        prompt: str,
        images: Sequence[Any],
        *,
        grounded_instruction: str = "",
        **gen_cfg: Any,
    ) -> str:
        pass

    def interleaved_teach_loss(self, sample: Any, *, objective: str = "sft") -> dict:
        raise NotImplementedError("interleaved_teach_loss is model-specific; override in the adapter.")

    @abstractmethod
    def trainable_modules(self) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def save_adapter(self, path: str) -> None: ...

    @abstractmethod
    def load_adapter(self, path: str) -> None: ...

    @property
    @abstractmethod
    def tokenizer(self) -> Any: ...

    @property
    @abstractmethod
    def gen_defaults(self) -> dict: ...
