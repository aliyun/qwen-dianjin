
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Quadrant(str, Enum):

    SOLVED = "solved"
    T2I = "t2i"
    I2T = "i2t"
    UNTEACHABLE = "unteachable"

    @classmethod
    def from_valids(cls, valid_text: bool, valid_image: bool) -> "Quadrant":
        if valid_text and valid_image:
            return cls.SOLVED
        if valid_text and not valid_image:
            return cls.T2I
        if not valid_text and valid_image:
            return cls.I2T
        return cls.UNTEACHABLE


@dataclass
class Step:

    text: str
    caption: Optional[str] = None
    image_index: Optional[int] = None
    image_tag: Optional[str] = None


@dataclass
class Trajectory:

    prompt: str
    raw_output: str = ""
    think_text: str = ""
    final_answer: str = ""
    steps: list[Step] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    input_images: list[Any] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    parse_mismatch: int = 0
    think_mode: bool = True

    def text_chain(self) -> list[str]:
        return [self.prompt] + [s.text for s in self.steps if s.text]

    def reasoning_text(self) -> str:
        return "\n\n".join(s.text for s in self.steps if s.text)

    def image_chain(self) -> list[Any]:
        return list(self.images)

    def captions(self) -> list[str]:
        return [s.caption or "" for s in self.steps]

    def raw_segments(self) -> list[str]:
        import re

        return re.split(r"<image>", self.raw_output or "")

    @property
    def num_images(self) -> int:
        return len(self.images)


@dataclass
class AuditResult:

    valid: bool
    reason: str = ""
    description: str = ""
    quote: str = ""
    raw: str = ""


@dataclass
class Audit:

    text: AuditResult
    image: AuditResult

    @property
    def quadrant(self) -> Quadrant:
        return Quadrant.from_valids(self.text.valid, self.image.valid)


@dataclass
class InterleaveStep:

    text: str = ""
    image: Any = None


@dataclass
class TeachSample:

    kind: str
    prompt: str
    steps: list[InterleaveStep] = field(default_factory=list)
    reliable_side: Optional[str] = None
    context_text: Optional[str] = None
    cond_images: list[Any] = field(default_factory=list)
    target_image: Any = None
    target_text: Optional[str] = None
    neg_text: Optional[str] = None
    quadrant: Optional[str] = None
    audit: Optional[dict] = None
    meta: dict = field(default_factory=dict)

    @property
    def chain_texts(self) -> list[str]:
        return [s.text for s in self.steps]

    @property
    def chain_images(self) -> list[Any]:
        return [s.image for s in self.steps if s.image is not None]

    @property
    def target_images(self) -> list[Any]:
        return self.chain_images if self.reliable_side == "text" else []

    @property
    def target_texts(self) -> list[str]:
        return [s.text for s in self.steps if s.text] if self.reliable_side == "image" else []
