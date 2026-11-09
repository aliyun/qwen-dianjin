
from __future__ import annotations

import re

from . import prompts as _base


INTERLEAVE_ROLLOUT_TEMPLATE = """Write an illustrated answer to the task below: prose \
with pictures placed inside it.

You do not draw anything yourself. Writing the line <image> is how you ASK for a picture: \
the picture is drawn from the passage right before it and appears there. So never say you \
are unable to make images, and never tell the reader to draw them.

Example task: Describe a kettle heating up on a wooden table.
Example answer:

The kettle sits on a bare wooden table, its copper body still dull.
<image>
Steam lifts from the spout and the polished copper gleams.
<image>

Rules:
- Every <image> line must be immediately preceded by at least one full sentence of prose \
describing what that picture shows. Never put two <image> lines in a row.
- Write only the answer a reader is meant to see. Do not restate or quote the task, do not \
announce what you are about to do, and do not add notes about your process.
- Answering in words alone is not enough: an answer with no <image> line is a failed answer.
- If the task asks for a specific number of scenes, steps, or pictures, produce exactly that \
many pictures. If it does not say how many, that is yours to decide: use as many as the \
content genuinely needs and no more — there is no target to hit.
- Use at most {max_images} markers.
- Write plain <image>. Never number it (`<image1>`, `<image 2>`) and never refer to a \
picture by number.
- Keep each passage to one to three sentences, and make it concrete about what is visible: \
the subjects, their attributes, their number, and where they are placed.
- Every passage must move the situation on from the one before it. Do not reuse wording you \
have already written, and do not spend a passage on something that leaves the situation \
unchanged.
- If the task states the moment the answer must end on, that moment belongs to the LAST \
passage and to no other: do not show it, or the state it leaves behind, in any earlier passage.

Your task:
TASK:
{prompt}

Write the illustrated answer now."""


def interleave_rollout_prompt(prompt: str, max_images: int = 8) -> str:
    return INTERLEAVE_ROLLOUT_TEMPLATE.format(
        prompt=(prompt or "").strip(), max_images=int(max_images)
    )


_BAGEL_REWRITE_NOTE = """

BAGEL-specific rewrite requirements:
- Write direct, concrete prompts that BAGEL can render as one image at a time.
- Prefer concise visual instructions with explicit objects, attributes, count, spatial relations, and style when required.
- Do not use internal planning language, numbered image references such as <image1>, or markdown scaffolding unless the output contract explicitly asks for sections.
"""


def text_audit_prompt(prompt: str, text_chain: str, *, think_mode: bool = True) -> str:
    return _base.text_audit_prompt(prompt, text_chain, think_mode=think_mode)


def image_audit_prompt(prompt: str, n_images: int) -> str:
    return _base.image_audit_prompt(prompt, n_images)


def text_desc_prompt(prompt: str, text_chain: str, findings: str, *, think: bool = True) -> str:
    return _base.text_desc_prompt(prompt, text_chain, findings, think=think) + _BAGEL_REWRITE_NOTE


def image_desc_prompt(prompt: str, n_images: int, findings: str, *, think: bool = True) -> str:
    return _base.image_desc_prompt(prompt, n_images, findings, think=think) + _BAGEL_REWRITE_NOTE


def span_repair_prompt(prompt: str, reason: str, passage: str, span: str) -> str:
    return _base.span_repair_prompt(prompt, reason, passage, span) + _BAGEL_REWRITE_NOTE


def final_answer_prompt(prompt: str, think_text: str, n_images: int, findings: str, orig_answer: str = "") -> str:
    return _base.final_answer_prompt(
        prompt, think_text, n_images, findings, orig_answer
    ) + _BAGEL_REWRITE_NOTE
