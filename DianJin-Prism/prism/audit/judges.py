
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import nullcontext
from typing import Any, Optional, Sequence

from ..core.types import AuditResult
from . import prompts
from .parse import parse_audit, extract_description, _strip_own_think


class Judge(ABC):

    @abstractmethod
    def audit_text(self, prompt: str, text_chain: str) -> AuditResult: ...

    @abstractmethod
    def audit_image(self, prompt: str, images: Sequence[Any]) -> AuditResult: ...


_NUM_IMG_RE = re.compile(r"<image(\d+)>")
_BARE_IMG_RE = re.compile(r"<image>[ \t]*\n?")

ANSWER_TOKENS = 2048


def _refs_complete(text: str, n_images: int) -> bool:
    nums = sorted(int(x) for x in _NUM_IMG_RE.findall(text or ""))
    return bool(n_images) and nums == list(range(1, n_images + 1))


def _needs_retry(res: AuditResult, expect_desc: bool = True) -> bool:
    from .parse import _strip_own_think, extract_description

    raw = res.raw or ""
    t = raw.lstrip()
    if t.startswith("<think>") and "</think>" not in t:
        return True
    body = _strip_own_think(raw)
    if body.count("{") > body.count("}"):
        return True
    if "<description>" in body.lower() and extract_description(body) is None:
        return True
    return False


class SelfJudge(Judge):

    def __init__(
        self,
        adapter: Any,
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        retry_on_empty: bool = True,
        think: bool = True,
        think_cap_text: int = 0,
        think_cap_image: int = 0,
        finalizer_tokens: int = 1024,
        desc_think: bool = True,
        desc_max_tokens: int = 12288,
        repair_tokens: int = 512,
        answer_tries: int = 4,
        audit_sample: bool = False,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        prompt_module: Any = None,
    ) -> None:
        self.adapter = adapter
        self.think = think
        self.desc_think = desc_think
        self.repair_tokens = max(128, int(repair_tokens))
        self.answer_tries = max(1, int(answer_tries))
        self.last_repair_seconds = None
        self.last_answer_seconds = None
        self.think_cap_text = max(0, int(think_cap_text))
        self.think_cap_image = max(0, int(think_cap_image))
        self.finalizer_tokens = max(256, int(finalizer_tokens))
        self.P = prompt_module or prompts
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.desc_max_tokens = desc_max_tokens
        self.retry_on_empty = retry_on_empty
        self.audit_sample = bool(audit_sample)
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    def _frozen_base(self):
        fn = getattr(self.adapter, "frozen_base_context", None)
        return fn() if callable(fn) else nullcontext()

    def _unwrapped(self):
        fn = getattr(self.adapter, "unwrapped_context", None)
        return fn() if callable(fn) else nullcontext()

    def _raw_understand(self, images: Sequence[Any], question: str, *, think: bool,
                        sample: bool, max_new_tokens: int, prefill: Optional[str] = None) -> str:
        use_sample = sample or self.audit_sample
        kw = dict(think=think, do_sample=use_sample, temperature=self.temperature,
                  max_new_tokens=max_new_tokens,
                  repetition_penalty=self.repetition_penalty)
        if use_sample:
            kw.update(top_p=self.top_p, top_k=self.top_k)
        if prefill is not None:
            kw["prefill"] = prefill
        ctx = self._frozen_base()
        with ctx:
            return self.adapter.understand(images, question, **kw)

    def _needs_finalizer(self, raw: str, res: AuditResult) -> bool:
        r = raw or ""
        if r.lstrip().startswith("<think>") and "</think>" not in r:
            return True
        if not (res.reason or "").strip() and "{" not in _strip_own_think(r):
            return True
        return False

    def _section_between(self, text: str, start_pat: str, end_pat: str) -> str:
        m = re.search(start_pat + r"\n(.*?)(?:\n\n" + end_pat + r"|\Z)", text or "", re.S)
        return (m.group(1).strip() if m else "")[:60000]

    def _compact_finalizer_context(self, question: str, *, is_image: bool) -> str:
        task = self._section_between(question, r"TASK:", r"SOLUTION TEXT|WHAT TO CHECK|Core rules|Highest-priority|$")
        if is_image:
            return f"TASK:\n{task}\n"
        sol = self._section_between(question, r"SOLUTION TEXT[^\n]*:", r"STRUCTURE|Core rules|Highest-priority")
        return f"TASK:\n{task}\n\nSOLUTION TEXT:\n{sol}\n"

    def _finalizer_prompt(self, question: str, truncated_raw: str, *, is_image: bool) -> str:
        tail = (truncated_raw or "")[-6000:]
        if tail:
            common = (
                "The previous audit reasoning was truncated by a token cap. Do NOT restart the review. "
                "Do NOT summarize the whole solution/sequence. Do NOT list multiple issues. "
                "Continue from the last concrete suspicious point in the truncated reasoning; if it mentions "
                "a concrete possible error, decide that issue first. Only return valid=true if the relevant "
                "high-risk checks below are actually cleared.\n\n"
                "TRUNCATED REASONING EXCERPT:\n"
                f"{tail}\n\n"
            )
        else:
            common = ""
        if is_image:
            rules = (
                "IMAGE FINALIZER TASK:\n"
                "The TASK is the user's target specification. The IMAGE SEQUENCE is the model's visual answer. Decide whether the visible images satisfy the TASK well enough as a visual sequence. Step text is not shown, so do not assume hidden explanations.\n"
                "Judge only what you can actually see. Do not say a number, object, or relation is correct unless it is visibly present and readable. Do not reject small aesthetic issues, lighting/color shifts, clothing or scene changes, flashbacks, missing anchor objects, or two similar but distinct props.\n"
                "IMAGE 1 — classify it, do not deliberate. The two cases are mutually exclusive and fixed by what you see, because a genuine first step cannot depict the finished result: if image 1 shows the finished result, then it IS the cover/goal preview; if image 1 shows a starting state, then it IS the first step. There is no third reading.\n"
                "  - Cover case: 'image 1 shows the finished result' is then NOT an error and must not be reported as one under any wording ('final state out of order', 'regression', 'non-chronological', 'premature final state', 'not a logical progression' are the same forbidden claim here). Judge the rest of the sequence on its own.\n"
                "  - First-step case: judge image 1 like any other image, including whether image 2 advances beyond it.\n"
                "'Intermediate' below means images that are neither the cover nor the last one — the premature-finish rule never applies to a cover or to the last image.\n"
                "Return invalid only for a concrete visible problem: wrong/garbled numbers or labels, a diagram that visibly fails to support required quantities, an intermediate cooking image that already shows a later cooked/finished state, adjacent near-duplicate story frames with no visible progress, a key prop drawn as the wrong kind of object, or a badly broken key face/text/structure.\n"
                "Concrete examples:\n"
                "- If a tutorial's FIRST image is the finished plated dish and the second image goes back to raw ingredients, that is the normal goal-preview opening: do NOT call it a regression, a premature final state, or a non-chronological sequence.\n"
                "- If a tomato-and-egg tutorial is at the middle step 'slice tomatoes' or 'beat eggs', but that same middle-step image already shows cooked tomato-and-egg stir-fry, return invalid. The issue is not plating; the issue is that the middle step shows an already-cooked later state instead of the current state.\n"
                "- If a pyramid problem requires layers of 1, 3, 9, and 27 blocks, but the image only visibly shows a small vague pile with far fewer blocks, do not call it correct; return invalid.\n"
                "- If two breakup-story images show the same room, same two people, same seated pose, and no new visible action, return invalid for near-duplicate frames.\n"
                "- If a beach story shows a red bucket in one image and a blue bucket in another, do not call it a color error if they could be two different buckets.\n"
                "If you find a real visible error, return invalid and explain only that first error. If you do not find a concrete visible error, return valid and mention the specific visual risks you checked; do not merely say the sequence is coherent.\n"
            )
            contract = '{"reason": "<=2 sentences; cite image number if invalid", "valid": true | false}'
        else:
            is_nothink_text = "no-think interleaved text-and-image response" in (question or "")
            if is_nothink_text:
                rules = (
                    "NO-THINK TEXT FINALIZER TASK:\n"
                    "The TASK is the user's target specification. The SOLUTION TEXT is the visible no-think answer with bare <image> markers. Images are not shown here; judge only the written content and marker structure.\n"
                    "A bare <image> marker means an image is generated at that position. Do not fail the text because pixels are absent, visual quality is unknown, or because the answer has no <think> reasoning.\n"
                    "If the task says to say/name/state/describe something after a drawing, accept that information when it is clearly attached to the same image unit: before the marker, after the marker, or in the same paragraph/section. Do not require a rigid label or literal post-marker sentence.\n"
                    "Enforce explicit marker constraints: 'exactly one image' requires exactly one bare <image>; tasks asking for N images/views/scenes/steps/illustrations require N markers unless the task allows otherwise. If the task says to stop after one image, text after the single marker must not introduce another scene, next step, or second visual plan.\n"
                    "For tutorials or recipes, cover/final-result previews, ingredient overviews, setup images, and style notes before the main instructions are acceptable. 'Each key step' means each main instructional action should have a nearby image marker; introductory paragraphs and ingredient lists do not each need their own marker.\n"
                    "Repeated or near-duplicate wording is not an error by itself if the main progression remains clear and no required state is contradicted. Do not fail for minor wording, future tense, harmless continuity wording, or missing minor descriptive details when the required moment is otherwise covered.\n"
                    "Return invalid only for a concrete text-side error: explicit marker-count/stop violation, missing required verbal facts, required objects/stages not covered, clear internal contradiction, skipped main event, impossible order, unexplained regression, or a required ingredient/tool/action clearly never used.\n"
                    "If the truncated reasoning ends on an unresolved concrete suspicion, finish checking THAT point against the SOLUTION TEXT before deciding.\n"
                    "If you find a real text error, return invalid and explain only that first error. If you do not find a concrete text error, return valid and mention the specific check you cleared.\n"
                )
            else:
                rules = (
                    "TEXT FINALIZER TASK:\n"
                    "The TASK is the user's target specification. The SOLUTION TEXT is the model's generated answer. Images are not shown. Decide whether the text content satisfies the TASK and is internally consistent.\n"
                    "Do not reject formatting, verbosity, planning paragraphs, image markers, <imageN> labels, marker counts, opening goal previews, final results in the final step/answer, anchor objects absent from some scenes, or outside world knowledge not stated by the task.\n"
                    "Narrative-vs-directive differences: a step's image directive may differ from that step's text in pure presentation — pose, camera angle, framing, lighting, which hand holds what, wording — and that is exempt. It is NOT exempt when the directive depicts a PROCESSED STATE that only exists after LATER steps: e.g. the text says raw/whole/unmixed/undressed/unbaked while the directive says sliced, seared, cooked, browned, charred, assembled, garnished, frosted, glazed, or already-mixed, or the directive puts into a bowl/pan/tray something that a later step is supposed to produce. That is future-state leakage and you must return invalid for it. Compare each step's own text against its own directive before you clear this check.\n"
                    "An opening image or paragraph that previews the finished result is always fine — a tutorial may either open with a preview of the final dish or go straight into step 1; both are normal. That exemption covers the OPENING only. 'Step' here means a real numbered tutorial step or story scene, not the model's planning block: planning is never itself an error.\n"
                    "If the truncated reasoning ends on an unresolved concrete suspicion (for example it was in the middle of noting that a listed ingredient never gets used, or that two steps disagree), you must finish checking THAT point against the SOLUTION TEXT before deciding. Never declare such a scan cleared just because the earlier reasoning did not finish it — if the suspicion holds, return invalid for it.\n"
                    "Return invalid only for a real text-content problem: a required action never delivered, wrong or missing final answer, internal contradiction, unexplained time/state reversal, non-exempt middle-step future-state leakage, a listed ingredient or tool that no later step ever uses, or explicit task constraint not delivered.\n"
                    "Concrete examples:\n"
                    "- If a story first says a festival has already reached the next-day celebration or climax, but a later step goes back to the previous evening's dinner/preparation without explicitly saying flashback or memory, return invalid.\n"
                    "- If a later step says night has arrived but a following step says afternoon begins, return invalid unless it explicitly says time jump or flashback.\n"
                    "- If a tomato-and-egg tutorial has a middle step such as 'slice tomatoes' or 'beat eggs', but that same middle step describes already cooked tomato-and-egg stir-fry, return invalid because the middle step does not show its current state.\n"
                    "- If a preparation step's image directive already describes the ingredients as sliced/diced when the slicing happens in a LATER step, return invalid for future-state leakage.\n"
                    "- If the ingredient list includes celery but no cooking step ever adds or cooks the celery, return invalid even if the plating step mentions it in passing.\n"
                    "- If one image directive says the subject is 'seen from behind' and also 'moving toward the viewer' in the same shot, return invalid for internal contradiction.\n"
                    "If you find a real content error, return invalid and explain only that first error. If you do not find a concrete content error, return valid and mention the specific high-risk checks you cleared; do not merely say the text is coherent.\n"
                )
            contract = (
                '{"quote": "if invalid, one span copied character for character out of the '
                'SOLUTION TEXT above — never from the TASK line and never from these '
                'instructions. Do not paraphrase, summarize, or describe: the span must be '
                'findable in the SOLUTION TEXT by exact search. Make it long enough to occur '
                'only once (one line is usually enough); when the fault is that something is '
                'missing or never used, quote the line where that item DOES appear. Write null '
                'when no single span carries the fault — for example the complaint is about how '
                'the whole response is arranged, or the fault only exists between two places far '
                'apart. Never invent a span to fill this field. Empty string when valid", '
                '"reason": "<=2 sentences; cite step/location if invalid", '
                '"valid": true | false}'
            )
        ctx = self._compact_finalizer_context(question, is_image=is_image)
        return f"{ctx}\n\n{common}{rules}\nReturn exactly this JSON shape and nothing else:\n{contract}"

    def _finalize_once(self, images: Sequence[Any], question: str, truncated_raw: str,
                       *, temperature: float = 0) -> AuditResult:
        final_q = self._finalizer_prompt(question, truncated_raw, is_image=bool(images))
        if temperature > 0:
            old_temp = self.temperature
            self.temperature = temperature
            old_sample = self.audit_sample
            self.audit_sample = True
            try:
                raw2 = self._raw_understand(images, final_q, think=False, sample=False,
                                            max_new_tokens=self.finalizer_tokens)
            finally:
                self.temperature = old_temp
                self.audit_sample = old_sample
        else:
            raw2 = self._raw_understand(images, final_q, think=False, sample=False,
                                        max_new_tokens=self.finalizer_tokens)
        res = parse_audit(raw2)
        res.raw = (truncated_raw or "") + "\n</think>\n[finalizer]\n" + raw2
        return res

    def _audit_once(self, images: Sequence[Any], question: str, sample: bool, max_new_tokens: int) -> AuditResult:
        cap = self.think_cap_image if images else self.think_cap_text
        first_budget = min(max_new_tokens, cap) if self.think and cap > 0 else max_new_tokens
        raw = self._raw_understand(
            images,
            question,
            think=self.think,
            sample=sample,
            max_new_tokens=first_budget,
        )
        res = parse_audit(raw)
        if self.think and cap > 0 and self._needs_finalizer(raw, res):
            closed_raw = raw.rstrip() + "\n"
            res_closed = parse_audit(closed_raw)
            if res_closed.reason and res_closed.reason.strip():
                res_closed.raw = raw
                return res_closed
            return self._finalize_once(images, question, raw)
        return res

    def _run(self, images: Sequence[Any], question: str) -> AuditResult:
        self.last_audit_retried = False
        if not self.think and getattr(self, 'skip_to_finalizer', False):
            return self._finalize_once(images, question, "")
        res = self._audit_once(images, question, False, self.max_new_tokens)
        if self.retry_on_empty and _needs_retry(res):
            self.last_audit_retried = True
            retry = self._finalize_once(images, question, "", temperature=0.7)
            if not _needs_retry(retry):
                res = retry
        return res

    def audit_text(self, prompt: str, text_chain: str, *, think_mode: bool = True) -> AuditResult:
        return self._run([], self.P.text_audit_prompt(prompt, text_chain, think_mode=think_mode))

    def audit_image(self, prompt: str, images: Sequence[Any]) -> AuditResult:
        return self._run(list(images), self.P.image_audit_prompt(prompt, len(images)))

    def _desc_raw(self, images, question: str, max_new_tokens: int, temperature: float = 0.6,
                  think: Optional[bool] = None) -> str:
        with self._frozen_base():
            raw = self.adapter.understand(
                images, question, think=(self.desc_think if think is None else think),
                do_sample=True, temperature=temperature, top_p=0.95, top_k=20, repetition_penalty=1.05,
                max_new_tokens=max_new_tokens,
            )
        self.last_desc_raw = raw or ""
        return self.last_desc_raw

    def _desc_once(self, images, question: str, max_new_tokens: int, temperature: float = 0.6) -> str:
        from .parse import _strip_own_think

        raw = self._desc_raw(images, question, max_new_tokens, temperature)
        body = _strip_own_think(raw)
        desc = extract_description(body)
        if desc is None and "<description>" in (body or "").lower():
            i = body.lower().index("<description>") + len("<description>")
            desc = body[i:]
        if not desc and "<slot" in (body or "").lower():
            desc = body
        return desc or ""

    def _slots_ok(self, desc: str, n: int) -> bool:
        from .parse import parse_slots_lenient
        sl = parse_slots_lenient(desc or "")
        return len(sl) == n

    def text_desc(self, prompt: str, text_chain: str, findings: str, *, think: bool = True) -> str:
        n = text_chain.count("<image>")
        q = self.P.text_desc_prompt(prompt, text_chain, findings, think=think)
        self.last_desc_attempts = 1
        desc = self._desc_once([], q, self.desc_max_tokens)
        if n and not self._slots_ok(desc, n):
            self.last_desc_attempts = 2
            desc = self._desc_once([], q, self.desc_max_tokens, temperature=0.7)
        return desc

    def image_desc(self, prompt: str, images: Sequence[Any], findings: str,
                   *, think: bool = True, temperature: float = 0.6) -> str:
        from .parse import image_desc_to_tplus

        n = len(images)
        q = self.P.image_desc_prompt(prompt, n, findings, think=think)
        self.last_desc_attempts = 1
        raw = self._desc_raw(list(images), q, self.desc_max_tokens, temperature=temperature)
        tplus = image_desc_to_tplus(raw, n, think=think)
        if n and not tplus:
            self.last_desc_attempts = 2
            retry_temp = min(1.0, max(0.7, temperature + 0.1))
            raw = self._desc_raw(list(images), q, self.desc_max_tokens, temperature=retry_temp)
            tplus = image_desc_to_tplus(raw, n, think=think)
        return tplus

    def repair_span(self, prompt: str, images: Sequence[Any], reason: str,
                    passage: str, span: tuple, *, temperature: float = 0.6) -> Optional[str]:
        from .parse import _strip_own_think

        q = self.P.span_repair_prompt(prompt, reason, passage, span)
        t0 = time.time()
        raw = self._desc_raw(list(images), q, self.repair_tokens, temperature=temperature, think=False)
        self.last_repair_seconds = round(time.time() - t0, 1)
        body = _strip_own_think(raw or "").strip()
        m = re.search(r"<replacement>(.*?)(?:</replacement>|\Z)", body, re.S | re.I)
        new = (m.group(1) if m else body).strip()
        new = re.sub(r"</?replacement>", "", new, flags=re.I).strip()
        return new or None

    def final_answer(self, prompt: str, think_text: str, n_images: int, findings: str,
                     orig_answer: str, *, base_temperature: float = 0.6) -> str:
        from .parse import _strip_own_think

        q = self.P.final_answer_prompt(prompt, think_text, n_images, findings, orig_answer)
        t0 = time.time()
        self.last_answer_attempts = []
        ans = ""
        for k in range(max(1, self.answer_tries)):
            temp = base_temperature if k == 0 else min(1.0, base_temperature + 0.1)
            cand = _strip_own_think(self._desc_raw(
                [], q, ANSWER_TOKENS, temperature=temp, think=False) or "").strip()
            cand = _BARE_IMG_RE.sub("", cand).strip()
            self.last_answer_attempts.append(cand)
            if not n_images or _refs_complete(cand, n_images):
                ans = cand
                break
            ans = ""
        self.last_answer_seconds = round(time.time() - t0, 1)
        self.last_answer_budget = ANSWER_TOKENS
        return ans


class ExternalVerdictJudge(Judge):

    def __init__(self, target_judge: SelfJudge, client: Any) -> None:
        self.target_judge = target_judge
        self.client = client
        self.last_audit_retried = False

    def audit_text(self, prompt: str, text_chain: str, *, think_mode: bool = True) -> AuditResult:
        result = self.client.audit_text(prompt, text_chain, think_mode=think_mode)
        self.last_audit_retried = bool(getattr(self.client, "last_audit_retried", False))
        return result

    def audit_image(self, prompt: str, images: Sequence[Any]) -> AuditResult:
        result = self.client.audit_image(prompt, images)
        self.last_audit_retried = bool(getattr(self.client, "last_audit_retried", False))
        return result

    def text_desc(self, prompt: str, text_chain: str, findings: str, *, think: bool = True) -> str:
        return self.target_judge.text_desc(prompt, text_chain, findings, think=think)

    def image_desc(self, prompt: str, images: Sequence[Any], findings: str,
                   *, think: bool = True, temperature: float = 0.6) -> str:
        return self.target_judge.image_desc(prompt, images, findings, think=think, temperature=temperature)

    def repair_span(self, prompt: str, images: Sequence[Any], reason: str,
                    passage: str, span: tuple, *, temperature: float = 0.6) -> Optional[str]:
        return self.target_judge.repair_span(prompt, images, reason, passage, span, temperature=temperature)

    def final_answer(self, prompt: str, think_text: str, n_images: int, findings: str,
                     orig_answer: str, *, base_temperature: float = 0.6) -> str:
        return self.target_judge.final_answer(
            prompt, think_text, n_images, findings, orig_answer, base_temperature=base_temperature
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target_judge, name)


def build_judge(cfg: Any, adapter: Any) -> Judge:
    prompt_module = None
    if adapter.__class__.__name__ == "BagelAdapter":
        from . import prompts_bagel as prompt_module

    target_judge = SelfJudge(
        adapter,
        temperature=getattr(cfg, "temperature", 0.7),
        max_new_tokens=getattr(cfg, "max_new_tokens", 1024),
        retry_on_empty=getattr(cfg, "retry_on_empty", True),
        think=getattr(cfg, "think", True),
        think_cap_text=getattr(cfg, "think_cap_text", 0),
        think_cap_image=getattr(cfg, "think_cap_image", 0),
        finalizer_tokens=getattr(cfg, "finalizer_tokens", 1024),
        repair_tokens=getattr(cfg, "repair_tokens", 512),
        answer_tries=getattr(cfg, "answer_tries", 4),
        prompt_module=prompt_module,
    )
    provider = getattr(cfg, "verdict_provider", "self")
    if provider == "self":
        return target_judge
    if provider == "external":
        from .external_judge import ExternalVerdictClient

        client = ExternalVerdictClient(
            prompt_module=target_judge.P,
            max_new_tokens=getattr(cfg, "external_max_new_tokens", 8192),
        )
        return ExternalVerdictJudge(target_judge, client)
    raise ValueError(f"unknown audit verdict provider: {provider!r}")


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> Optional[str]:
    t = text or ""
    if "####" in t:
        t = t.split("####", 1)[1]
    nums = _NUM_RE.findall(t)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def gsm8k_correct(pred_text: str, gt_answer: str) -> bool:
    p, g = extract_final_number(pred_text), extract_final_number(gt_answer)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p == g
