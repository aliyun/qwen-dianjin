from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from . import prompts
from .parse import parse_audit


class ExternalVerdictError(RuntimeError):
    pass


class ExternalVerdictClient:
    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        prompt_module: Any = None,
        max_new_tokens: int = 8192,
        timeout: int = 300,
        retries: int = 4,
    ) -> None:
        self.api_url = api_url or os.environ.get("PRISM_EXT_JUDGE_URL", "")
        self.api_key = api_key or os.environ.get("PRISM_EXT_JUDGE_KEY", "")
        self.model = model or os.environ.get("PRISM_EXT_JUDGE_MODEL", "")
        missing = [name for name, value in (
            ("PRISM_EXT_JUDGE_URL", self.api_url),
            ("PRISM_EXT_JUDGE_KEY", self.api_key),
            ("PRISM_EXT_JUDGE_MODEL", self.model),
        ) if not value]
        if missing:
            raise ExternalVerdictError(f"missing external judge configuration: {', '.join(missing)}")
        self.prompt_module = prompt_module or prompts
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = int(timeout)
        self.retries = max(1, int(retries))
        self.last_audit_retried = False

    @staticmethod
    def _image_data_url(image: Any) -> str:
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                rgb = opened.convert("RGB")
                return ExternalVerdictClient._jpeg_data_url(rgb)
        if not callable(getattr(image, "convert", None)):
            raise TypeError(f"external image audit requires a PIL image or path, got {type(image).__name__}")
        return ExternalVerdictClient._jpeg_data_url(image.convert("RGB"))

    @staticmethod
    def _jpeg_data_url(image: Any) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return "data:image/jpeg;base64," + encoded

    def _chat(self, content: Any) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode()
        last_error = ""
        self.last_audit_retried = False
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    self.api_url,
                    data=data,
                    headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read())
                return decoded["choices"][0]["message"]
            except urllib.error.HTTPError as exc:
                body = exc.read()[:300].decode("utf-8", "replace")
                last_error = f"HTTP {exc.code}: {body}"
                retryable = exc.code == 429 or exc.code >= 500
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = True
            if not retryable or attempt + 1 == self.retries:
                break
            self.last_audit_retried = True
            time.sleep(min(30, 3 * (2 ** attempt)))
        raise ExternalVerdictError(f"external judge request failed after {self.retries} attempt(s): {last_error}")

    @staticmethod
    def _parse(message: dict):
        content = message.get("content") or ""
        result = parse_audit(content)
        result.raw = content
        return result

    def audit_text(self, prompt: str, text_chain: str, *, think_mode: bool = True):
        question = self.prompt_module.text_audit_prompt(prompt, text_chain, think_mode=think_mode)
        return self._parse(self._chat(question))

    def audit_image(self, prompt: str, images: Sequence[Any]):
        content = [
            {"type": "image_url", "image_url": {"url": self._image_data_url(image)}}
            for image in images
        ]
        question = self.prompt_module.image_audit_prompt(prompt, len(images))
        content.append({"type": "text", "text": question})
        return self._parse(self._chat(content))
