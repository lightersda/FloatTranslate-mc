"""Translation via a pluggable LLM provider, with a small in-memory cache."""
from __future__ import annotations

import base64
import io

from PIL import Image

from providers import get_provider

_SYSTEM = (
    "You are a translation engine embedded in a screen-translation tool. "
    "The input text comes from OCR of a screenshot, so it may contain minor "
    "recognition errors, broken words, or stray characters — silently correct "
    "obvious ones. Translate the text into {target}. "
    "Output ONLY the translation, with no quotes, no explanations, no "
    "transliteration, and no notes. Preserve line breaks where meaningful. "
    "If the text is already in the target language, return it unchanged."
)

_IMAGE_SYSTEM = (
    "你是屏幕翻译工具。识别截图中的文字，纠正明显识别错误（如 0/O、1/l），"
    "翻译成{target}。只输出译文，不要引号、解释、音译或原文。"
)


class Translator:
    def __init__(self, provider_id: str, api_key: str, model: str,
                 target_language: str):
        self._provider = get_provider(provider_id)
        if not api_key and self._provider.requires_key:
            raise ValueError("missing_api_key")
        self._api_key = api_key
        self._model = model
        self._target = target_language
        self._cache: dict[str, str] = {}

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text in self._cache:
            return self._cache[text]

        system = _SYSTEM.format(target=self._target)
        out = self._provider.translate(self._api_key, self._model, system, text)

        # Bound cache growth.
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[text] = out
        return out

    def translate_image(self, img: Image.Image) -> str:
        """One-shot path: send the screenshot straight to the model and get
        the translation back directly (no separate local OCR step).

        Before sending, the image is downscaled (longest edge capped at 1280px),
        converted to grayscale and JPEG-compressed: screen text is black-on-white,
        so this cuts both payload size and the vision-token count on local models
        without hurting accuracy.
        """
        img = img.convert("L")  # grayscale — screen text is B&W anyway
        img.thumbnail((1280, 1280))  # cap longest edge, keeps aspect ratio
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        system = _IMAGE_SYSTEM.format(target=self._target)
        user_text = f"识别图中文字并翻译成{self._target}，直接输出译文。"
        return self._provider.translate_image(
            self._api_key, self._model, system, image_b64, "image/jpeg",
            user_text)
