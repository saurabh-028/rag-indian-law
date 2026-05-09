"""
Language detection and translation for the Indian Law RAG system.
Uses OpenAI (gpt-4o-mini) for translation to handle legal terminology correctly.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

LANGUAGE_CONFIG: dict[str, dict] = {
    "hi": {"name": "Hindi"},
    "mr": {"name": "Marathi"},
}

SUPPORTED_LANGS = set(LANGUAGE_CONFIG.keys())

LANG_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "en": "English",
}


def detect_language(text: str) -> str:
    """Detect language of text. Returns BCP-47 code, falls back to 'en'."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        logger.warning("Language detection failed; defaulting to 'en'.")
        return "en"


class Translator:
    """OpenAI-powered translator using gpt-4o-mini."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)
        client = self._get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a precise legal translator. "
                        f"Translate the following {src_name} text to {tgt_name}. "
                        f"Preserve all legal terms, section numbers, and Act names exactly. "
                        f"Return only the translation, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def to_english(self, text: str, src_lang: str) -> str:
        """Translate text from src_lang to English."""
        if src_lang not in LANGUAGE_CONFIG:
            logger.warning(f"Unsupported lang '{src_lang}'; returning original text.")
            return text
        return self._translate(text, src_lang=src_lang, tgt_lang="en")

    def from_english(self, text: str, tgt_lang: str) -> str:
        """Translate text from English to tgt_lang."""
        if tgt_lang not in LANGUAGE_CONFIG:
            logger.warning(f"Unsupported lang '{tgt_lang}'; returning original text.")
            return text
        return self._translate(text, src_lang="en", tgt_lang=tgt_lang)

    def is_supported(self, lang: str) -> bool:
        return lang in LANGUAGE_CONFIG


translator = Translator()
