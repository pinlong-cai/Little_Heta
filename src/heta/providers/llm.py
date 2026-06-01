"""LLM provider validation."""

from __future__ import annotations

from typing import Final

import requests

VALIDATION_TIMEOUT_SECONDS: Final[float] = 10.0


def validate_llm(provider: str, api_key: str, base_url: str | None = None) -> bool:
    """Validate that an LLM provider API key can reach its provider."""
    api_key = api_key.strip()
    if provider == "qwen":
        return _validate_bearer_models(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
            api_key,
        )
    if provider == "chatgpt":
        return _validate_bearer_models("https://api.openai.com/v1/models", api_key)
    if provider == "gemini":
        return _validate_gemini_models(api_key)
    if provider == "custom" and base_url:
        return _validate_bearer_models(base_url.rstrip("/") + "/models", api_key)
    return False


def _validate_bearer_models(url: str, api_key: str) -> bool:
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=VALIDATION_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def _validate_gemini_models(api_key: str) -> bool:
    try:
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=VALIDATION_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200
