"""OpenAI-compatible client factories for configured LLM capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from heta.config.schema import HetaConfig

EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class ModelClient:
    client: OpenAI
    model: str


def build_chat_client(config: HetaConfig, *, timeout: int = 60, max_retries: int | None = None) -> ModelClient:
    """Return the text chat client and model for the configured provider."""
    model = _required(config.llm.chat_model, "chat_model")
    return ModelClient(
        client=_openai_client(
            api_key=_required(config.llm.chat_api_key, "chat_api_key"),
            base_url=config.llm.chat_base_url,
            timeout=timeout,
            max_retries=max_retries,
        ),
        model=model,
    )


def build_multimodal_client(config: HetaConfig, *, timeout: int = 300) -> ModelClient:
    """Return the multimodal client and model for image/audio-capable calls."""
    model = _required(config.llm.multimodal_model, "multimodal_model")
    return ModelClient(
        client=_openai_client(
            api_key=_required(config.llm.multimodal_api_key, "multimodal_api_key"),
            base_url=config.llm.multimodal_base_url,
            timeout=timeout,
        ),
        model=model,
    )


def build_embedding_client(config: HetaConfig, *, timeout: int = 120) -> ModelClient:
    """Return the embedding client and fixed-dimension embedding model."""
    model = _required(config.llm.embedding_model, "embedding_model")
    return ModelClient(
        client=_openai_client(
            api_key=_required(config.llm.embedding_api_key, "embedding_api_key"),
            base_url=config.llm.embedding_base_url,
            timeout=timeout,
        ),
        model=model,
    )


def extra_body(config: HetaConfig) -> dict | None:
    """Return provider-specific request options for chat completions."""
    if config.llm.chat_extra_body is not None:
        return config.llm.chat_extra_body
    if config.llm.provider == "qwen":
        return {"enable_thinking": False}
    return None


def _openai_client(
    *,
    api_key: str,
    base_url: str | None,
    timeout: int,
    max_retries: int | None = None,
) -> OpenAI:
    kwargs: dict = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return OpenAI(**kwargs)


def _required(value: str | None, field: str) -> str:
    if not value:
        raise ValueError(f"Missing LLM {field} in config.")
    return value


__all__ = [
    "EMBEDDING_DIM",
    "ModelClient",
    "build_chat_client",
    "build_embedding_client",
    "build_multimodal_client",
    "extra_body",
]
