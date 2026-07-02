"""Model factories for configured LLM capabilities."""

from __future__ import annotations

from typing import Any

from heta.config.schema import HetaConfig, LLMProvider
from heta.providers.litellm_models import LiteLLMChatModel, LiteLLMEmbeddingModel
from heta.providers.model_protocols import (
    ChatModelConfig,
    ChatModelProtocol,
    EmbeddingModelConfig,
    EmbeddingModelProtocol,
)

EMBEDDING_DIM = 1024

_GEMINI_OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def build_chat_model(
    config: HetaConfig,
    *,
    timeout: int = 60,
    max_retries: int | None = None,
) -> LiteLLMChatModel:
    """Return the configured text chat model."""
    model_name = _required(config.llm.chat_model, "chat_model")
    return LiteLLMChatModel(
        ChatModelConfig(
            model_name=resolve_litellm_model_name(
                provider=config.llm.provider,
                model_name=model_name,
                api_base=config.llm.chat_base_url,
            ),
            api_key=_required(config.llm.chat_api_key, "chat_api_key"),
            api_base=_effective_api_base(config.llm.provider, config.llm.chat_base_url),
            request_timeout=timeout,
            max_retries=_max_retries_or_default(max_retries),
            provider_options=chat_provider_options(config),
        )
    )


def build_multimodal_model(
    config: HetaConfig,
    *,
    timeout: int = 300,
    max_retries: int | None = None,
) -> LiteLLMChatModel:
    """Return the configured multimodal chat model."""
    model_name = _required(config.llm.multimodal_model, "multimodal_model")
    return LiteLLMChatModel(
        ChatModelConfig(
            model_name=resolve_litellm_model_name(
                provider=config.llm.provider,
                model_name=model_name,
                api_base=config.llm.multimodal_base_url,
            ),
            api_key=_required(config.llm.multimodal_api_key, "multimodal_api_key"),
            api_base=_effective_api_base(config.llm.provider, config.llm.multimodal_base_url),
            request_timeout=timeout,
            max_retries=_max_retries_or_default(max_retries),
            provider_options=multimodal_provider_options(config),
        )
    )


def build_embedding_model(
    config: HetaConfig,
    *,
    timeout: int = 120,
    max_retries: int | None = None,
) -> LiteLLMEmbeddingModel:
    """Return the configured fixed-dimension embedding model."""
    model_name = _required(config.llm.embedding_model, "embedding_model")
    resolved_model_name = resolve_litellm_model_name(
        provider=config.llm.provider,
        model_name=model_name,
        api_base=config.llm.embedding_base_url,
    )
    return LiteLLMEmbeddingModel(
        EmbeddingModelConfig(
            model_name=resolved_model_name,
            api_key=_required(config.llm.embedding_api_key, "embedding_api_key"),
            api_base=_effective_api_base(config.llm.provider, config.llm.embedding_base_url),
            request_timeout=timeout,
            max_retries=_max_retries_or_default(max_retries),
            dimensions=default_embedding_request_dimensions(resolved_model_name),
            provider_options=embedding_provider_options(config),
        )
    )


def resolve_litellm_model_name(*, provider: LLMProvider, model_name: str, api_base: str | None) -> str:
    """Resolve a configured model name into LiteLLM provider/model syntax."""
    normalized = model_name.strip()
    if "/" in normalized:
        return normalized
    if provider == "gemini":
        return f"gemini/{normalized}"
    if provider == "custom" and not api_base:
        raise ValueError(
            "Custom LLM model names without a LiteLLM provider prefix require a base_url."
        )
    return f"openai/{normalized}"


def chat_provider_options(config: HetaConfig) -> dict[str, Any] | None:
    """Return provider-specific default options for text chat requests."""
    if config.llm.chat_extra_body is not None:
        return dict(config.llm.chat_extra_body)
    if config.llm.provider == "qwen":
        return {"enable_thinking": False}
    return None


def multimodal_provider_options(config: HetaConfig) -> dict[str, Any] | None:
    """Return provider-specific default options for multimodal requests."""
    return chat_provider_options(config)


def embedding_provider_options(config: HetaConfig) -> dict[str, Any] | None:
    """Return provider-specific default options for embedding requests."""
    return None


def default_embedding_request_dimensions(model_name: str) -> int | None:
    """Return request-time dimensions only for LiteLLM models known to accept it."""
    if model_name.startswith("openai/text-embedding-3"):
        return EMBEDDING_DIM
    return None


def extra_body(config: HetaConfig) -> dict[str, Any] | None:
    """Compatibility alias for older chat completion call sites."""
    return chat_provider_options(config)


def build_chat_client(
    config: HetaConfig,
    *,
    timeout: int = 60,
    max_retries: int | None = None,
) -> ChatModelProtocol:
    """Compatibility alias for the previous OpenAI-compatible chat factory."""
    return build_chat_model(config, timeout=timeout, max_retries=max_retries)


def build_multimodal_client(
    config: HetaConfig,
    *,
    timeout: int = 300,
    max_retries: int | None = None,
) -> ChatModelProtocol:
    """Compatibility alias for the previous OpenAI-compatible multimodal factory."""
    return build_multimodal_model(config, timeout=timeout, max_retries=max_retries)


def build_embedding_client(
    config: HetaConfig,
    *,
    timeout: int = 120,
    max_retries: int | None = None,
) -> EmbeddingModelProtocol:
    """Compatibility alias for the previous OpenAI-compatible embedding factory."""
    return build_embedding_model(config, timeout=timeout, max_retries=max_retries)


def _effective_api_base(provider: LLMProvider, api_base: str | None) -> str | None:
    if provider == "gemini" and api_base is not None:
        normalized = api_base.rstrip("/")
        if normalized == _GEMINI_OPENAI_COMPATIBLE_BASE_URL:
            return None
    return api_base


def _max_retries_or_default(max_retries: int | None) -> int:
    if max_retries is None:
        return 3
    return max_retries


def _required(value: str | None, field: str) -> str:
    if not value:
        raise ValueError(f"Missing LLM {field} in config.")
    return value


__all__ = [
    "EMBEDDING_DIM",
    "build_chat_client",
    "build_chat_model",
    "build_embedding_client",
    "build_embedding_model",
    "build_multimodal_client",
    "build_multimodal_model",
    "chat_provider_options",
    "default_embedding_request_dimensions",
    "embedding_provider_options",
    "extra_body",
    "multimodal_provider_options",
    "resolve_litellm_model_name",
]
