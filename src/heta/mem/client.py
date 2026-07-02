"""Model factories for the memory module."""

from __future__ import annotations

from heta.config.schema import HetaConfig
from heta.providers.clients import (
    EMBEDDING_DIM,
    build_chat_model as build_provider_chat_model,
    build_embedding_model as build_provider_embedding_model,
    extra_body,
)
from heta.providers.model_protocols import ChatModelProtocol, EmbeddingModelProtocol


def build_chat_model(config: HetaConfig) -> ChatModelProtocol:
    """Return the configured text generation model."""
    return build_provider_chat_model(config, timeout=60)


def build_embedding_model(config: HetaConfig) -> EmbeddingModelProtocol:
    """Return the configured embedding model."""
    return build_provider_embedding_model(config, timeout=120)


def build_client(config: HetaConfig) -> ChatModelProtocol:
    """Compatibility alias for older memory call sites."""
    return build_chat_model(config)


def build_embedding_client(config: HetaConfig) -> EmbeddingModelProtocol:
    """Compatibility alias for older memory call sites."""
    return build_embedding_model(config)


__all__ = [
    "EMBEDDING_DIM",
    "build_chat_model",
    "build_client",
    "build_embedding_client",
    "build_embedding_model",
    "extra_body",
]
