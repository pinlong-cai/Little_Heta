"""LLM and embedding client factories for the memory module."""

from __future__ import annotations

from openai import OpenAI

from heta.config.schema import HetaConfig
from heta.providers.clients import (
    EMBEDDING_DIM,
    build_chat_client,
    build_embedding_client as build_provider_embedding_client,
    extra_body,
)


def build_client(config: HetaConfig) -> tuple[OpenAI, str]:
    """Return (client, model) for text generation."""
    resolved = build_chat_client(config, timeout=60)
    return resolved.client, resolved.model


def build_embedding_client(config: HetaConfig) -> tuple[OpenAI, str]:
    """Return (client, model) for embedding generation."""
    resolved = build_provider_embedding_client(config, timeout=120)
    return resolved.client, resolved.model
