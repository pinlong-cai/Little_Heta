"""Embedding calls for the memory module."""

from __future__ import annotations

from heta.mem.client import EMBEDDING_DIM
from heta.providers.model_protocols import EmbeddingModelProtocol, EmbeddingRequest


def embed_text(embedding_model: EmbeddingModelProtocol, text: str) -> list[float]:
    response = embedding_model.embed(EmbeddingRequest(texts=[text]))
    vector = response.vectors[0]
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding model {embedding_model.model_name} returned {len(vector)} dimensions; "
            f"Little Heta requires {EMBEDDING_DIM}."
        )
    return vector


def fact_text(subject: str, predicate: str, object_: str) -> str:
    """Convert a triple to a natural language string for embedding."""
    return f"{subject} {predicate.replace('_', ' ')} {object_}"
