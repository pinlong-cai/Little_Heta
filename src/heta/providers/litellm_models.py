"""LiteLLM-backed implementations of the internal model protocols."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from heta.providers.model_protocols import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    ChatMessageLike,
    ChatModelConfig,
    ChatModelOptions,
    ChatModelRequestError,
    ChatModelResponseError,
    EmbeddingModelConfig,
    EmbeddingModelOptions,
    EmbeddingModelRequestError,
    EmbeddingModelResponseError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingUsage,
    TokenUsage,
)


class LiteLLMChatModel:
    """Chat model implementation backed by LiteLLM."""

    def __init__(self, config: ChatModelConfig) -> None:
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def api_base(self) -> str | None:
        return self._config.api_base

    @property
    def provider_options(self) -> dict[str, Any] | None:
        if self._config.provider_options is None:
            return None
        return dict(self._config.provider_options)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        options = request.options or ChatModelOptions()
        payload = self._build_payload(request, options)

        try:
            response = _load_litellm().completion(**payload)
        except Exception as exc:  # pragma: no cover - provider exceptions vary.
            raise ChatModelRequestError(
                f"Chat model request failed for {self.model_name}.",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc

        response_data = _to_mapping(response)
        try:
            choices = _field(response_data, "choices")
            first_choice = choices[0]
            message_data = _to_mapping(_field(first_choice, "message"))
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise ChatModelResponseError(
                f"Chat model response was malformed for {self.model_name}.",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc

        usage = _parse_token_usage(response_data.get("usage"))
        finish_reason = _optional_field(first_choice, "finish_reason")
        if finish_reason is not None:
            finish_reason = str(finish_reason)

        return ChatCompletionResult(
            message=ChatMessage.from_mapping(message_data),
            model_name=str(response_data.get("model") or self.model_name),
            usage=usage,
            finish_reason=finish_reason,
            trace_context=request.trace_context,
            raw_response=dict(response_data),
        )

    def _build_payload(self, request: ChatCompletionRequest, options: ChatModelOptions) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_message_to_provider_dict(message) for message in request.messages],
            "timeout": self._config.request_timeout,
            "num_retries": self._config.max_retries,
            "drop_params": self._config.drop_unsupported_params,
            "temperature": (
                options.temperature
                if options.temperature is not None
                else self._config.default_temperature
            ),
        }
        if self._config.api_key:
            payload["api_key"] = self._config.api_key
        if self._config.api_base:
            payload["api_base"] = self._config.api_base
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if options.max_output_tokens is not None:
            payload["max_tokens"] = options.max_output_tokens
        if options.top_p is not None:
            payload["top_p"] = options.top_p
        if options.stop_sequences is not None:
            payload["stop"] = options.stop_sequences
        if options.response_format is not None:
            payload["response_format"] = options.response_format

        payload.update(_merged_provider_options(self._config.provider_options, options.provider_options))
        return payload


class LiteLLMEmbeddingModel:
    """Embedding model implementation backed by LiteLLM."""

    def __init__(self, config: EmbeddingModelConfig) -> None:
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def api_base(self) -> str | None:
        return self._config.api_base

    @property
    def provider_options(self) -> dict[str, Any] | None:
        if self._config.provider_options is None:
            return None
        return dict(self._config.provider_options)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        options = request.options or EmbeddingModelOptions()
        payload = self._build_payload(request, options)

        try:
            response = _load_litellm().embedding(**payload)
        except Exception as exc:  # pragma: no cover - provider exceptions vary.
            raise EmbeddingModelRequestError(
                f"Embedding model request failed for {self.model_name}.",
                trace_context=request.trace_context,
                cause=exc,
            ) from exc

        response_data = _to_mapping(response)
        vectors = self._parse_vectors(response_data, expected_count=len(request.texts), trace_context=request.trace_context)
        return EmbeddingResult(
            vectors=vectors,
            model_name=str(response_data.get("model") or self.model_name),
            usage=_parse_embedding_usage(response_data.get("usage")),
            trace_context=request.trace_context,
            raw_response=dict(response_data),
        )

    def _build_payload(self, request: EmbeddingRequest, options: EmbeddingModelOptions) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": list(request.texts),
            "timeout": self._config.request_timeout,
            "num_retries": self._config.max_retries,
            "drop_params": self._config.drop_unsupported_params,
        }
        if self._config.api_key:
            payload["api_key"] = self._config.api_key
        if self._config.api_base:
            payload["api_base"] = self._config.api_base

        dimensions = options.dimensions if options.dimensions is not None else self._config.dimensions
        if dimensions is not None:
            payload["dimensions"] = dimensions

        encoding_format = options.encoding_format or self._config.encoding_format
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format

        payload.update(_merged_provider_options(self._config.provider_options, options.provider_options))
        return payload

    def _parse_vectors(
        self,
        response_data: Mapping[str, Any],
        *,
        expected_count: int,
        trace_context: dict[str, Any] | None,
    ) -> list[list[float]]:
        raw_data = response_data.get("data")
        if not isinstance(raw_data, list):
            raise EmbeddingModelResponseError(
                f"Embedding model response was malformed for {self.model_name}.",
                trace_context=trace_context,
            )

        try:
            indexed_items = sorted(
                (_to_mapping(item) for item in raw_data),
                key=lambda item: int(item.get("index", 0)),
            )
            vectors = [
                [float(value) for value in item["embedding"]]
                for item in indexed_items
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingModelResponseError(
                f"Embedding model response was malformed for {self.model_name}.",
                trace_context=trace_context,
                cause=exc,
            ) from exc

        if len(vectors) != expected_count:
            raise EmbeddingModelResponseError(
                f"Embedding model returned {len(vectors)} vectors for {expected_count} inputs.",
                trace_context=trace_context,
            )
        return vectors


def _load_litellm() -> Any:
    return importlib.import_module("litellm")


def _message_to_provider_dict(message: ChatMessageLike) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return message.to_provider_dict()
    if isinstance(message, Mapping):
        return dict(message)
    return asdict(message)


def _merged_provider_options(
    default_options: dict[str, Any] | None,
    request_options: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if default_options is not None:
        merged.update(default_options)
    if request_options is not None:
        merged.update(request_options)
    return merged


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"Expected mapping-like response, got {type(value).__name__}.")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _optional_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _parse_token_usage(value: Any) -> TokenUsage | None:
    if value is None:
        return None
    usage = _to_mapping(value)
    return TokenUsage(
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
    )


def _parse_embedding_usage(value: Any) -> EmbeddingUsage | None:
    if value is None:
        return None
    usage = _to_mapping(value)
    return EmbeddingUsage(
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "LiteLLMChatModel",
    "LiteLLMEmbeddingModel",
]
