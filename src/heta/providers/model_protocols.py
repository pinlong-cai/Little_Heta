"""Internal model protocols and result types for Little Heta."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by a model provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ChatToolCallFunction:
    """Function metadata for one chat tool call."""

    name: str
    arguments: str

    def to_provider_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChatToolCallFunction":
        return cls(name=str(data.get("name") or ""), arguments=str(data.get("arguments") or ""))


@dataclass(frozen=True)
class ChatToolCall:
    """One tool call requested by a chat model."""

    id: str
    function: ChatToolCallFunction
    type: str = "function"

    def to_provider_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_provider_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChatToolCall":
        raw_function = data.get("function") or {}
        if not isinstance(raw_function, Mapping):
            raw_function = {}
        return cls(
            id=str(data.get("id") or ""),
            type=str(data.get("type") or "function"),
            function=ChatToolCallFunction.from_mapping(raw_function),
        )


ChatContent = str | list[dict[str, Any]] | None


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat completion request or response."""

    role: str
    content: ChatContent = None
    tool_call_id: str | None = None
    tool_calls: list[ChatToolCall] = field(default_factory=list)

    def to_provider_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_provider_dict() for tool_call in self.tool_calls]
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChatMessage":
        raw_tool_calls = data.get("tool_calls") or []
        tool_calls = [
            ChatToolCall.from_mapping(item)
            for item in raw_tool_calls
            if isinstance(item, Mapping)
        ]
        return cls(
            role=str(data.get("role") or "assistant"),
            content=data.get("content"),
            tool_call_id=data.get("tool_call_id") if isinstance(data.get("tool_call_id"), str) else None,
            tool_calls=tool_calls,
        )


ChatMessageLike = ChatMessage | Mapping[str, Any]


@dataclass(frozen=True)
class ChatModelOptions:
    """Per-request chat model options."""

    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    response_format: str | dict[str, Any] | None = None
    provider_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatCompletionRequest:
    """One chat completion request."""

    messages: Sequence[ChatMessageLike]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    options: ChatModelOptions | None = None
    trace_context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty")


@dataclass(frozen=True)
class ChatCompletionResult:
    """Final result for one chat completion request."""

    message: ChatMessage
    model_name: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    trace_context: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatModelConfig:
    """Long-lived chat model configuration."""

    model_name: str
    api_key: str | None = None
    api_base: str | None = None
    request_timeout: float = 60
    max_retries: int = 3
    default_temperature: float = 0.2
    drop_unsupported_params: bool = True
    provider_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.model_name.strip() == "":
            raise ValueError("model_name must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")


@dataclass(frozen=True)
class EmbeddingUsage:
    """Token usage reported by an embedding model."""

    prompt_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingModelOptions:
    """Per-request embedding model options."""

    dimensions: int | None = None
    encoding_format: str | None = None
    provider_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")


@dataclass(frozen=True)
class EmbeddingRequest:
    """One embedding request."""

    texts: Sequence[str]
    options: EmbeddingModelOptions | None = None
    trace_context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("texts must not be empty")
        if any(text.strip() == "" for text in self.texts):
            raise ValueError("texts must not contain empty values")


@dataclass(frozen=True)
class EmbeddingResult:
    """Final result for one embedding request."""

    vectors: list[list[float]]
    model_name: str
    usage: EmbeddingUsage | None = None
    trace_context: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Long-lived embedding model configuration."""

    model_name: str
    api_key: str | None = None
    api_base: str | None = None
    request_timeout: float = 120
    max_retries: int = 3
    dimensions: int | None = None
    encoding_format: str | None = None
    drop_unsupported_params: bool = True
    provider_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.model_name.strip() == "":
            raise ValueError("model_name must not be empty")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")


class ChatModelError(RuntimeError):
    """Base error for chat model failures."""

    def __init__(
        self,
        message: str,
        *,
        trace_context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.trace_context = trace_context
        self.__cause__ = cause


class ChatModelRequestError(ChatModelError):
    """Raised when a chat model request fails."""


class ChatModelResponseError(ChatModelError):
    """Raised when a chat model response is malformed."""


class EmbeddingModelError(RuntimeError):
    """Base error for embedding model failures."""

    def __init__(
        self,
        message: str,
        *,
        trace_context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.trace_context = trace_context
        self.__cause__ = cause


class EmbeddingModelRequestError(EmbeddingModelError):
    """Raised when an embedding model request fails."""


class EmbeddingModelResponseError(EmbeddingModelError):
    """Raised when an embedding model response is malformed."""


@runtime_checkable
class ChatModelProtocol(Protocol):
    """Capability protocol for chat models."""

    @property
    def model_name(self) -> str:
        ...

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        ...


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Capability protocol for embedding models."""

    @property
    def model_name(self) -> str:
        ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        ...


__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResult",
    "ChatContent",
    "ChatMessage",
    "ChatMessageLike",
    "ChatModelConfig",
    "ChatModelError",
    "ChatModelOptions",
    "ChatModelProtocol",
    "ChatModelRequestError",
    "ChatModelResponseError",
    "ChatToolCall",
    "ChatToolCallFunction",
    "EmbeddingModelConfig",
    "EmbeddingModelError",
    "EmbeddingModelOptions",
    "EmbeddingModelProtocol",
    "EmbeddingModelRequestError",
    "EmbeddingModelResponseError",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingUsage",
    "TokenUsage",
]
