"""Typed configuration schema for Little Heta."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LLMProvider = Literal["qwen", "chatgpt", "gemini", "custom"]
MinerUProvider = Literal["cloud", "local"]

DEFAULT_LLM_PROFILES: dict[str, dict[str, str | None]] = {
    "qwen": {
        "chat_model": "qwen3.5-flash",
        "chat_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "multimodal_model": "qwen3.5-omni-flash",
        "multimodal_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_model": "text-embedding-v4",
        "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "chatgpt": {
        "chat_model": "gpt-5.4-nano",
        "chat_base_url": None,
        "multimodal_model": "gpt-5.4-nano",
        "multimodal_base_url": None,
        "embedding_model": "text-embedding-3-small",
        "embedding_base_url": None,
    },
    "gemini": {
        "chat_model": "gemini-2.5-flash",
        "chat_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "multimodal_model": "gemini-2.5-flash",
        "multimodal_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "embedding_model": "text-embedding-004",
        "embedding_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}


@dataclass(frozen=True)
class LLMConfig:
    provider: LLMProvider
    api_key: str
    chat_api_key: str | None = None
    chat_model: str | None = None
    chat_base_url: str | None = None
    chat_extra_body: dict[str, Any] | None = None
    multimodal_api_key: str | None = None
    multimodal_model: str | None = None
    multimodal_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    audio_api_key: str | None = None
    audio_model: str | None = None
    audio_base_url: str | None = None

    def __post_init__(self) -> None:
        defaults = DEFAULT_LLM_PROFILES.get(self.provider, {})
        for field in (
            "chat_api_key",
            "chat_model",
            "chat_base_url",
            "multimodal_api_key",
            "multimodal_model",
            "multimodal_base_url",
            "embedding_api_key",
            "embedding_model",
            "embedding_base_url",
            "audio_api_key",
            "audio_model",
            "audio_base_url",
        ):
            if getattr(self, field) is None and field in defaults:
                object.__setattr__(self, field, defaults[field])
        if self.provider != "custom":
            if self.chat_api_key is None:
                object.__setattr__(self, "chat_api_key", self.api_key)
            if self.multimodal_api_key is None:
                object.__setattr__(self, "multimodal_api_key", self.api_key)
            if self.embedding_api_key is None:
                object.__setattr__(self, "embedding_api_key", self.api_key)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        provider = data.get("provider")
        api_key = data.get("api_key")
        if provider not in {"qwen", "chatgpt", "gemini", "custom"}:
            raise ValueError("Invalid LLM provider in config.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Invalid LLM api_key in config.")

        defaults = DEFAULT_LLM_PROFILES.get(provider, {})
        values: dict[str, str | None] = {}
        for field in (
            "chat_api_key",
            "chat_model",
            "chat_base_url",
            "multimodal_api_key",
            "multimodal_model",
            "multimodal_base_url",
            "embedding_api_key",
            "embedding_model",
            "embedding_base_url",
            "audio_api_key",
            "audio_model",
            "audio_base_url",
        ):
            raw = data.get(field, defaults.get(field))
            if raw is not None and not isinstance(raw, str):
                raise ValueError(f"Invalid LLM {field} in config.")
            values[field] = raw.strip() if isinstance(raw, str) and raw.strip() else None

        chat_extra_body = data.get("chat_extra_body")
        if chat_extra_body is not None and not isinstance(chat_extra_body, dict):
            raise ValueError("Invalid LLM chat_extra_body in config.")

        if provider == "custom":
            missing = [
                field
                for field in (
                    "chat_api_key",
                    "chat_model",
                    "embedding_api_key",
                    "embedding_model",
                )
                if values[field] is None
            ]
            chat_model = values["chat_model"]
            embedding_model = values["embedding_model"]
            if chat_model is not None and "/" not in chat_model and values["chat_base_url"] is None:
                missing.append("chat_base_url")
            if (
                embedding_model is not None
                and "/" not in embedding_model
                and values["embedding_base_url"] is None
            ):
                missing.append("embedding_base_url")
            if missing:
                raise ValueError(f"Custom LLM config requires: {', '.join(missing)}.")

        return cls(provider=provider, api_key=api_key.strip(), chat_extra_body=chat_extra_body, **values)


@dataclass(frozen=True)
class MinerUConfig:
    enable: bool
    provider: MinerUProvider | None
    api_key: str | None
    endpoint: str | None

    @classmethod
    def disabled(cls) -> "MinerUConfig":
        return cls(enable=False, provider=None, api_key=None, endpoint=None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinerUConfig":
        enable = data.get("enable")
        provider = data.get("provider")
        api_key = data.get("api_key")
        endpoint = data.get("endpoint")

        if not isinstance(enable, bool):
            raise ValueError("Invalid MinerU enable flag in config.")
        if not enable:
            return cls.disabled()
        if provider not in {"cloud", "local"}:
            raise ValueError("Invalid MinerU provider in config.")
        if provider == "cloud" and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("MinerU cloud config requires api_key.")
        if provider == "local" and (not isinstance(endpoint, str) or not endpoint.strip()):
            raise ValueError("MinerU local config requires endpoint.")

        return cls(enable=True, provider=provider, api_key=api_key, endpoint=endpoint)


@dataclass(frozen=True)
class VectorIndexConfig:
    enable: bool

    @classmethod
    def enabled(cls) -> "VectorIndexConfig":
        return cls(enable=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorIndexConfig":
        enable = data.get("enable")
        if not isinstance(enable, bool):
            raise ValueError("Invalid vector_index enable flag in config.")
        return cls(enable=enable)


@dataclass(frozen=True)
class InsertPlanningConfig:
    enable: bool

    @classmethod
    def enabled(cls) -> "InsertPlanningConfig":
        return cls(enable=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsertPlanningConfig":
        enable = data.get("enable")
        if not isinstance(enable, bool):
            raise ValueError("Invalid insert_planning enable flag in config.")
        return cls(enable=enable)


@dataclass(frozen=True)
class DynamicInsertConfig:
    enable: bool

    @classmethod
    def disabled(cls) -> "DynamicInsertConfig":
        return cls(enable=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicInsertConfig":
        enable = data.get("enable")
        if not isinstance(enable, bool):
            raise ValueError("Invalid dynamic_insert enable flag in config.")
        return cls(enable=enable)


@dataclass(frozen=True)
class HetaConfig:
    version: int
    llm: LLMConfig
    mineru: MinerUConfig
    vector_index: VectorIndexConfig
    insert_planning: InsertPlanningConfig
    dynamic_insert: DynamicInsertConfig = field(default_factory=DynamicInsertConfig.disabled)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HetaConfig":
        version = data.get("version")
        if version != 1:
            raise ValueError("Unsupported config version.")
        llm = data.get("llm")
        mineru = data.get("mineru")
        vector_index = data.get("vector_index")
        insert_planning = data.get("insert_planning")
        dynamic_insert = data.get("dynamic_insert", {"enable": False})
        if not isinstance(llm, dict):
            raise ValueError("Missing LLM config.")
        if not isinstance(mineru, dict):
            raise ValueError("Missing MinerU config.")
        if not isinstance(vector_index, dict):
            raise ValueError("Missing vector_index config.")
        if not isinstance(insert_planning, dict):
            raise ValueError("Missing insert_planning config.")
        if not isinstance(dynamic_insert, dict):
            raise ValueError("Invalid dynamic_insert config.")
        return cls(
            version=1,
            llm=LLMConfig.from_dict(llm),
            mineru=MinerUConfig.from_dict(mineru),
            vector_index=VectorIndexConfig.from_dict(vector_index),
            insert_planning=InsertPlanningConfig.from_dict(insert_planning),
            dynamic_insert=DynamicInsertConfig.from_dict(dynamic_insert),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
