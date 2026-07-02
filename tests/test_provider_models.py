import sys

from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.providers.clients import build_chat_model, build_embedding_model, build_multimodal_model
from heta.providers.litellm_models import LiteLLMChatModel, LiteLLMEmbeddingModel
from heta.providers.model_protocols import (
    ChatCompletionRequest,
    ChatMessage,
    ChatModelConfig,
    ChatModelOptions,
    ChatToolCall,
    ChatToolCallFunction,
    EmbeddingModelConfig,
    EmbeddingModelOptions,
    EmbeddingRequest,
)


def _heta_config(llm: LLMConfig) -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=llm,
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def test_build_models_normalizes_litellm_model_names() -> None:
    qwen = _heta_config(LLMConfig(provider="qwen", api_key="sk-qwen"))
    assert build_chat_model(qwen).model_name == "openai/qwen3.5-flash"
    assert build_multimodal_model(qwen).model_name == "openai/qwen3.5-omni-flash"
    assert build_embedding_model(qwen).model_name == "openai/text-embedding-v4"
    assert build_chat_model(qwen).provider_options == {"enable_thinking": False}

    chatgpt = _heta_config(LLMConfig(provider="chatgpt", api_key="sk-openai"))
    assert build_chat_model(chatgpt).model_name == "openai/gpt-5.4-nano"
    assert build_embedding_model(chatgpt).model_name == "openai/text-embedding-3-small"

    gemini = _heta_config(LLMConfig(provider="gemini", api_key="gemini-key"))
    gemini_chat = build_chat_model(gemini)
    assert gemini_chat.model_name == "gemini/gemini-2.5-flash"
    assert gemini_chat.api_base is None
    assert build_embedding_model(gemini).model_name == "gemini/text-embedding-004"


def test_custom_models_allow_explicit_litellm_provider_without_api_base() -> None:
    config = _heta_config(
        LLMConfig(
            provider="custom",
            api_key="legacy-key",
            chat_api_key="anthropic-key",
            chat_model="anthropic/claude-sonnet-4-5",
            embedding_api_key="openai-key",
            embedding_model="openai/text-embedding-3-small",
        )
    )

    assert build_chat_model(config).model_name == "anthropic/claude-sonnet-4-5"
    assert build_chat_model(config).api_base is None
    assert build_embedding_model(config).model_name == "openai/text-embedding-3-small"


def test_custom_openai_compatible_models_default_to_openai_prefix() -> None:
    config = _heta_config(
        LLMConfig(
            provider="custom",
            api_key="legacy-key",
            chat_api_key="chat-key",
            chat_model="local-chat",
            chat_base_url="http://chat.local/v1",
            embedding_api_key="embedding-key",
            embedding_model="local-embedding",
            embedding_base_url="http://embedding.local/v1",
        )
    )

    chat_model = build_chat_model(config)
    embedding_model = build_embedding_model(config)

    assert chat_model.model_name == "openai/local-chat"
    assert chat_model.api_base == "http://chat.local/v1"
    assert embedding_model.model_name == "openai/local-embedding"
    assert embedding_model.api_base == "http://embedding.local/v1"


def test_provider_embedding_dimension_payload_matches_litellm_compatibility(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeLiteLLM:
        @staticmethod
        def embedding(**kwargs):
            calls.append(kwargs)
            return {
                "data": [{"index": 0, "embedding": [0.0] * 1024}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM)

    qwen = _heta_config(LLMConfig(provider="qwen", api_key="sk-qwen"))
    build_embedding_model(qwen).embed(EmbeddingRequest(texts=["hello"]))
    assert calls[0]["model"] == "openai/text-embedding-v4"
    assert "dimensions" not in calls[0]

    calls.clear()
    chatgpt = _heta_config(LLMConfig(provider="chatgpt", api_key="sk-openai"))
    build_embedding_model(chatgpt).embed(EmbeddingRequest(texts=["hello"]))
    assert calls[0]["model"] == "openai/text-embedding-3-small"
    assert calls[0]["dimensions"] == 1024


def test_litellm_chat_model_sends_messages_tools_and_parses_tool_calls(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeLiteLLM:
        @staticmethod
        def completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_page",
                                        "arguments": '{"path":"index.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM)
    model = LiteLLMChatModel(
        ChatModelConfig(
            model_name="openai/test-chat",
            api_key="test-key",
            api_base="http://llm.local/v1",
            request_timeout=30,
            max_retries=2,
            provider_options={"default_flag": True},
        )
    )

    result = model.complete(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are Little Heta."),
                ChatMessage(role="user", content="Read the index."),
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            id="call_prev",
                            function=ChatToolCallFunction(name="read_page", arguments="{}"),
                        )
                    ],
                ),
                ChatMessage(role="tool", tool_call_id="call_prev", content="# Wiki Index"),
            ],
            tools=[{"type": "function", "function": {"name": "read_page"}}],
            tool_choice="auto",
            options=ChatModelOptions(
                temperature=0.2,
                max_output_tokens=128,
                provider_options={"request_flag": "yes"},
            ),
        )
    )

    assert calls[0]["model"] == "openai/test-chat"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "http://llm.local/v1"
    assert calls[0]["timeout"] == 30
    assert calls[0]["num_retries"] == 2
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["max_tokens"] == 128
    assert calls[0]["tools"] == [{"type": "function", "function": {"name": "read_page"}}]
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["default_flag"] is True
    assert calls[0]["request_flag"] == "yes"
    assert calls[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "read_page"
    assert calls[0]["messages"][3]["tool_call_id"] == "call_prev"

    assert result.message.tool_calls[0].id == "call_1"
    assert result.message.tool_calls[0].function.name == "read_page"
    assert result.message.tool_calls[0].function.arguments == '{"path":"index.md"}'
    assert result.usage is not None
    assert result.usage.total_tokens == 14
    assert result.finish_reason == "tool_calls"


def test_litellm_embedding_model_preserves_input_order_and_options(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeLiteLLM:
        @staticmethod
        def embedding(**kwargs):
            calls.append(kwargs)
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4, 0.5]},
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                ],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM)
    model = LiteLLMEmbeddingModel(
        EmbeddingModelConfig(
            model_name="openai/test-embedding",
            api_key="embedding-key",
            api_base="http://embedding.local/v1",
            dimensions=1024,
            provider_options={"user": "default"},
        )
    )

    result = model.embed(
        EmbeddingRequest(
            texts=["first", "second"],
            options=EmbeddingModelOptions(
                dimensions=3,
                provider_options={"user": "request"},
            ),
        )
    )

    assert calls[0]["model"] == "openai/test-embedding"
    assert calls[0]["input"] == ["first", "second"]
    assert calls[0]["api_key"] == "embedding-key"
    assert calls[0]["api_base"] == "http://embedding.local/v1"
    assert calls[0]["dimensions"] == 3
    assert calls[0]["user"] == "request"
    assert result.vectors == [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]
    assert result.usage is not None
    assert result.usage.total_tokens == 3
