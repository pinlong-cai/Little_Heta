from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.providers import clients


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(
            provider="custom",
            api_key="legacy-key",
            chat_api_key="chat-key",
            chat_model="chat-model",
            chat_base_url="http://chat.local/v1",
            multimodal_api_key="mm-key",
            multimodal_model="mm-model",
            multimodal_base_url="http://mm.local/v1",
            embedding_api_key="embedding-key",
            embedding_model="embedding-model",
            embedding_base_url="http://embedding.local/v1",
        ),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def test_provider_clients_use_capability_specific_api_keys(monkeypatch) -> None:
    seen: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setattr(clients, "OpenAI", FakeOpenAI)
    config = _config()

    chat = clients.build_chat_client(config)
    multimodal = clients.build_multimodal_client(config)
    embedding = clients.build_embedding_client(config)

    assert chat.model == "chat-model"
    assert multimodal.model == "mm-model"
    assert embedding.model == "embedding-model"
    assert seen == [
        {"api_key": "chat-key", "timeout": 60, "base_url": "http://chat.local/v1"},
        {"api_key": "mm-key", "timeout": 300, "base_url": "http://mm.local/v1"},
        {"api_key": "embedding-key", "timeout": 120, "base_url": "http://embedding.local/v1"},
    ]


def test_extra_body_prefers_explicit_config() -> None:
    config = _config()
    explicit = HetaConfig(
        version=config.version,
        llm=LLMConfig(
            provider="custom",
            api_key="legacy-key",
            chat_api_key="chat-key",
            chat_model="chat-model",
            chat_base_url="http://chat.local/v1",
            chat_extra_body={"enable_thinking": False},
            embedding_api_key="embedding-key",
            embedding_model="embedding-model",
            embedding_base_url="http://embedding.local/v1",
        ),
        mineru=config.mineru,
        vector_index=config.vector_index,
        insert_planning=config.insert_planning,
    )

    assert clients.extra_body(explicit) == {"enable_thinking": False}


def test_extra_body_keeps_qwen_default() -> None:
    config = HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )

    assert clients.extra_body(config) == {"enable_thinking": False}
