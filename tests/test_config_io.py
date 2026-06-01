from pathlib import Path

from heta.config.io import load_config, save_config
from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig


def test_save_and_load_config(tmp_path: Path) -> None:
    path = tmp_path / ".heta" / "heta.yaml"
    config = HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig.enabled(),
        insert_planning=InsertPlanningConfig.enabled(),
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
    assert path.exists()


def test_load_config_fills_default_llm_profile(tmp_path: Path) -> None:
    path = tmp_path / ".heta" / "heta.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
version: 1
llm:
  provider: qwen
  api_key: sk-test
mineru:
  enable: false
  provider:
  api_key:
  endpoint:
vector_index:
  enable: true
insert_planning:
  enable: true
""",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded is not None
    assert loaded.llm.chat_model == "qwen3.5-flash"
    assert loaded.llm.chat_api_key == "sk-test"
    assert loaded.llm.multimodal_model == "qwen3.5-omni-flash"
    assert loaded.llm.multimodal_api_key == "sk-test"
    assert loaded.llm.embedding_model == "text-embedding-v4"
    assert loaded.llm.embedding_api_key == "sk-test"


def test_load_config_accepts_custom_llm_profile(tmp_path: Path) -> None:
    path = tmp_path / ".heta" / "heta.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
version: 1
llm:
  provider: custom
  api_key: sk-test
  chat_api_key: sk-chat
  chat_model: custom-chat
  chat_base_url: http://llm.local/v1
  chat_extra_body:
    enable_thinking: false
  multimodal_api_key: sk-mm
  multimodal_model: custom-mm
  multimodal_base_url: http://mm.local/v1
  embedding_api_key: sk-embedding
  embedding_model: custom-embedding
  embedding_base_url: http://embedding.local/v1
mineru:
  enable: false
  provider:
  api_key:
  endpoint:
vector_index:
  enable: true
insert_planning:
  enable: true
""",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded is not None
    assert loaded.llm.provider == "custom"
    assert loaded.llm.chat_api_key == "sk-chat"
    assert loaded.llm.chat_model == "custom-chat"
    assert loaded.llm.chat_base_url == "http://llm.local/v1"
    assert loaded.llm.chat_extra_body == {"enable_thinking": False}
    assert loaded.llm.multimodal_api_key == "sk-mm"
    assert loaded.llm.embedding_api_key == "sk-embedding"
    assert loaded.llm.embedding_model == "custom-embedding"


def test_custom_config_requires_embedding_fields(tmp_path: Path) -> None:
    path = tmp_path / ".heta" / "heta.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
version: 1
llm:
  provider: custom
  api_key: sk-test
  chat_api_key: sk-chat
  chat_model: custom-chat
  chat_base_url: http://llm.local/v1
mineru:
  enable: false
  provider:
  api_key:
  endpoint:
vector_index:
  enable: true
insert_planning:
  enable: true
""",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert "embedding_model" in str(exc)
    else:
        raise AssertionError("missing custom embedding fields should fail")


def test_load_missing_config_returns_none(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.yaml") is None


def test_config_requires_insert_planning(tmp_path: Path) -> None:
    path = tmp_path / ".heta" / "heta.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
version: 1
llm:
  provider: qwen
  api_key: sk-test
mineru:
  enable: false
  provider:
  api_key:
  endpoint:
vector_index:
  enable: true
""",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert "insert_planning" in str(exc)
    else:
        raise AssertionError("missing insert_planning should fail")
