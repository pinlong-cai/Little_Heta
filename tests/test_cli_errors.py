from typer.testing import CliRunner

from heta.cli import app
from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _assert_clean_cli_error(output: str, title: str) -> None:
    assert title in output
    assert "insufficient_quota" in output
    assert "Reason:" in output
    assert "Traceback" not in output
    assert "File \"" not in output
    assert "smart_query(" not in output
    assert "remember(text" not in output
    assert "recall(query" not in output


def test_ask_api_error_is_user_facing(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.ask.load_config", lambda: _config())

    def fail(*args, **kwargs):
        raise RuntimeError("Error code: 429 - insufficient_quota")

    monkeypatch.setattr("heta.cli.ask.smart_query", fail)

    result = CliRunner().invoke(app, ["ask", "hello"])

    assert result.exit_code == 1
    _assert_clean_cli_error(result.output, "Ask failed.")


def test_remember_api_error_is_user_facing(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.remember.load_config", lambda: _config())

    def fail(*args, **kwargs):
        raise RuntimeError("Error code: 429 - insufficient_quota")

    monkeypatch.setattr("heta.cli.remember.remember", fail)

    result = CliRunner().invoke(app, ["remember", "save this"])

    assert result.exit_code == 1
    _assert_clean_cli_error(result.output, "Remember failed.")


def test_recall_api_error_is_user_facing(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.recall.load_config", lambda: _config())

    def fail(*args, **kwargs):
        raise RuntimeError("Error code: 429 - insufficient_quota")

    monkeypatch.setattr("heta.cli.recall.recall", fail)

    result = CliRunner().invoke(app, ["recall", "anything"])

    assert result.exit_code == 1
    _assert_clean_cli_error(result.output, "Recall failed.")


def test_query_api_error_is_user_facing(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.query.load_config", lambda: _config())

    def fail(*args, **kwargs):
        raise RuntimeError("Error code: 429 - insufficient_quota")

    monkeypatch.setattr("heta.cli.query.run_wiki_query", fail)

    result = CliRunner().invoke(app, ["query", "anything"])

    assert result.exit_code == 1
    _assert_clean_cli_error(result.output, "Query failed.")
