"""Tests for the remember CLI output."""

from __future__ import annotations

from typer.testing import CliRunner

from heta.cli import app
from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.mem.pipeline import RememberResult


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _result() -> RememberResult:
    return RememberResult(
        session_id="session-1",
        l0_count=1,
        l1_count=1,
        l2_count=2,
        elapsed_s=1.23,
        timings={"extract": 0.5, "dedup_conflict": 0.3, "persist_l1": 0.01},
    )


def test_remember_cli_hides_timing_by_default(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.remember.load_config", lambda: _config())
    monkeypatch.setattr("heta.cli.remember.remember", lambda text, config, mode="high": _result())

    result = CliRunner().invoke(app, ["remember", "save this"])

    assert result.exit_code == 0
    assert "L0 turns:" in result.output
    assert "elapsed: 1.23s" in result.output
    assert "timing:" not in result.output
    assert "dedup_conflict" not in result.output


def test_remember_cli_shows_timing_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.remember.load_config", lambda: _config())
    monkeypatch.setattr("heta.cli.remember.remember", lambda text, config, mode="high": _result())

    result = CliRunner().invoke(
        app,
        ["remember", "save this"],
        env={"HETA_REMEMBER_TIMING": "1"},
    )

    assert result.exit_code == 0
    assert "timing:" in result.output
    assert "extract: 0.500s" in result.output
    assert "dedup_conflict: 0.300s" in result.output


def test_remember_cli_fast_passes_fast_mode(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("heta.cli.remember.load_config", lambda: _config())

    def fake_remember(text, config, mode="high"):
        captured["text"] = text
        captured["mode"] = mode
        return _result()

    monkeypatch.setattr("heta.cli.remember.remember", fake_remember)

    result = CliRunner().invoke(app, ["remember", "--fast", "save this"])

    assert result.exit_code == 0
    assert captured == {"text": "save this", "mode": "fast"}
    assert "mode: fast" in result.output


def test_remember_cli_rejects_invalid_mode(monkeypatch) -> None:
    monkeypatch.setattr("heta.cli.remember.load_config", lambda: _config())

    result = CliRunner().invoke(app, ["remember", "--mode", "medium", "save this"])

    assert result.exit_code == 1
    assert "Invalid remember mode" in result.output
