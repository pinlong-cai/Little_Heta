"""Tests for ranked evidence selection in recall."""

from __future__ import annotations

from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.mem.recall import LayerEvidence, _rank, _select_ranked_evidence
from heta.providers.model_protocols import ChatCompletionRequest, ChatCompletionResult, ChatMessage


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig.enabled(),
        insert_planning=InsertPlanningConfig.enabled(),
    )


class FakeChatModel:
    def __init__(self):
        self.calls = []

    @property
    def model_name(self) -> str:
        return "mock-chat"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.calls.append(request)
        if len(self.calls) == 1:
            content = '{"ranking": ["atomic_fact", "episode", "raw", "kb_insight"], "reason": "facts first"}'
        else:
            content = '{"answer": "张明在北京智谱工作。", "sufficient": true}'
        return ChatCompletionResult(
            message=ChatMessage(role="assistant", content=content),
            model_name=self.model_name,
        )


def _evidence() -> list[LayerEvidence]:
    return [
        LayerEvidence(layer="raw", items=[{"score": 0.9, "text_content": "raw text should not answer"}]),
        LayerEvidence(layer="episode", items=[{"score": 0.8, "summary": "张明换工作到了北京"}]),
        LayerEvidence(layer="atomic_fact", items=[{"score": 0.95, "fact_text": "张明 就职于 北京智谱"}]),
        LayerEvidence(layer="kb_insight", items=[]),
    ]


def test_select_ranked_evidence_uses_top_two_non_empty_layers() -> None:
    selected = _select_ranked_evidence(
        _evidence(),
        ["kb_insight", "atomic_fact", "missing", "episode", "raw"],
        max_layers=2,
    )

    assert [layer.layer for layer in selected] == ["atomic_fact", "episode"]


def test_select_ranked_evidence_falls_back_when_ranking_has_no_useful_layers() -> None:
    evidence = _evidence()

    selected = _select_ranked_evidence(evidence, ["kb_insight", "missing"], max_layers=2)

    assert selected is evidence


def test_rank_generates_answer_from_top_two_ranked_layers_only() -> None:
    chat = FakeChatModel()

    ranking, answer, reason, sufficient = _rank(
        query="张明在哪里工作？",
        evidence=_evidence(),
        chat_model=chat,
    )

    assert ranking == ["atomic_fact", "episode", "raw", "kb_insight"]
    assert reason == "facts first"
    assert answer == "张明在北京智谱工作。"
    assert sufficient is True
    assert len(chat.calls) == 2

    answer_user_msg = chat.calls[1].messages[1].content
    assert "## atomic_fact" in answer_user_msg
    assert "张明 就职于 北京智谱" in answer_user_msg
    assert "## episode" in answer_user_msg
    assert "张明换工作到了北京" in answer_user_msg
    assert "## raw" not in answer_user_msg
    assert "raw text should not answer" not in answer_user_msg
