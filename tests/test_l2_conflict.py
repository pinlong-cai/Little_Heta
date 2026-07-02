from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.mem.l2_conflict import detect_conflicts_batch
from heta.providers.model_protocols import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    EmbeddingRequest,
    EmbeddingResult,
)


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    @property
    def model_name(self) -> str:
        return "embedding"

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.calls.append(request)
        vectors = [
            [float(i + 1)] + [0.0] * 1023
            for i, _ in enumerate(request.texts)
        ]
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)


class FakeChatModel:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    @property
    def model_name(self) -> str:
        return "chat"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.calls.append(request)
        return ChatCompletionResult(
            message=ChatMessage(role="assistant", content=self.content),
            model_name=self.model_name,
        )


def test_detect_conflicts_batch_skips_low_score_candidates(monkeypatch) -> None:
    embeddings = FakeEmbeddingModel()
    chat = FakeChatModel('{"deprecate": []}')

    def fake_search(conn, embedding, top_k, exclude_session_id):
        return [{"memory_id": "old-low", "fact_text": "old fact", "distance": 1.0}]

    monkeypatch.setattr("heta.mem.l2_conflict.search_similar_facts", fake_search)

    results = detect_conflicts_batch(
        conn=object(),
        new_fact_texts=["new fact"],
        chat_model=chat,
        embedding_model=embeddings,
        config=_config(),
        min_candidate_score=0.60,
    )

    assert len(results) == 1
    assert results[0].ids_to_deprecate == []
    assert len(embeddings.calls) == 1
    assert embeddings.calls[0].texts == ["new fact"]
    assert chat.calls == []


def test_detect_conflicts_batch_judges_multiple_facts_once(monkeypatch) -> None:
    embeddings = FakeEmbeddingModel()
    chat = FakeChatModel(
        '{"deprecate": [{"new_fact_index": 1, "memory_ids": ["old-2"]}]}'
    )

    def fake_search(conn, embedding, top_k, exclude_session_id):
        old_id = f"old-{int(embedding[0])}"
        return [{"memory_id": old_id, "fact_text": f"old fact {old_id}", "distance": 0.0}]

    monkeypatch.setattr("heta.mem.l2_conflict.search_similar_facts", fake_search)

    results = detect_conflicts_batch(
        conn=object(),
        new_fact_texts=["new fact 1", "new fact 2", "new fact 3"],
        chat_model=chat,
        embedding_model=embeddings,
        config=_config(),
        min_candidate_score=0.60,
    )

    assert [result.ids_to_deprecate for result in results] == [[], ["old-2"], []]
    assert len(embeddings.calls) == 1
    assert embeddings.calls[0].texts == ["new fact 1", "new fact 2", "new fact 3"]
    assert len(chat.calls) == 1
    user_msg = chat.calls[0].messages[1].content
    assert "New fact index: 0" in user_msg
    assert "New fact index: 1" in user_msg
    assert "New fact index: 2" in user_msg


def test_detect_conflicts_batch_marks_exact_duplicate_without_judge(monkeypatch) -> None:
    embeddings = FakeEmbeddingModel()
    chat = FakeChatModel('{"deprecate": [{"new_fact_index": 0, "memory_ids": ["old-1"]}]}')

    def fake_search(conn, embedding, top_k, exclude_session_id):
        return [{"memory_id": "old-1", "fact_text": "汪文武 就读于 中国科学院大学", "distance": 0.0}]

    monkeypatch.setattr("heta.mem.l2_conflict.search_similar_facts", fake_search)

    results = detect_conflicts_batch(
        conn=object(),
        new_fact_texts=["汪文武 就读于 中国科学院大学"],
        chat_model=chat,
        embedding_model=embeddings,
        config=_config(),
        min_candidate_score=0.60,
    )

    assert results[0].duplicate_of == "old-1"
    assert results[0].ids_to_deprecate == []
    assert chat.calls == []
