from types import SimpleNamespace

from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.mem.l1_dedup import detect_episode_duplicates_batch


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _embedding_response(count: int, dim: int = 1024):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[float(i + 1)] + [0.0] * (dim - 1)) for i in range(count)]
    )


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _embedding_response(len(kwargs["input"]), kwargs["dimensions"])


class FakeChatCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def test_detect_episode_duplicates_batch_skips_low_score_candidates(monkeypatch) -> None:
    embeddings = FakeEmbeddings()
    chat = FakeChatCompletions('{"duplicates": [{"new_episode_index": 0, "memory_id": "old-1"}]}')
    emb_client = SimpleNamespace(embeddings=embeddings)
    llm_client = SimpleNamespace(chat=SimpleNamespace(completions=chat))

    def fake_search(conn, embedding, top_k):
        return [{"memory_id": "old-low", "summary": "old episode", "score": 0.5}]

    monkeypatch.setattr("heta.mem.l1_dedup.search_episodes", fake_search)

    results = detect_episode_duplicates_batch(
        conn=object(),
        new_episode_summaries=["new episode"],
        llm_client=llm_client,
        llm_model="chat",
        emb_client=emb_client,
        emb_model="embedding",
        config=_config(),
        min_candidate_score=0.72,
    )

    assert results[0].duplicate_of is None
    assert len(embeddings.calls) == 1
    assert embeddings.calls[0]["input"] == ["new episode"]
    assert chat.calls == []


def test_detect_episode_duplicates_batch_judges_multiple_episodes_once(monkeypatch) -> None:
    embeddings = FakeEmbeddings()
    chat = FakeChatCompletions(
        '{"duplicates": [{"new_episode_index": 1, "memory_id": "old-2"}]}'
    )
    emb_client = SimpleNamespace(embeddings=embeddings)
    llm_client = SimpleNamespace(chat=SimpleNamespace(completions=chat))

    def fake_search(conn, embedding, top_k):
        old_id = f"old-{int(embedding[0])}"
        return [{"memory_id": old_id, "summary": f"old episode {old_id}", "score": 0.95}]

    monkeypatch.setattr("heta.mem.l1_dedup.search_episodes", fake_search)

    results = detect_episode_duplicates_batch(
        conn=object(),
        new_episode_summaries=["episode 1", "episode 2", "episode 3"],
        llm_client=llm_client,
        llm_model="chat",
        emb_client=emb_client,
        emb_model="embedding",
        config=_config(),
        min_candidate_score=0.72,
    )

    assert [result.duplicate_of for result in results] == [None, "old-2", None]
    assert len(embeddings.calls) == 1
    assert embeddings.calls[0]["input"] == ["episode 1", "episode 2", "episode 3"]
    assert len(chat.calls) == 1
    user_msg = chat.calls[0]["messages"][1]["content"]
    assert "New episode index: 0" in user_msg
    assert "New episode index: 1" in user_msg
    assert "New episode index: 2" in user_msg
