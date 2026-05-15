from pathlib import Path

import pytest

from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.kb import paths
from heta.kb.text import frontmatter_page
from heta.query.agent import _parse_final_answer, _vector_match_map
from heta.query.models import QueryResult, QuerySource, VectorMatch
from heta.query.pipeline import run_wiki_query
from heta.query.tools import format_vector_matches, read_page, read_raw, source_from_page_path


def _config(vector_enabled: bool = False) -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=vector_enabled),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def test_run_wiki_query_requires_existing_wiki(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_wiki_query("What is HetaGen?", _config(), base_dir=tmp_path)


def test_run_wiki_query_delegates_to_read_only_agent(monkeypatch, tmp_path: Path) -> None:
    paths.index_path(tmp_path).parent.mkdir(parents=True)
    paths.index_path(tmp_path).write_text("# Wiki Index\n", encoding="utf-8")

    def fake_agent(**kwargs):
        assert kwargs["question"] == "What is HetaGen?"
        assert kwargs["base_dir"] == tmp_path
        return QueryResult(answer="HetaGen is a wiki page.", sources=[QuerySource(1, "HetaGen", "pages/1-hetagen.md")])

    monkeypatch.setattr("heta.query.pipeline.run_query_agent", fake_agent)

    result = run_wiki_query("What is HetaGen?", _config(), base_dir=tmp_path)

    assert result.answer == "HetaGen is a wiki page."
    assert result.sources[0].wiki_id == 1


def test_read_page_is_limited_to_pages(tmp_path: Path) -> None:
    page = paths.pages_dir(tmp_path) / "1-hetagen.md"
    page.parent.mkdir(parents=True)
    page.write_text("hello", encoding="utf-8")

    assert read_page("pages/1-hetagen.md", tmp_path) == "hello"
    assert read_page("../heta.yaml", tmp_path).startswith("error:")
    assert read_page("index.md", tmp_path).startswith("error:")


def test_read_raw_is_limited_to_raw_directory(tmp_path: Path) -> None:
    raw = paths.raw_dir(tmp_path)
    raw.mkdir(parents=True)
    (raw / "module.py").write_text("def run():\n    return True\n", encoding="utf-8")

    assert "def run" in read_raw("raw/module.py", tmp_path)
    assert "def run" in read_raw("../../raw/module.py", tmp_path)
    assert read_raw("../heta.yaml", tmp_path).startswith("error:")


def test_source_from_page_path_reads_frontmatter_and_wiki_id(tmp_path: Path) -> None:
    page = paths.pages_dir(tmp_path) / "12-hetagen.md"
    page.parent.mkdir(parents=True)
    page.write_text(frontmatter_page("HetaGen", "source.md", "Summary.", "Body."), encoding="utf-8")

    source = source_from_page_path("pages/12-hetagen.md", tmp_path, heading_path="Content")

    assert source == QuerySource(12, "HetaGen", "pages/12-hetagen.md", "Content")


def test_query_sources_include_validated_vector_chunks_only(tmp_path: Path) -> None:
    pages = paths.pages_dir(tmp_path)
    pages.mkdir(parents=True)
    (pages / "8-image.md").write_text(frontmatter_page("Image", "image.png", "Image summary.", "Body."), encoding="utf-8")
    (pages / "10-audio.md").write_text(
        frontmatter_page("Audio", "audio.mp3", "Audio summary.", "Transcript."),
        encoding="utf-8",
    )
    vector_matches = _vector_match_map(
        [
            VectorMatch(8, "8-image.md", "pages/8-image.md", "8:abc", "Content > Visible Text", "image text", 0.8),
            VectorMatch(10, "10-audio.md", "pages/10-audio.md", "10:def", "Content > Transcript", "hello", 0.9),
        ]
    )

    final = _parse_final_answer(
        text=(
            '{"answer": "The audio says hello.", "used_sources": ['
            '{"path": "pages/10-audio.md", "heading_path": "Content > Transcript"},'
            '{"path": "pages/8-image.md", "heading_path": "Content > Missing"}'
            "]}"
        ),
        read_paths=set(),
        vector_matches=vector_matches,
        base_dir=tmp_path,
    )

    assert final.answer == "The audio says hello."
    assert final.sources == [QuerySource(10, "Audio", "pages/10-audio.md", "Content > Transcript")]


def test_query_sources_accept_read_pages_without_vector_heading(tmp_path: Path) -> None:
    pages = paths.pages_dir(tmp_path)
    pages.mkdir(parents=True)
    (pages / "10-audio.md").write_text(
        frontmatter_page("Audio", "audio.mp3", "Audio summary.", "Transcript."),
        encoding="utf-8",
    )

    final = _parse_final_answer(
        text='{"answer": "From the full page.", "used_sources": [{"path": "pages/10-audio.md"}]}',
        read_paths={"pages/10-audio.md"},
        vector_matches={},
        base_dir=tmp_path,
    )

    assert final.sources == [QuerySource(10, "Audio", "pages/10-audio.md")]


def test_query_sources_reject_raw_used_sources(tmp_path: Path) -> None:
    final = _parse_final_answer(
        text='{"answer": "Raw helped.", "used_sources": [{"path": "raw/module.py"}]}',
        read_paths=set(),
        vector_matches={},
        base_dir=tmp_path,
    )

    assert final.sources == []


def test_format_vector_matches_includes_chunk_identity() -> None:
    text = format_vector_matches(
        [
            VectorMatch(
                wiki_id=1,
                page_name="1-hetagen.md",
                path="pages/1-hetagen.md",
                chunk_id="1:abc",
                heading_path="Content > API",
                content="API details",
                score=0.75,
            )
        ]
    )

    assert "wiki_id: 1" in text
    assert "chunk_id: 1:abc" in text
    assert "heading: Content > API" in text
