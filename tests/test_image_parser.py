from pathlib import Path

from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.kb.image_parser import build_image_markdown
from heta.kb.parser import parse_document
from heta.kb.text import extract_title


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def test_build_image_markdown_uses_compact_retrieval_sections() -> None:
    markdown = build_image_markdown(
        title="Image - Architecture Diagram",
        source_name="diagram.png",
        image_path="../../raw/diagram.png",
        summary="A system architecture diagram.",
        visual_facts="Scene/type: diagram. Main subject: service pipeline.",
        visible_text="API Gateway",
        interpretation_keywords="Represents a backend data flow. keywords: API, pipeline.",
    )

    assert extract_title(markdown, "fallback") == "Image - Architecture Diagram"
    assert "![diagram.png](<../../raw/diagram.png>)" in markdown
    assert "### Visual Facts" in markdown
    assert "### Visible Text" in markdown
    assert "### Interpretation and Keywords" in markdown
    assert "## Related Pages" in markdown
    assert "## Source" in markdown


def test_parse_document_accepts_image_branch(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "diagram.png"
    archived = tmp_path / "raw_diagram.png"
    source.write_bytes(b"png")
    archived.write_bytes(b"png")

    monkeypatch.setattr(
        "heta.kb.parser.parse_image_markdown",
        lambda source_path, archived_path, config: build_image_markdown(
            title="Image - Diagram",
            source_name=archived_path.name,
            image_path="../../raw/raw_diagram.png",
            summary="A diagram.",
            visual_facts="A simple diagram.",
            visible_text="None detected.",
            interpretation_keywords="diagram, test",
        ),
    )

    document = parse_document(source, archived, _config())

    assert document.title == "Image - Diagram"
    assert document.source_name == "raw_diagram.png"
    assert document.metadata["extension"] == ".png"
    assert "### Visual Facts" in document.markdown_content
