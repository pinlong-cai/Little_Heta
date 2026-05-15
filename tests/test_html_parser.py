from pathlib import Path

from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.kb.html_parser import parse_html_markdown
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


def test_parse_html_markdown_preserves_structure_and_inline_images(tmp_path: Path) -> None:
    image = tmp_path / "arch.png"
    image.write_bytes(b"png")
    source = tmp_path / "attention.html"
    archived = tmp_path / "raw" / "2026-05-15_attention.html"
    archived.parent.mkdir()
    source.write_text(
        """
<!doctype html>
<html>
<head>
  <title>Attention Mechanism</title>
  <meta name="description" content="A short educational page about attention.">
  <style>body { color: red; }</style>
</head>
<body>
  <nav>Navigation noise</nav>
  <h1>Attention Mechanism</h1>
  <p><strong>Source:</strong> Bahdanau and Vaswani.</p>
  <h2>Overview</h2>
  <p>The model focuses on relevant input tokens.</p>
  <h2>Types</h2>
  <table><tr><th>Type</th><th>Description</th></tr><tr><td>Self-Attention</td><td>Tokens attend to tokens.</td></tr></table>
  <h2>Architecture</h2>
  <p>The Transformer uses multi-head attention.</p>
  <img src="arch.png" alt="Transformer architecture">
</body>
</html>
""",
        encoding="utf-8",
    )
    archived.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    markdown = parse_html_markdown(source, archived)

    assert extract_title(markdown, "fallback") == "Web Page - Attention Mechanism"
    assert "### Metadata" not in markdown
    assert "Raw HTML" not in markdown
    assert "Navigation noise" not in markdown
    assert "### Attention Mechanism" in markdown
    assert "#### Overview" in markdown
    assert "| Type | Description |" in markdown
    assert "![Transformer architecture](<../../raw/assets/2026-05-15_attention/img-001.png>)" in markdown
    assert "Image note: Transformer architecture." in markdown
    assert (tmp_path / "raw" / "assets" / "2026-05-15_attention" / "img-001.png").exists()
    manifest = (tmp_path / "raw" / "assets" / "2026-05-15_attention" / "manifest.json").read_text(encoding="utf-8")
    assert '"original_src": "arch.png"' in manifest
    assert '"section": "Architecture"' in manifest


def test_parse_html_markdown_keeps_remote_images_as_urls(tmp_path: Path) -> None:
    source = tmp_path / "remote.htm"
    archived = tmp_path / "raw" / "2026-05-15_remote.htm"
    archived.parent.mkdir()
    source.write_text(
        '<html><body><h1>Remote Page</h1><p>Intro.</p><img src="https://example.com/plot.png" alt="Remote plot"></body></html>',
        encoding="utf-8",
    )
    archived.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    markdown = parse_html_markdown(source, archived)

    assert "![Remote plot](<https://example.com/plot.png>)" in markdown
    manifest = (tmp_path / "raw" / "assets" / "2026-05-15_remote" / "manifest.json").read_text(encoding="utf-8")
    assert '"original_src": "https://example.com/plot.png"' in manifest
    assert '"raw_path": null' in manifest


def test_parse_document_accepts_html_branch(tmp_path: Path) -> None:
    source = tmp_path / "page.html"
    archived = tmp_path / "2026-05-15_page.html"
    source.write_text("<html><body><h1>HTML Page</h1><p>Hello.</p></body></html>", encoding="utf-8")
    archived.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    document = parse_document(source, archived, _config())

    assert document.title == "Web Page - HTML Page"
    assert document.metadata["extension"] == ".html"
    assert "### HTML Page" in document.markdown_content
