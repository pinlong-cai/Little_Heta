from pathlib import Path

import pytest

from heta.config.schema import HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.kb.discovery import collect_insert_files
from heta.kb.models import FileChange
from heta.kb.insert import insert_paths
from heta.kb.text import frontmatter_page, slugify, summarize
from heta.kb.wiki import normalize_wiki_pages, repair_broken_wiki_links


def _config(mineru: MinerUConfig | None = None) -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=mineru or MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
    )


def _fake_agent(monkeypatch, calls: list[list[str]] | None = None) -> None:
    def run_merge_agent(*, task_id, documents, root_dir, config):
        assert len(documents) == 1
        if calls is not None:
            calls.append([document.source_name for document in documents])
        pages = root_dir / "pages"
        pages.mkdir(parents=True, exist_ok=True)
        added = []
        updated = []
        for document in documents:
            slug = slugify(document.title)
            page = next(
                (
                    candidate
                    for candidate in pages.glob("*.md")
                    if candidate.name.endswith(f"-{slug}.md") or candidate.name == f"{slug}.md"
                ),
                pages / f"{slug}.md",
            )
            rel = f"pages/{page.name}"
            content = frontmatter_page(
                document.title,
                document.source_name,
                summarize(document.markdown_content),
                document.markdown_content,
            )
            if page.exists():
                page.write_text(page.read_text(encoding="utf-8") + "\n## Imported Update\n", encoding="utf-8")
                updated.append(FileChange("updated", document.title, rel))
            else:
                page.write_text(content, encoding="utf-8")
                added.append(FileChange("added", document.title, rel))
            index = root_dir / "index.md"
            index.write_text(
                index.read_text(encoding="utf-8").rstrip()
                + f"\n\n## Imported Knowledge\n\n- [[{document.title}]] ({rel})\n  - {summarize(document.markdown_content)}\n",
                encoding="utf-8",
            )
            with (root_dir / "log.md").open("a", encoding="utf-8") as log:
                log.write(f"- Created page: {document.title} from {document.source_name}\n")
        return {"added": added, "updated": updated, "deleted": []}

    monkeypatch.setattr("heta.kb.insert.run_merge_agent", run_merge_agent)


def test_insert_markdown_creates_versioned_wiki(monkeypatch, tmp_path: Path) -> None:
    _fake_agent(monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("# Neural Network Basics\n\nLayers and activations.", encoding="utf-8")

    result = insert_paths([source], _config(), base_dir=tmp_path)

    wiki = tmp_path / "workspace" / "kb" / "wiki"
    raw = tmp_path / "workspace" / "kb" / "raw"
    page = wiki / "pages" / "1-neural-network-basics.md"

    assert result.commit_id
    assert page.exists()
    assert (wiki / ".git").exists()
    assert "[[Neural Network Basics]]" in (wiki / "index.md").read_text(encoding="utf-8")
    assert "Created page: Neural Network Basics" in (wiki / "log.md").read_text(encoding="utf-8")
    assert list(raw.glob("*notes.md"))


def test_insert_same_title_updates_existing_page(monkeypatch, tmp_path: Path) -> None:
    _fake_agent(monkeypatch)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# Shared Topic\n\nFirst version.", encoding="utf-8")
    second.write_text("# Shared Topic\n\nSecond version.", encoding="utf-8")

    insert_paths([first], _config(), base_dir=tmp_path)
    result = insert_paths([second], _config(), base_dir=tmp_path)

    page = tmp_path / "workspace" / "kb" / "wiki" / "pages" / "1-shared-topic.md"
    assert result.updated[0].title == "Shared Topic"
    assert result.updated[0].path == "pages/1-shared-topic.md"
    assert "## Imported Update" in page.read_text(encoding="utf-8")


def test_insert_multiple_files_runs_agent_sequentially(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    _fake_agent(monkeypatch, calls)
    first = tmp_path / "alpha.md"
    second = tmp_path / "beta.md"
    first.write_text("# Alpha\n\nFirst details.", encoding="utf-8")
    second.write_text("# Beta\n\nSecond details.", encoding="utf-8")

    result = insert_paths([first, second], _config(), base_dir=tmp_path)

    wiki = tmp_path / "workspace" / "kb" / "wiki"
    assert calls[0][0].endswith("_alpha.md")
    assert calls[1][0].endswith("_beta.md")
    assert (wiki / "pages" / "1-alpha.md").exists()
    assert (wiki / "pages" / "2-beta.md").exists()
    assert [change.path for change in result.added] == ["pages/1-alpha.md", "pages/2-beta.md"]


def test_pdf_requires_mineru_when_disabled(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF")

    with pytest.raises(ValueError, match="requires MinerU"):
        collect_insert_files([source], _config())


def test_collect_directory_skips_workspace(tmp_path: Path) -> None:
    source = tmp_path / "a.md"
    workspace_file = tmp_path / "workspace" / "kb" / "wiki" / "pages" / "old.md"
    source.write_text("# A", encoding="utf-8")
    workspace_file.parent.mkdir(parents=True)
    workspace_file.write_text("# Old", encoding="utf-8")

    files = collect_insert_files([tmp_path], _config())

    assert files == [source]


def test_normalize_wiki_pages_assigns_max_plus_one_without_reusing_deleted_ids(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    pages = wiki / "pages"
    pages.mkdir(parents=True)
    (wiki / "index.md").write_text("# Wiki Index\n\n", encoding="utf-8")
    (pages / "1-old.md").write_text(
        frontmatter_page("Old", "old.md", "Old summary.", "Old content."),
        encoding="utf-8",
    )
    (pages / "3-existing.md").write_text(
        frontmatter_page("Existing", "existing.md", "Existing summary.", "Existing content."),
        encoding="utf-8",
    )
    (pages / "new-topic.md").write_text(
        frontmatter_page("New Topic", "new.md", "New summary.", "New content."),
        encoding="utf-8",
    )

    result = normalize_wiki_pages(wiki)

    assert result.path_map == {"pages/new-topic.md": "pages/4-new-topic.md"}
    assert (pages / "4-new-topic.md").exists()
    index = (wiki / "index.md").read_text(encoding="utf-8")
    assert "- [1] [[Old]] (pages/1-old.md) — Old summary." in index
    assert "- [3] [[Existing]] (pages/3-existing.md) — Existing summary." in index
    assert "- [4] [[New Topic]] (pages/4-new-topic.md) — New summary." in index


def test_repair_broken_wiki_links_downgrades_missing_targets(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    pages = wiki / "pages"
    pages.mkdir(parents=True)
    page = pages / "1-topic.md"
    page.write_text(
        frontmatter_page(
            "Topic",
            "source.md",
            "Summary.",
            "See [[Existing Topic]] and [[Missing Topic]].",
        ),
        encoding="utf-8",
    )
    (pages / "2-existing-topic.md").write_text(
        frontmatter_page("Existing Topic", "source.md", "Summary.", "Body."),
        encoding="utf-8",
    )

    repair_broken_wiki_links(wiki)

    text = page.read_text(encoding="utf-8")
    assert "[[Existing Topic]]" in text
    assert "[[Missing Topic]]" not in text
    assert "Missing Topic" in text
