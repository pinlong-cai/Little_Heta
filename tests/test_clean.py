from pathlib import Path

from heta.kb import paths
from heta.kb.clean import clean_knowledge_base
from heta.kb.store import commit_wiki, ensure_wiki_layout
from heta.kb.text import frontmatter_page


def test_clean_knowledge_base_clears_pages_index_and_vector_db_but_keeps_raw(tmp_path: Path) -> None:
    ensure_wiki_layout(tmp_path)
    raw = paths.raw_dir(tmp_path) / "source.md"
    raw.parent.mkdir(parents=True)
    raw.write_text("# Source", encoding="utf-8")
    page = paths.pages_dir(tmp_path) / "1-hotpotqa.md"
    page.write_text(frontmatter_page("HotpotQA", "source.md", "Summary.", "Content."), encoding="utf-8")
    paths.index_path(tmp_path).write_text(
        "# Wiki Index\n\n- [1] [[HotpotQA]] (pages/1-hotpotqa.md) — Summary.\n",
        encoding="utf-8",
    )
    db = paths.vector_db_path(tmp_path)
    db.parent.mkdir(parents=True)
    db.write_bytes(b"vector db")
    commit_wiki("ingest: source.md", tmp_path)

    summary = clean_knowledge_base(base_dir=tmp_path)

    assert summary.deleted_pages == 1
    assert summary.deleted_vector_files == 1
    assert summary.commit_id
    assert raw.exists()
    assert not page.exists()
    assert not db.exists()
    assert paths.index_path(tmp_path).read_text(encoding="utf-8") == "# Wiki Index\n\n"
    assert "Cleaned Little Heta knowledge base." in paths.log_path(tmp_path).read_text(encoding="utf-8")


def test_clean_knowledge_base_is_idempotent(tmp_path: Path) -> None:
    ensure_wiki_layout(tmp_path)
    first = clean_knowledge_base(base_dir=tmp_path)
    second = clean_knowledge_base(base_dir=tmp_path)

    assert first.deleted_pages == 0
    assert first.deleted_vector_files == 0
    assert second.deleted_pages == 0
    assert second.deleted_vector_files == 0
