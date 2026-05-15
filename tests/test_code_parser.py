from pathlib import Path

from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.kb.code_parser import extract_code_symbols, parse_code_markdown
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


def test_parse_code_markdown_keeps_small_code_inline(tmp_path: Path) -> None:
    source = tmp_path / "vector_index.py"
    archived = tmp_path / "2026-05-15_vector_index.py"
    source.write_text(
        'def search_wiki_vector_index(query, config):\n    """Search semantic wiki chunks."""\n    return []\n',
        encoding="utf-8",
    )
    archived.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    markdown = parse_code_markdown(source, archived)

    assert extract_title(markdown, "fallback") == "Code - vector_index.py"
    assert "[Raw source](<../../raw/2026-05-15_vector_index.py>)" in markdown
    assert "### Code" in markdown
    assert "def search_wiki_vector_index" in markdown


def test_parse_code_markdown_uses_symbol_index_for_large_code(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    archived = tmp_path / "2026-05-15_service.py"
    body = "\n".join(["class MemoryService:", "    \"\"\"Coordinates memory writes.\"\"\"", "    pass", *["x = 1"] * 220])
    source.write_text(body, encoding="utf-8")
    archived.write_text(body, encoding="utf-8")

    markdown = parse_code_markdown(source, archived)

    assert "### Symbol Index" in markdown
    assert "#### MemoryService" in markdown
    assert "Lines: 1-" in markdown
    assert "Coordinates memory writes." in markdown
    assert "### Code" not in markdown


def test_extract_code_symbols_handles_config_and_sql() -> None:
    yaml_symbols = extract_code_symbols(Path("heta.yaml"), "vector_index:\n  enable: true\nllm:\n  provider: qwen\n")
    sql_symbols = extract_code_symbols(Path("schema.sql"), "CREATE TABLE wiki_chunks (id integer);\nSELECT * FROM wiki_chunks;\n")

    assert [symbol.name for symbol in yaml_symbols] == ["vector_index", "llm"]
    assert sql_symbols[0].name == "CREATE TABLE wiki_chunks"


def test_extract_code_symbols_keeps_regex_language_signatures() -> None:
    cases = [
        (Path("sample.go"), "type QueryService struct{}\n\nfunc SearchWiki(query string) string {\n    return query\n}\n", "SearchWiki", "func SearchWiki"),
        (Path("sample.rs"), "pub struct QueryService;\n\npub fn search_wiki(query: &str) -> &str {\n    query\n}\n", "search_wiki", "pub fn search_wiki"),
        (Path("sample.js"), "export function runQuery(query) {\n  return query;\n}\n", "runQuery", "export function runQuery"),
        (Path("sample.ts"), "export function formatAnswer(result: QueryResult) {\n  return result.answer;\n}\n", "formatAnswer", "export function formatAnswer"),
    ]

    for path, text, name, signature_prefix in cases:
        symbols = extract_code_symbols(path, text)
        symbol = next(symbol for symbol in symbols if symbol.name == name)
        assert symbol.signature.startswith(signature_prefix)


def test_parse_document_accepts_code_branch(tmp_path: Path) -> None:
    source = tmp_path / "tool.ts"
    archived = tmp_path / "2026-05-15_tool.ts"
    source.write_text("export function runTool() { return true; }\n", encoding="utf-8")
    archived.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    document = parse_document(source, archived, _config())

    assert document.title == "Code - tool.ts"
    assert document.metadata["extension"] == ".ts"
    assert "language: typescript" in document.markdown_content
