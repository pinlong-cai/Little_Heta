"""Static code-file parsing for Little Heta KB inserts."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".sh",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}

SMALL_CODE_LINE_LIMIT = 200

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c/cpp header",
    ".hpp": "cpp header",
    ".sh": "shell",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}


@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    signature: str
    start_line: int
    end_line: int
    summary: str


def parse_code_markdown(source_path: Path, archived_path: Path) -> str:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    suffix = source_path.suffix.lower()
    language = LANGUAGES.get(suffix, suffix.lstrip(".") or "text")
    lines = text.splitlines()
    symbols = extract_code_symbols(source_path, text)

    return build_code_markdown(
        title=f"Code - {source_path.name}",
        source_name=archived_path.name,
        raw_path=f"../../raw/{archived_path.name}",
        language=language,
        line_count=len(lines),
        symbols=symbols,
        code=text if len(lines) <= SMALL_CODE_LINE_LIMIT else None,
    )


def build_code_markdown(
    *,
    title: str,
    source_name: str,
    raw_path: str,
    language: str,
    line_count: int,
    symbols: list[CodeSymbol],
    code: str | None,
) -> str:
    summary = _summary(language, line_count, symbols)
    body = [
        "---",
        f"title: {title}",
        f"sources: [{source_name}]",
        f"updated: {date.today().isoformat()}",
        "---",
        "",
        "## Summary",
        summary,
        "",
        "## Content",
        "",
        f"[Raw source](<{raw_path}>)",
        "",
        "### File Overview",
        f"- language: {language}",
        f"- lines: {line_count}",
    ]
    if symbols:
        names = ", ".join(symbol.name for symbol in symbols[:20])
        suffix = "" if len(symbols) <= 20 else f", ... ({len(symbols)} total)"
        body.append(f"- symbols: {names}{suffix}")
    else:
        body.append("- symbols: none detected")

    if code is not None:
        body.extend(["", "### Code", f"```{_fence_language(language)}", code.rstrip(), "```"])
    else:
        body.extend(["", "### Symbol Index"])
        if symbols:
            for symbol in symbols:
                body.extend(
                    [
                        "",
                        f"#### {symbol.name}",
                        f"Lines: {symbol.start_line}-{symbol.end_line}",
                        f"Type: {symbol.kind}",
                    ]
                )
                if symbol.signature:
                    body.append(f"Signature: `{symbol.signature}`")
                body.append(f"Summary: {symbol.summary}")
        else:
            body.extend(["", "#### Lines 1-" + str(line_count), "Summary: Full source is available in raw."])

    body.extend(["", "## Source", f"- {source_name}", ""])
    return "\n".join(body)


def extract_code_symbols(path: Path, text: str) -> list[CodeSymbol]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _python_symbols(text)
    if suffix in {".yaml", ".yml", ".json", ".toml"}:
        return _config_symbols(suffix, text)
    if suffix == ".sql":
        return _sql_symbols(text)
    return _regex_symbols(suffix, text)


def _python_symbols(text: str) -> list[CodeSymbol]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    symbols: list[CodeSymbol] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind="class",
                    signature=f"class {node.name}",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    summary=_doc_summary(ast.get_docstring(node), f"Defines class `{node.name}`."),
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind=prefix,
                    signature=_python_signature(node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    summary=_doc_summary(ast.get_docstring(node), f"Defines {prefix} `{node.name}`."),
                )
            )
    return sorted(symbols, key=lambda symbol: (symbol.start_line, symbol.name))


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)})"


def _regex_symbols(suffix: str, text: str) -> list[CodeSymbol]:
    patterns = {
        ".js": [
            ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE)),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", re.MULTILINE)),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", re.MULTILINE)),
        ],
        ".ts": [],
        ".tsx": [],
        ".jsx": [],
        ".java": [
            ("class", re.compile(r"^\s*(?:public\s+)?(?:final\s+)?class\s+([A-Za-z_]\w*)", re.MULTILINE)),
            ("method", re.compile(r"^\s*(?:public|private|protected)\s+[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)),
        ],
        ".go": [
            ("function", re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(", re.MULTILINE)),
            ("type", re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+", re.MULTILINE)),
        ],
        ".rs": [
            ("function", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)),
            ("type", re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.MULTILINE)),
        ],
        ".cpp": [],
        ".c": [],
        ".h": [],
        ".hpp": [],
        ".sh": [
            ("function", re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{?", re.MULTILINE)),
        ],
    }
    patterns[".ts"] = patterns[".js"]
    patterns[".tsx"] = patterns[".js"]
    patterns[".jsx"] = patterns[".js"]
    c_like = [
        (
            "function",
            re.compile(
                r"^\s*(?:static\s+|inline\s+|extern\s+)?[\w:*&<>\[\]\s]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
                re.MULTILINE,
            ),
        )
    ]
    patterns[".cpp"] = c_like
    patterns[".c"] = c_like
    patterns[".h"] = c_like
    patterns[".hpp"] = c_like

    lines = text.splitlines()
    symbols: list[CodeSymbol] = []
    seen: set[tuple[str, int]] = set()
    for kind, pattern in patterns.get(suffix, []):
        for match in pattern.finditer(text):
            name = match.group(1)
            start_line = text[: match.start()].count("\n") + 1
            if (name, start_line) in seen:
                continue
            seen.add((name, start_line))
            signature = lines[start_line - 1].strip() if start_line - 1 < len(lines) else name
            symbols.append(
                CodeSymbol(
                    name=name,
                    kind=kind,
                    signature=signature.rstrip("{").strip(),
                    start_line=start_line,
                    end_line=_next_symbol_end(start_line, lines),
                    summary=f"Defines {kind} `{name}`.",
                )
            )
    return sorted(symbols, key=lambda symbol: (symbol.start_line, symbol.name))


def _config_symbols(suffix: str, text: str) -> list[CodeSymbol]:
    names: list[tuple[str, int]] = []
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            line_lookup = text.splitlines()
            for key in data:
                line = _find_key_line(str(key), line_lookup)
                names.append((str(key), line))
    else:
        pattern = re.compile(r"^([A-Za-z0-9_.-]+)\s*[:=]", re.MULTILINE)
        names = [(match.group(1), text[: match.start()].count("\n") + 1) for match in pattern.finditer(text)]

    lines = text.splitlines()
    symbols = [
        CodeSymbol(
            name=name,
            kind="config block",
            signature=name,
            start_line=line,
            end_line=_next_symbol_end(line, lines),
            summary=f"Configuration block `{name}`.",
        )
        for name, line in names
    ]
    return sorted(symbols, key=lambda symbol: (symbol.start_line, symbol.name))


def _sql_symbols(text: str) -> list[CodeSymbol]:
    pattern = re.compile(
        r"^\s*(CREATE\s+(?:TABLE|VIRTUAL\s+TABLE|INDEX|VIEW)|SELECT|INSERT|UPDATE|DELETE)\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w.]*)?",
        re.IGNORECASE | re.MULTILINE,
    )
    lines = text.splitlines()
    symbols: list[CodeSymbol] = []
    for index, match in enumerate(pattern.finditer(text), start=1):
        start_line = text[: match.start()].count("\n") + 1
        op = " ".join(match.group(1).upper().split())
        target = match.group(2) or f"statement-{index}"
        name = f"{op} {target}"
        symbols.append(
            CodeSymbol(
                name=name,
                kind="sql statement",
                signature=lines[start_line - 1].strip() if start_line - 1 < len(lines) else name,
                start_line=start_line,
                end_line=_next_symbol_end(start_line, lines),
                summary=f"SQL statement `{name}`.",
            )
        )
    return symbols


def _next_symbol_end(start_line: int, lines: list[str]) -> int:
    return min(len(lines), start_line + 80)


def _find_key_line(key: str, lines: list[str]) -> int:
    quoted = re.compile(rf'^\s*"{re.escape(key)}"\s*:')
    plain = re.compile(rf"^\s*{re.escape(key)}\s*[:=]")
    for index, line in enumerate(lines, start=1):
        if quoted.search(line) or plain.search(line):
            return index
    return 1


def _doc_summary(docstring: str | None, fallback: str) -> str:
    if not docstring:
        return fallback
    first = " ".join(docstring.strip().splitlines()[0].split())
    return first.rstrip(".") + "."


def _summary(language: str, line_count: int, symbols: list[CodeSymbol]) -> str:
    if symbols:
        names = ", ".join(symbol.name for symbol in symbols[:8])
        suffix = "" if len(symbols) <= 8 else f", and {len(symbols) - 8} more"
        return f"{language} source file with {line_count} lines. Main indexed symbols: {names}{suffix}."
    return f"{language} source file with {line_count} lines. Full source is available through the raw file link."


def _fence_language(language: str) -> str:
    return {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "cpp": "cpp",
        "c": "c",
        "shell": "bash",
        "sql": "sql",
        "yaml": "yaml",
        "json": "json",
        "toml": "toml",
    }.get(language, "")


__all__ = ["CODE_EXTENSIONS", "CodeSymbol", "build_code_markdown", "extract_code_symbols", "parse_code_markdown"]
