# Little Heta

Little Heta is a lightweight command line tool for personal knowledge, memory,
and document intelligence workflows. It converts local documents into a
Markdown wiki, keeps wiki page identity stable, and can maintain a SQLite
vector index for faster semantic retrieval.

## Status

This repository is an early `v0.1.0` implementation. The current focus is a
fast local workflow for initialization, document insertion, wiki maintenance,
and optional vector indexing.

## Features

- Interactive first-time setup with `heta init`
- Provider configuration for Qwen, ChatGPT, or Gemini
- Optional MinerU integration for PDF parsing
- Markdown wiki generation under the Little Heta workspace
- Stable numeric wiki page ids in page filenames
- Optional SQLite + sqlite-vec wiki chunk index
- CLI status view with provider, MinerU, KB, wiki, and space usage summaries

## Install

From a local checkout:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

Initialize Little Heta:

```bash
heta init
```

The wizard writes configuration to:

```text
~/.heta/heta.yaml
```

Check the current workspace and provider status:

```bash
heta status
```

Insert one file or a directory:

```bash
heta insert ./docs
```

Large PDFs are split before parsing by default. Disable this behavior when you
want to parse a PDF as one source file:

```bash
heta insert --no-pdf-planning ./large.pdf
```

Ask a read-only question against the wiki:

```bash
heta query "What is HetaGen?"
```

Clean wiki pages and the vector database while keeping raw files:

```bash
heta clean
```

Manage vector indexing:

```bash
heta vector status
heta vector on
heta vector off
```

## Workspace

Little Heta stores local runtime data under:

```text
~/.heta/
```

The workspace contains raw source files, generated wiki pages, worktrees, and
the local database used by the vector index. Runtime workspace data is not
intended to be committed to this repository.

## Development

Run tests:

```bash
pytest
```

Project layout:

```text
src/heta/          CLI, config, providers, and KB implementation
tests/             unit tests
pyproject.toml     package metadata and dependencies
```

## License

Little Heta is released under the MIT License. See [LICENSE](LICENSE).
