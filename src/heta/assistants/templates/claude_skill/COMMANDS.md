# Little Heta — full command reference

Every `heta` command. The four core commands (`ask`, `query`, `recall`,
`remember`) are covered in `SKILL.md`; this file documents the rest. Run any
command with the Bash tool.

## Setup

### `heta init`
Interactive first-time setup — LLM provider + API key, MinerU document
parsing, and so on. Interactive: the user must run this themselves. Never run
it for them.

### `heta status`
Show what is configured and how much is indexed. No arguments.

## Indexing documents

### `heta insert [PATHS...]`
Add files or folders to the knowledge base. Defaults to the current directory.
Supports PDF, Office, images, audio, code, HTML, and Markdown.

```bash
heta insert ./docs
heta insert report.pdf notes.md
```

### `heta clean [-y]`
Remove generated wiki pages and the vector index. Original raw files are kept.
`-y` / `--yes` skips the confirmation prompt.

## Core command options

`ask`, `query`, `recall`, and `remember` are documented in `SKILL.md`. Their
extra options:

- `heta ask "<question>" [-k N] [-d]` — `-k` / `--top-k` results per layer
  (default 5); `-d` / `--debug` shows agent steps and evidence.
- `heta query "<question>" [--top-k N]` — `--top-k` initial vector matches,
  1–10 (default 5).
- `heta recall "<query>" [-k N] [-d]` — `-k` / `--top-k` results per layer
  (default 10); `-d` / `--debug` shows layer ranking, reason, and scored evidence.
- `heta remember "<text>"` — no extra options.

## Inspecting & clearing memory

### `heta mem-show insights [-s SOURCE] [-q QUESTION] [-n LIMIT] [-f]`
List stored KB-insight memories, newest first. `-s` / `--source` filters by
source path, `-q` / `--question` filters by question, `-n` / `--limit` caps
rows (default 50), `-f` / `--full` shows full untruncated text.

### `heta mem-clean [-y]`
Erase all saved memory. `-y` / `--yes` skips confirmation. Irreversible.

## Settings

### `heta vector on | off | status`
Turn document search vector indexing on or off, or show its current state.

### `heta insert-planning on | off | status`
Turn smart insert planning (such as large-PDF splitting) on or off, or show
its current state.
