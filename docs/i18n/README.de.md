# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta ist eine lokale CLI-Wissensinfrastruktur für persönliche Dokumente, Agenten-Gedächtnis und Dokumentenintelligenz. Es verwandelt PDFs, Office-Dateien, Bilder, Audio, Code, HTML, Markdown und Notizen in ein stabiles Markdown-Wiki mit Vektorsuche und wiederverwendbarem Gedächtnis.

## Installation

Von PyPI installieren:

```bash
pip install little-heta
```

Aus einem lokalen Checkout:

```bash
pip install -e .
```

## Quick Start

```bash
heta init
heta status
heta insert ./docs
heta ask "What does my knowledge base say about this?"
```

`heta init` benötigt einen LLM-API-Schlüssel. Für PDF- und Office-Parsing kann optional MinerU verwendet werden: https://mineru.net/.

## Core Concepts

- **Wiki first**: Rohdateien werden zu stabilen Markdown-Wiki-Seiten mit numerischen IDs.
- **Vector Wiki**: Seiten werden anhand der Markdown-Struktur in Abschnitte geteilt.
- **Memory reuse**: `heta ask` speichert nützliche Erkenntnisse aus der Wissensbasis zur späteren Wiederverwendung.
- **Agent skills**: `heta init` installiert den Little-Heta-Skill für Codex und Claude Code.

## Minimal Examples

```bash
heta query "What does the design doc say?"
heta remember "We decided to use Postgres."
heta recall "database decision"
heta skill
```

## Community Links

- GitHub: https://github.com/KnowledgeXLab/Little_Heta
- Team: https://knowledgexlab.github.io/
- License: MIT
