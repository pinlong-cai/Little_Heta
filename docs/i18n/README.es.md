# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta es una infraestructura local de conocimiento por CLI para documentos personales, memoria de agentes e inteligencia documental. Convierte PDFs, archivos de Office, imágenes, audio, código, HTML, Markdown y notas en una wiki Markdown estable, con búsqueda vectorial y memoria reutilizable.

## Installation

Instalar desde PyPI:

```bash
pip install little-heta
```

Desde una copia local del repositorio:

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

`heta init` requiere una clave API de un LLM. El análisis de PDF y Office puede usar MinerU de forma opcional: https://mineru.net/.

## Core Concepts

- **Wiki first**: los archivos originales se convierten en páginas Markdown estables con identificadores numéricos.
- **Vector Wiki**: las páginas se dividen según la estructura Markdown para recuperar secciones concretas.
- **Memory reuse**: `heta ask` guarda conocimientos útiles de la base de conocimiento para reutilizarlos después.
- **Agent skills**: `heta init` instala la skill de Little Heta para Codex y Claude Code.

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
