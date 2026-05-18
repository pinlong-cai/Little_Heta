# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta é uma infraestrutura local de conhecimento via CLI para documentos pessoais, memória de agentes e inteligência documental. Ele converte PDFs, arquivos Office, imagens, áudio, código, HTML, Markdown e notas em uma wiki Markdown estável, com busca vetorial e memória reutilizável.

## Installation

Instale pelo PyPI:

```bash
pip install little-heta
```

A partir de um repositório local:

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

`heta init` exige uma chave de API de um LLM. A análise de PDF e Office pode usar o MinerU opcionalmente: https://mineru.net/.

## Core Concepts

- **Wiki first**: arquivos originais viram páginas Markdown estáveis com identificadores numéricos.
- **Vector Wiki**: páginas são divididas pela estrutura Markdown para recuperação por seção.
- **Memory reuse**: `heta ask` salva conhecimentos úteis da base para reutilização em perguntas futuras.
- **Agent skills**: `heta init` instala a skill do Little Heta para Codex e Claude Code.

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
