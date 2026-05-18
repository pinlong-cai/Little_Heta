# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta 是一個本地 CLI 知識基礎設施，用於個人文件、Agent 記憶與文件智慧。它會將 PDF、Office、圖片、音訊、程式碼、HTML、Markdown 和筆記轉成穩定的 Markdown Wiki，並提供向量檢索與可重複使用的記憶層。

## Installation

從 PyPI 安裝：

```bash
pip install little-heta
```

從本地倉庫安裝：

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

`heta init` 需要你準備一個 LLM API key。PDF 和 Office 解析可選接入 MinerU：https://mineru.net/。

## Core Concepts

- **Wiki first**：原始文件會被編譯成帶穩定編號的 Markdown Wiki。
- **Vector Wiki**：依照 Markdown 層級切分頁面，讓查詢更容易命中特定章節。
- **Memory reuse**：`heta ask` 可以把昂貴查詢得到的知識沉澱為記憶，供後續問題重複使用。
- **Agent skills**：`heta init` 會自動安裝 Codex 和 Claude Code 可使用的 Little Heta skill。

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
