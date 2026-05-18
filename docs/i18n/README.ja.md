# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta は、個人ドキュメント、Agent メモリ、ドキュメントインテリジェンスのためのローカル CLI 知識基盤です。PDF、Office、画像、音声、コード、HTML、Markdown、ノートを安定した Markdown Wiki に変換し、ベクトル検索と再利用可能なメモリ層を提供します。

## Installation

PyPI からインストール：

```bash
pip install little-heta
```

ローカルリポジトリから：

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

`heta init` には LLM API key が必要です。PDF と Office の解析には MinerU を任意で利用できます：https://mineru.net/。

## Core Concepts

- **Wiki first**：元ファイルを安定した番号付き Markdown Wiki に変換します。
- **Vector Wiki**：Markdown の階層に沿ってページを分割し、必要な章へ素早く到達します。
- **Memory reuse**：`heta ask` は高コストな検索結果を知識として保存し、後続の質問で再利用できます。
- **Agent skills**：`heta init` は Codex と Claude Code 用の Little Heta skill を自動でインストールします。

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
