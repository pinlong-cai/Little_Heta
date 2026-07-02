# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta 是一个本地 CLI 知识基础设施，用于个人文档、Agent 记忆和文档智能。它把 PDF、Office、图片、音频、代码、HTML、Markdown 和笔记转成稳定的 Markdown Wiki，并提供向量检索和可复用的记忆层。

## Installation

从 PyPI 安装：

```bash
pip install little-heta
```

从本地仓库安装：

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

`heta init` 需要你准备一个 LLM API key，可选择 Qwen、ChatGPT、Gemini 或 custom。custom 支持 LiteLLM-native 模型名（如 `openai/gpt-5.4-nano`，可不填 base URL），也支持裸模型名加 OpenAI-compatible `/v1` base URL。PDF 和 Office 解析可选接入 MinerU：https://mineru.net/apiManage/docs。

## Core Concepts

- **Wiki foundation**：Wiki 是知识基础层，原始文件会被编译成带稳定编号的 Markdown 页面。
- **Vector Wiki**：按照 Markdown 层级切分页面，让查询更容易命中具体章节。
- **Memory reuse**：`heta ask` 可以把昂贵查询得到的知识沉淀为记忆，后续问题复用。
- **Agent skills**：`heta init` 会自动安装 Codex 和 Claude Code 可使用的 Little Heta skill。

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
