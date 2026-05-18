# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta는 개인 문서, Agent 메모리, 문서 지능을 위한 로컬 CLI 지식 인프라입니다. PDF, Office, 이미지, 오디오, 코드, HTML, Markdown, 노트를 안정적인 Markdown Wiki로 변환하고, 벡터 검색과 재사용 가능한 메모리 계층을 제공합니다.

## Installation

PyPI에서 설치:

```bash
pip install little-heta
```

로컬 저장소에서 설치:

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

`heta init`에는 LLM API key가 필요합니다. PDF 및 Office 파싱에는 선택적으로 MinerU를 사용할 수 있습니다: https://mineru.net/.

## Core Concepts

- **Wiki first**: 원본 파일을 안정적인 번호가 있는 Markdown Wiki로 컴파일합니다.
- **Vector Wiki**: Markdown 구조에 따라 페이지를 나누어 관련 섹션을 더 빠르게 찾습니다.
- **Memory reuse**: `heta ask`는 비용이 큰 검색 결과를 메모리로 저장하고 이후 질문에서 재사용합니다.
- **Agent skills**: `heta init`은 Codex와 Claude Code용 Little Heta skill을 자동 설치합니다.

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
