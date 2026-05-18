# Little Heta

[English](../../README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Português](README.pt.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

## What is Heta

Little Heta est une infrastructure locale de connaissance en ligne de commande pour les documents personnels, la mémoire d'agents et l'intelligence documentaire. Il transforme les PDF, fichiers Office, images, fichiers audio, code, HTML, Markdown et notes en wiki Markdown stable, avec recherche vectorielle et mémoire réutilisable.

## Installation

Installer depuis PyPI :

```bash
pip install little-heta
```

Depuis une copie locale du dépôt :

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

`heta init` nécessite une clé API LLM. L'analyse des PDF et fichiers Office peut utiliser MinerU en option : https://mineru.net/.

## Core Concepts

- **Wiki first** : les fichiers bruts deviennent des pages Markdown stables avec identifiants numériques.
- **Vector Wiki** : les pages sont découpées selon la structure Markdown pour retrouver les bonnes sections.
- **Memory reuse** : `heta ask` enregistre les connaissances utiles de la base pour les réutiliser plus tard.
- **Agent skills** : `heta init` installe la skill Little Heta pour Codex et Claude Code.

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
