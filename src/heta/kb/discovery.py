"""Input discovery for `heta insert`."""

from __future__ import annotations

from pathlib import Path

from heta.config.schema import HetaConfig

PLAIN_EXTENSIONS = {".md", ".markdown", ".txt"}
MINERU_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def supported_extensions(config: HetaConfig) -> set[str]:
    extensions = set(PLAIN_EXTENSIONS) | IMAGE_EXTENSIONS
    if config.mineru.enable:
        extensions |= MINERU_EXTENSIONS
    return extensions


def collect_insert_files(targets: list[Path], config: HetaConfig) -> list[Path]:
    effective_targets = targets or [Path.cwd()]
    extensions = supported_extensions(config)
    files: list[Path] = []

    for target in effective_targets:
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")
        if target.is_file():
            _add_supported_file(files, target, extensions)
            continue
        if target.is_dir():
            for child in sorted(target.rglob("*")):
                if _is_ignored_path(child):
                    continue
                if child.is_file():
                    _add_supported_file(files, child, extensions)
            continue
        raise ValueError(f"Unsupported path type: {target}")

    unique: dict[Path, Path] = {}
    for file in files:
        unique[file.resolve()] = file
    return list(unique.values())


def _add_supported_file(files: list[Path], path: Path, extensions: set[str]) -> None:
    suffix = path.suffix.lower()
    if suffix not in extensions:
        if suffix in MINERU_EXTENSIONS:
            raise ValueError(
                f"{path} requires MinerU. Run `heta init` and enable MinerU, or skip this file."
            )
        if path.is_file():
            return
    files.append(path)


def _is_ignored_path(path: Path) -> bool:
    ignored = {".git", ".worktrees", "__pycache__", ".pytest_cache", "workspace"}
    return any(part in ignored for part in path.parts)
