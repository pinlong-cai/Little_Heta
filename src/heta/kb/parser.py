"""File parsing for Little Heta KB."""

from __future__ import annotations

import time
from pathlib import Path

import requests

from heta.config.schema import HetaConfig
from heta.kb.image_parser import IMAGE_EXTENSIONS, parse_image_markdown
from heta.kb.models import ParsedDocument
from heta.kb.text import extract_title


def parse_document(source_path: Path, archived_path: Path, config: HetaConfig) -> ParsedDocument:
    suffix = source_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        markdown = source_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        markdown = _parse_pdf_with_mineru(archived_path, config)
    elif suffix in IMAGE_EXTENSIONS:
        markdown = parse_image_markdown(source_path, archived_path, config)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    title = extract_title(markdown, source_path.stem.replace("_", " ").replace("-", " ").title())
    return ParsedDocument(
        source_path=source_path,
        archived_path=archived_path,
        title=title,
        markdown_content=markdown,
        source_name=archived_path.name,
        metadata={"extension": suffix},
    )


def _parse_pdf_with_mineru(path: Path, config: HetaConfig) -> str:
    if not config.mineru.enable:
        raise ValueError(f"PDF parsing requires MinerU: {path.name}")
    if config.mineru.provider == "local":
        return _parse_pdf_with_local_mineru(path, config.mineru.endpoint or "")
    if config.mineru.provider == "cloud":
        return _parse_pdf_with_cloud_mineru(path)
    raise ValueError("Invalid MinerU configuration.")


def _parse_pdf_with_local_mineru(path: Path, endpoint: str) -> str:
    url = endpoint.rstrip("/") + "/file_parse"
    with path.open("rb") as file:
        response = requests.post(url, files={"file": (path.name, file, "application/pdf")}, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(f"MinerU local parse failed: HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
        for key in ("markdown", "content", "text", "md"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("markdown", "content", "text", "md"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        raise RuntimeError("MinerU local response did not include markdown content.")

    return response.text


def _parse_pdf_with_cloud_mineru(path: Path) -> str:
    create_response = requests.post(
        "https://mineru.net/api/v1/agent/parse/file",
        json={
            "file_name": path.name,
            "language": "ch",
            "enable_table": True,
            "is_ocr": False,
            "enable_formula": True,
        },
        timeout=30,
    )
    if create_response.status_code != 200:
        raise RuntimeError(f"MinerU cloud task creation failed: HTTP {create_response.status_code}")

    payload = create_response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"MinerU cloud task creation failed: {payload.get('msg')}")
    task_id = payload.get("data", {}).get("task_id")
    file_url = payload.get("data", {}).get("file_url")
    if not task_id or not file_url:
        raise RuntimeError("MinerU cloud did not return task_id and file_url.")

    with path.open("rb") as file:
        upload_response = requests.put(file_url, data=file, timeout=120)
    if upload_response.status_code not in {200, 204}:
        raise RuntimeError(f"MinerU cloud upload failed: HTTP {upload_response.status_code}")

    markdown_url = _poll_mineru_markdown_url(task_id)
    markdown_response = requests.get(markdown_url, timeout=60)
    if markdown_response.status_code != 200:
        raise RuntimeError(f"MinerU markdown download failed: HTTP {markdown_response.status_code}")
    markdown = markdown_response.text.strip()
    if not markdown:
        raise RuntimeError("MinerU cloud returned empty markdown.")
    return markdown


def _poll_mineru_markdown_url(task_id: str, *, timeout_seconds: int = 180) -> str:
    deadline = time.time() + timeout_seconds
    url = f"https://mineru.net/api/v1/agent/parse/{task_id}"
    while time.time() < deadline:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"MinerU cloud polling failed: HTTP {response.status_code}")
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"MinerU cloud polling failed: {payload.get('msg')}")
        data = payload.get("data", {})
        state = data.get("state")
        if state == "done":
            markdown_url = data.get("markdown_url")
            if not markdown_url:
                raise RuntimeError("MinerU cloud result did not include markdown_url.")
            return markdown_url
        if state == "failed":
            raise RuntimeError(f"MinerU cloud parsing failed: {data.get('err_msg') or data.get('err_code')}")
        time.sleep(2)
    raise TimeoutError(f"MinerU cloud parsing timed out after {timeout_seconds}s: {task_id}")
