"""Structure-preserving HTML parsing for Little Heta KB inserts."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
NOISE_TAGS = {"script", "style", "nav", "footer", "aside", "iframe", "button", "noscript"}


@dataclass(frozen=True)
class HtmlAsset:
    id: str
    raw_path: str | None
    original_src: str
    alt: str
    title: str
    section: str
    near_text_before: str
    near_text_after: str


def parse_html_markdown(source_path: Path, archived_path: Path) -> str:
    html = source_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    _remove_noise(soup)

    title = _page_title(soup, source_path)
    description = _description(soup)
    body = soup.find("body") or soup
    asset_dir = archived_path.parent / "assets" / archived_path.stem
    converter = _HtmlMarkdownConverter(source_path=source_path, asset_dir=asset_dir, asset_stem=archived_path.stem)
    content = converter.convert(body).strip()
    summary = _summary(title, description, content)

    if converter.assets:
        asset_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source_html": archived_path.name,
            "assets": [asdict(asset) for asset in converter.assets],
        }
        (asset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return build_html_markdown(
        title=f"Web Page - {title}",
        source_name=archived_path.name,
        summary=summary,
        content=content,
    )


def build_html_markdown(*, title: str, source_name: str, summary: str, content: str) -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"sources: [{source_name}]\n"
        f"updated: {date.today().isoformat()}\n"
        "---\n\n"
        "## Summary\n"
        f"{summary.strip()}\n\n"
        "## Content\n\n"
        f"{content.strip() or 'No main HTML content extracted.'}\n\n"
        "## Source\n"
        f"- {source_name}\n"
    )


class _HtmlMarkdownConverter:
    def __init__(self, *, source_path: Path, asset_dir: Path, asset_stem: str) -> None:
        self.source_path = source_path
        self.asset_dir = asset_dir
        self.asset_stem = asset_stem
        self.assets: list[HtmlAsset] = []
        self.section_stack: list[str] = []
        self.recent_text: str = ""

    def convert(self, node: Tag) -> str:
        parts = [self._convert_child(child) for child in node.children]
        return _compact_blocks("\n".join(part for part in parts if part.strip()))

    def _convert_child(self, node) -> str:
        if isinstance(node, NavigableString):
            text = _clean_text(str(node))
            self._remember_text(text)
            return text
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower() if node.name else ""
        if name in NOISE_TAGS:
            return ""
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return self._heading(node, int(name[1]))
        if name == "p":
            return self._paragraph(node)
        if name in {"ul", "ol"}:
            return self._list(node, ordered=name == "ol")
        if name == "li":
            return self._inline_children(node)
        if name == "blockquote":
            text = self.convert(node)
            return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
        if name == "pre":
            return self._pre(node)
        if name == "code":
            return f"`{_clean_text(node.get_text(' ', strip=True))}`"
        if name == "table":
            return self._table(node)
        if name == "img":
            return self._image(node)
        if name == "br":
            return "\n"
        if name in {"strong", "b"}:
            return f"**{self._inline_children(node)}**"
        if name in {"em", "i"}:
            return f"*{self._inline_children(node)}*"
        if name == "a":
            return self._link(node)

        return self.convert(node)

    def _heading(self, node: Tag, html_level: int) -> str:
        text = _clean_text(node.get_text(" ", strip=True))
        if not text:
            return ""
        markdown_level = min(6, html_level + 2)
        depth = markdown_level - 2
        self.section_stack = self.section_stack[: max(0, depth - 1)] + [text]
        self._remember_text(text)
        return f"{'#' * markdown_level} {text}"

    def _paragraph(self, node: Tag) -> str:
        text = self._inline_children(node)
        self._remember_text(text)
        return text

    def _list(self, node: Tag, *, ordered: bool) -> str:
        lines: list[str] = []
        index = 1
        for child in node.find_all("li", recursive=False):
            text = _compact_inline(self._inline_children(child))
            if not text:
                continue
            marker = f"{index}." if ordered else "-"
            lines.append(f"{marker} {text}")
            index += 1
        return "\n".join(lines)

    def _pre(self, node: Tag) -> str:
        code = node.get_text("\n", strip=False).strip("\n")
        language = ""
        code_tag = node.find("code")
        if code_tag:
            classes = " ".join(code_tag.get("class", []))
            match = re.search(r"language-([\w+-]+)", classes)
            if match:
                language = match.group(1)
        return f"```{language}\n{code}\n```"

    def _table(self, node: Tag) -> str:
        rows: list[list[str]] = []
        for tr in node.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if cells:
                rows.append([_compact_inline(cell.get_text(" ", strip=True)) for cell in cells])
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        separator = ["---"] * width
        body = normalized[1:]
        table_lines = [_markdown_row(header), _markdown_row(separator), *[_markdown_row(row) for row in body]]
        text = "\n".join(table_lines)
        self._remember_text(" ".join(" ".join(row) for row in normalized))
        return text

    def _image(self, node: Tag) -> str:
        src = _img_src(node)
        if not src:
            return ""
        alt = _clean_text(str(node.get("alt") or ""))
        title = _clean_text(str(node.get("title") or ""))
        markdown_src, raw_path = self._image_path(src)
        label = alt or title or Path(urlparse(src).path).name or "HTML image"
        section = self.section_stack[-1] if self.section_stack else ""
        asset = HtmlAsset(
            id=f"img-{len(self.assets) + 1:03d}",
            raw_path=raw_path,
            original_src=src,
            alt=alt,
            title=title,
            section=section,
            near_text_before=self.recent_text,
            near_text_after="",
        )
        self.assets.append(asset)
        note = alt or title
        if note:
            return f"![{_escape_brackets(label)}](<{markdown_src}>)\n\nImage note: {note}."
        return f"![{_escape_brackets(label)}](<{markdown_src}>)"

    def _image_path(self, src: str) -> tuple[str, str | None]:
        parsed = urlparse(src)
        if parsed.scheme in {"http", "https", "data"} or src.startswith("//"):
            return src, None
        local = (self.source_path.parent / src).resolve()
        if not local.exists() or local.suffix.lower() not in IMAGE_EXTENSIONS:
            return src, None
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        target = self.asset_dir / f"img-{len(self.assets) + 1:03d}{local.suffix.lower()}"
        shutil.copy2(local, target)
        raw_path = f"raw/assets/{self.asset_stem}/{target.name}"
        markdown_path = f"../../raw/assets/{self.asset_stem}/{target.name}"
        return markdown_path, raw_path

    def _link(self, node: Tag) -> str:
        text = self._inline_children(node) or _clean_text(node.get_text(" ", strip=True))
        href = str(node.get("href") or "").strip()
        if not href:
            return text
        return f"[{_escape_brackets(text)}](<{href}>)"

    def _inline_children(self, node: Tag) -> str:
        parts = [self._convert_child(child) for child in node.children]
        return _compact_inline(" ".join(part for part in parts if part.strip()))

    def _remember_text(self, text: str) -> None:
        cleaned = _compact_inline(text)
        if cleaned:
            self.recent_text = cleaned[-240:]


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(list(NOISE_TAGS)):
        tag.decompose()


def _page_title(soup: BeautifulSoup, source_path: Path) -> str:
    for selector in ("h1", "title"):
        tag = soup.find(selector)
        if tag:
            text = _clean_text(tag.get_text(" ", strip=True))
            if text:
                return text
    return source_path.stem.replace("_", " ").replace("-", " ").title()


def _description(soup: BeautifulSoup) -> str:
    for attrs in ({"name": "description"}, {"property": "og:description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return _clean_text(str(tag.get("content")))
    return ""


def _summary(title: str, description: str, content: str) -> str:
    if description:
        return description
    text = _strip_markdown(content)
    if text:
        return text[:240].rstrip() + ("..." if len(text) > 240 else "")
    return f"HTML page about {title}."


def _img_src(node: Tag) -> str:
    for key in ("data-src", "data-original", "data-lazy-src", "src"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    srcset = node.get("srcset")
    if isinstance(srcset, str) and srcset.strip():
        return srcset.split(",")[-1].strip().split()[0]
    return ""


def _markdown_row(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compact_inline(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _compact_blocks(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                compact.append("")
            blank = True
            continue
        compact.append(line)
        blank = False
    return "\n".join(compact).strip()


def _strip_markdown(text: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[#*_`>|-]+", " ", cleaned)
    return _clean_text(cleaned)


def _escape_brackets(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")


__all__ = ["HTML_EXTENSIONS", "HtmlAsset", "build_html_markdown", "parse_html_markdown"]
