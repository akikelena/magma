"""Stage 1 of the Documentation Processing Agent: source detection + parsing.

Detects whether a raw source is structured (HTML/OpenAPI/reStructuredText)
or unstructured (free-form Markdown/prose) and parses it into a normalized
tree of sections, each carrying a heading path and a flag for whether it is
a code block. Kept dependency-light (stdlib only) so the deterministic
first stage never depends on model availability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass
class ParsedSection:
    heading_path: list[str]
    content: str
    is_code_block: bool = False


SourceKind = str  # "html" | "markdown" | "openapi" | "rst" | "unstructured"

_HTML_TAG_RE = re.compile(r"<\s*[a-zA-Z][a-zA-Z0-9]*[^>]*>")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_MD_FENCE_RE = re.compile(r"^```")
_OPENAPI_HINT_RE = re.compile(r'"openapi"\s*:|^openapi:\s*["\']?3', re.MULTILINE)
_RST_HEADING_RE = re.compile(r"^(=+|-+|~+)\s*$")


def detect_source_kind(raw: str) -> SourceKind:
    if _OPENAPI_HINT_RE.search(raw):
        return "openapi"
    if _HTML_TAG_RE.search(raw) and raw.strip().startswith(("<", "<!DOCTYPE", "<!doctype")):
        return "html"
    if _MD_HEADING_RE.search(raw) or _MD_FENCE_RE.search(raw):
        return "markdown"
    lines = raw.splitlines()
    for i in range(1, len(lines)):
        if _RST_HEADING_RE.match(lines[i]) and lines[i - 1].strip():
            return "rst"
    return "unstructured"


class _HeadingCollectingHTMLParser(HTMLParser):
    """Minimal HTML->section extractor: walks h1-h6/p/pre tags in document order."""

    _HEADING_TAGS = {f"h{i}" for i in range(1, 7)}

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[ParsedSection] = []
        self._heading_stack: list[tuple[int, str]] = []
        self._current_tag: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._HEADING_TAGS or tag in {"p", "pre", "code"}:
            self._flush()
            self._current_tag = tag

    def handle_endtag(self, tag):
        if tag == self._current_tag:
            self._flush()
            self._current_tag = None

    def handle_data(self, data):
        if self._current_tag is not None and data.strip():
            self._buffer.append(data.strip())

    def _flush(self) -> None:
        if not self._buffer or self._current_tag is None:
            return
        text = " ".join(self._buffer)
        if self._current_tag in self._HEADING_TAGS:
            level = int(self._current_tag[1])
            self._heading_stack = [h for h in self._heading_stack if h[0] < level]
            self._heading_stack.append((level, text))
        else:
            path = [h[1] for h in self._heading_stack]
            self.sections.append(
                ParsedSection(heading_path=path, content=text, is_code_block=self._current_tag in {"pre", "code"})
            )
        self._buffer = []


def parse_html(raw: str) -> list[ParsedSection]:
    parser = _HeadingCollectingHTMLParser()
    parser.feed(raw)
    parser._flush()
    return parser.sections


def parse_markdown(raw: str) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_code_block = False

    def flush(is_code: bool) -> None:
        text = "\n".join(buffer).strip()
        if text:
            path = [h[1] for h in heading_stack]
            sections.append(ParsedSection(heading_path=path, content=text, is_code_block=is_code))
        buffer.clear()

    for line in raw.splitlines():
        if _MD_FENCE_RE.match(line):
            if in_code_block:
                flush(is_code=True)
                in_code_block = False
            else:
                flush(is_code=False)
                in_code_block = True
            continue
        if in_code_block:
            buffer.append(line)
            continue
        heading_match = _MD_HEADING_RE.match(line)
        if heading_match:
            flush(is_code=False)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, title))
            continue
        if not line.strip():
            flush(is_code=False)
            continue
        buffer.append(line)
    flush(is_code=in_code_block)
    return sections


def parse_unstructured(raw: str) -> list[ParsedSection]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return [ParsedSection(heading_path=[], content=p) for p in paragraphs]


def parse_source(raw: str) -> tuple[SourceKind, list[ParsedSection]]:
    """Detect the source kind and parse it into a normalized section tree."""
    kind = detect_source_kind(raw)
    if kind == "html":
        return kind, parse_html(raw)
    if kind in ("markdown", "openapi", "rst"):
        # OpenAPI/RST share enough structure with Markdown-style heading
        # detection for our purposes; a dedicated parser is a drop-in swap.
        return kind, parse_markdown(raw)
    return kind, parse_unstructured(raw)
