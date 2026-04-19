"""HtmlExtractor — extracts sections from HTML files."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup

from policygpt.config import Config
from policygpt.constants import FileExtension, OCRProvider
from policygpt.ingestion.extraction.parsers.base import BaseExtractor
from policygpt.models.extraction import ExtractedSection
from policygpt.ingestion.extraction.ocr import (
    ClaudeVisionOCR,
    ImageFetcher,
    TextractOCR,
    image_bytes_to_data_uri,
)


class HtmlExtractor(BaseExtractor):
    """Extracts document title and sections from ``.html`` / ``.htm`` files."""

    SUPPORTED_EXTENSIONS: frozenset[FileExtension] = frozenset({
        FileExtension.HTML,
        FileExtension.HTM,
    })

    HTML_SEMANTIC_TAGS = (
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "table",
        "blockquote", "pre", "dd", "dt",
    )
    HTML_BLOCK_TAGS = (
        "article", "section", "main", "aside", "header", "footer", "nav",
        "div", "p", "li", "table", "blockquote", "pre",
        "dd", "dt", "dl", "ul", "ol",
        "h1", "h2", "h3", "h4", "h5", "h6",
    )
    HTML_CONTAINER_TAGS = {
        "article", "section", "main", "aside", "header", "footer", "nav",
        "div", "table", "tr", "dl", "ul", "ol",
    }
    HTML_NOISE_MARKERS = {
        "sidebar", "menu", "breadcrumb", "breadcrumbs", "pagination",
        "table-of-contents", "toc", "toolbar", "share", "social",
    }
    HTML_HEADING_MARKERS = {
        "title", "heading", "header", "question",
    }
    HTML_BODY_MARKERS = {
        "body", "content", "copy", "text", "details", "description",
        "answer", "summary", "subtitle", "issuer", "caption", "label",
    }

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if config.ocr_enabled:
            if getattr(config, "ocr_provider", OCRProvider.TEXTRACT) == OCRProvider.CLAUDE:
                self._ocr: TextractOCR | ClaudeVisionOCR | None = ClaudeVisionOCR()
            else:
                self._ocr = TextractOCR(
                    region=config.bedrock_region,
                    min_confidence=config.ocr_min_confidence,
                )
        else:
            self._ocr = None
        self._image_fetcher = ImageFetcher(max_bytes=config.image_max_bytes)

    # ── Public interface ───────────────────────────────────────────────────────

    def extract(self, path: str) -> tuple[str, list[ExtractedSection]]:
        return self.extract_from_html(path)

    def extract_from_html(self, path: str) -> tuple[str, list[ExtractedSection]]:
        import time as _t
        _fname = Path(path).name
        _t0 = _t.perf_counter()

        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        print(f"      [extract] {_fname} — read {len(html):,} bytes ({_t.perf_counter()-_t0:.2f}s)", flush=True)

        _ts = _t.perf_counter()
        raw_text = self._extract_text_from_raw_html(html)
        print(f"      [extract] {_fname} — raw_text {len(raw_text):,} chars ({_t.perf_counter()-_ts:.2f}s)", flush=True)

        _ts = _t.perf_counter()
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception:
                soup = None
        print(f"      [extract] {_fname} — BS4 parse done ({_t.perf_counter()-_ts:.2f}s)", flush=True)

        if soup is None:
            title = self._select_text_document_title(
                path=path,
                lines=[line.strip() for line in raw_text.split("\n") if line.strip()],
            )
            return title, self._group_units_into_sections(self._build_html_text_fallback_units(raw_text))

        # Image pass — OCR-capable images become standalone sections;
        # images without OCR text are collected as orphan display-only images.
        html_dir = Path(path).resolve().parent
        image_sections: list[ExtractedSection] = []
        orphan_images: list[str] = []

        for img_idx, img_tag in enumerate(soup.find_all("img")):
            src = img_tag.get("src") or img_tag.get("data-src") or ""
            alt = self.clean_whitespace(img_tag.get("alt") or "")

            image_bytes, mime_type = self._image_fetcher.fetch(src, html_dir)
            if not image_bytes:
                continue

            data_uri = image_bytes_to_data_uri(image_bytes, mime_type)

            ocr_text = ""
            if self._ocr is not None:
                try:
                    if hasattr(self._ocr, "extract_from_bytes"):
                        ocr_text = self._ocr.extract_from_bytes(image_bytes, mime_type or "image/png")
                    else:
                        ocr_text = self._ocr.extract_from_src(src, html_dir)
                except Exception:
                    pass

            if ocr_text:
                section_title = f"Image: {alt}" if alt else f"Image {img_idx + 1}"
                image_sections.append(ExtractedSection(
                    title=section_title,
                    text=ocr_text,
                    images=[data_uri],
                ))
            else:
                orphan_images.append(data_uri)

        _ts = _t.perf_counter()
        for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link"]):
            tag.decompose()
        print(f"      [extract] {_fname} — tag decompose done ({_t.perf_counter()-_ts:.2f}s)", flush=True)

        html_title = ""
        if soup.title and soup.title.text.strip():
            html_title = self.clean_whitespace(soup.title.text)

        _ts = _t.perf_counter()
        primary_units = self._extract_html_semantic_units(soup)
        print(f"      [extract] {_fname} — semantic units: {len(primary_units)} ({_t.perf_counter()-_ts:.2f}s)", flush=True)

        if self._has_substantive_body_units(primary_units):
            units = primary_units
        else:
            _ts = _t.perf_counter()
            fallback_units = self._extract_html_block_units(soup)
            print(f"      [extract] {_fname} — block units: {len(fallback_units)} ({_t.perf_counter()-_ts:.2f}s)", flush=True)

            _ts = _t.perf_counter()
            body_root = soup.body or soup
            body_text = self.clean_whitespace(body_root.get_text("\n", strip=True))
            print(f"      [extract] {_fname} — get_text {len(body_text):,} chars ({_t.perf_counter()-_ts:.2f}s)", flush=True)

            fallback_text = body_text if len(body_text) >= len(raw_text) else raw_text

            units = primary_units
            if self._should_prefer_html_block_units(primary_units, fallback_units, body_text):
                units = fallback_units

            if not self._has_substantive_body_units(units) and fallback_text:
                units = self._build_html_text_fallback_units(fallback_text)

        _ts = _t.perf_counter()
        title = self._select_document_title(path=path, html_title=html_title, units=units)
        print(f"      [extract] {_fname} — select_title done ({_t.perf_counter()-_ts:.2f}s)", flush=True)

        _ts = _t.perf_counter()
        text_sections = self._group_units_into_sections(units)

        if orphan_images and text_sections:
            text_sections[-1].images.extend(orphan_images)

        all_sections = text_sections + image_sections
        print(
            f"      [extract] {_fname} — grouped into {len(text_sections)} text + "
            f"{len(image_sections)} OCR image section(s), "
            f"{len(orphan_images)} display-only image(s) ({_t.perf_counter()-_ts:.2f}s)",
            flush=True,
        )
        print(f"      [extract] {_fname} — TOTAL extract time {_t.perf_counter()-_t0:.2f}s", flush=True)
        return title, all_sections

    # ── Semantic unit extraction ───────────────────────────────────────────────

    def _extract_html_semantic_units(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        units: list[tuple[str, str]] = []
        seen_texts: set[str] = set()
        emitted_table_ids: set[int] = set()

        for node in (soup.body or soup).find_all(list(self.HTML_SEMANTIC_TAGS)):
            if self._is_html_noise_node(node) or self._has_html_noise_ancestor(node):
                continue

            node_name = node.name.lower()

            if node_name == "table":
                ancestor_table = node.find_parent("table")
                if ancestor_table is not None and id(ancestor_table) in emitted_table_ids:
                    continue
                table_text = self._table_to_plain_text(node)
                if not table_text:
                    continue
                normalized = self._normalize_html_text(table_text)
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    emitted_table_ids.add(id(node))
                    units.append(("table", table_text))
                continue

            text = self.clean_whitespace(node.get_text(" ", strip=True))
            if not text:
                continue
            normalized = self._normalize_html_text(text)
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            units.append((node_name, text))

        return units

    @classmethod
    def _table_to_plain_text(cls, table_node) -> str:
        """Convert an HTML table to pipe-delimited plain text, one row per line."""
        rows: list[str] = []
        for tr in table_node.find_all("tr"):
            if tr.find_parent("table") is not table_node:
                continue
            cells: list[str] = []
            for cell in tr.find_all(["th", "td"]):
                if cell.find_parent("tr") is not tr:
                    continue
                nested_table = cell.find("table")
                if nested_table:
                    nested_text = cls._table_to_plain_text(nested_table)
                    cell_text = re.sub(r"\s+", " ", nested_text.replace("\n", "; ")).strip()
                else:
                    cell_text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                cells.append(cell_text)
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                rows.append(row_text)
        return "\n".join(rows)

    def _extract_html_block_units(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        units: list[tuple[str, str]] = []
        seen_texts: set[str] = set()
        root = soup.body or soup

        for node in root.find_all(list(self.HTML_BLOCK_TAGS)):
            if self._is_html_noise_node(node) or self._has_html_noise_ancestor(node):
                continue
            if self._should_skip_html_container(node):
                continue

            text = self.clean_whitespace(node.get_text(" ", strip=True))
            if not text:
                continue

            normalized = self._normalize_html_text(text)
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            units.append((self._classify_html_block_tag(node=node, text=text), text))

        return units

    def _should_prefer_html_block_units(
        self,
        primary_units: list[tuple[str, str]],
        fallback_units: list[tuple[str, str]],
        body_text: str,
    ) -> bool:
        if not fallback_units:
            return False

        primary_body_chars = self._html_body_chars(primary_units)
        fallback_body_chars = self._html_body_chars(fallback_units)

        if not self._has_substantive_body_units(primary_units):
            return True

        if fallback_body_chars > primary_body_chars + 300 and fallback_body_chars >= int(primary_body_chars * 1.35):
            return True

        if not body_text:
            return False

        primary_coverage = primary_body_chars / max(1, len(body_text))
        fallback_coverage = fallback_body_chars / max(1, len(body_text))
        return fallback_coverage >= 0.45 and fallback_coverage > primary_coverage + 0.15

    def _build_html_text_fallback_units(self, body_text: str) -> list[tuple[str, str]]:
        lines = [line.strip() for line in body_text.split("\n") if line.strip()]
        units = self._build_text_units(lines)
        if self._has_substantive_body_units(units):
            return units
        return [("p", body_text)] if body_text else []

    def _extract_text_from_raw_html(self, html: str) -> str:
        if not html:
            return ""

        import time as _t
        _ts0 = _t.perf_counter()

        stripped = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
        print(f"      [raw] comments {_t.perf_counter()-_ts0:.2f}s", flush=True)

        _ts = _t.perf_counter()
        for _tag in ("script", "style", "noscript", "svg"):
            stripped = re.sub(
                rf"<{_tag}\b[^>]*>.*?</{_tag}\s*>",
                " ",
                stripped,
                flags=re.IGNORECASE | re.DOTALL,
            )
        print(f"      [raw] block-tag strip {_t.perf_counter()-_ts:.2f}s", flush=True)

        _ts = _t.perf_counter()
        stripped = re.sub(r"<(?:meta|link|img|br|hr|input)\b[^>]*?/?>", " ", stripped, flags=re.IGNORECASE)
        print(f"      [raw] void-tag strip {_t.perf_counter()-_ts:.2f}s", flush=True)

        _ts = _t.perf_counter()
        stripped = re.sub(
            r"<\s*(?:br|/p|/div|/section|/article|/li|/tr|/td|/th|/h[1-6]|hr)\b[^>]*>",
            "\n",
            stripped,
            flags=re.IGNORECASE,
        )
        print(f"      [raw] newline-tags {_t.perf_counter()-_ts:.2f}s", flush=True)

        _ts = _t.perf_counter()
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        print(f"      [raw] remaining-tags {_t.perf_counter()-_ts:.2f}s", flush=True)

        _ts = _t.perf_counter()
        stripped = unescape(stripped)
        result = self.clean_whitespace(stripped)
        print(f"      [raw] unescape+clean {_t.perf_counter()-_ts:.2f}s | total {_t.perf_counter()-_ts0:.2f}s", flush=True)
        return result

    # ── Node classification helpers ────────────────────────────────────────────

    @staticmethod
    def _normalize_html_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()

    @staticmethod
    def _html_marker_tokens(node) -> set[str]:
        class_names = " ".join(str(value) for value in node.get("class", []))
        node_id = str(node.get("id", ""))
        return {
            token
            for token in re.split(r"[^a-z0-9]+", f"{class_names} {node_id}".casefold())
            if token
        }

    def _is_html_noise_node(self, node) -> bool:
        marker_tokens = self._html_marker_tokens(node)
        node_name = (node.name or "").casefold()
        if node_name == "nav":
            return True
        return any(marker in marker_tokens for marker in self.HTML_NOISE_MARKERS)

    def _has_html_noise_ancestor(self, node) -> bool:
        for parent in node.parents:
            if not getattr(parent, "name", ""):
                continue
            if self._is_html_noise_node(parent):
                return True
        return False

    def _should_skip_html_container(self, node) -> bool:
        node_name = (node.name or "").lower()
        if node_name not in self.HTML_CONTAINER_TAGS:
            return False
        child_blocks = [
            child
            for child in node.find_all(list(self.HTML_BLOCK_TAGS), recursive=False)
            if not self._is_html_noise_node(child)
        ]
        return bool(child_blocks)

    def _classify_html_block_tag(self, node, text: str) -> str:
        node_name = (node.name or "").lower()
        if node_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return node_name
        if self._looks_like_html_heading_block(node=node, text=text):
            return "h3"
        return "p"

    def _looks_like_html_heading_block(self, node, text: str) -> bool:
        marker_tokens = self._html_marker_tokens(node)

        if (node.name or "").lower() in {"th", "dt"} and len(text) <= 200:
            return True

        has_heading_marker = any(marker in marker_tokens for marker in self.HTML_HEADING_MARKERS)
        has_body_marker = any(marker in marker_tokens for marker in self.HTML_BODY_MARKERS)

        if not has_heading_marker:
            return False
        if len(text) > 240:
            return False
        if has_body_marker and len(text) > 120:
            return False
        if text.endswith((".", "?", "!")) and len(text) > 80:
            return False
        return True

    @staticmethod
    def _html_body_chars(units: list[tuple[str, str]]) -> int:
        return sum(len(text) for tag, text in units if not tag.startswith("h"))

    @staticmethod
    def _has_substantive_body_units(units: list[tuple[str, str]]) -> bool:
        return any(text.strip() for tag, text in units if not tag.startswith("h"))

    # ── Document title selection ───────────────────────────────────────────────

    def _select_document_title(
        self,
        path: str,
        html_title: str,
        units: list[tuple[str, str]],
    ) -> str:
        file_title = self._clean_title_candidate(Path(path).stem.replace("_", " "))
        heading_title = ""

        for tag, text in units:
            if not tag.startswith("h"):
                continue
            heading_title = self._clean_title_candidate(text)
            if heading_title:
                break

        cleaned_html_title = self._clean_title_candidate(html_title)
        html_overlap = self._title_overlap_score(cleaned_html_title, file_title)
        heading_overlap = self._title_overlap_score(heading_title, file_title)

        if heading_title and not self._looks_like_bad_title(heading_title, file_title) and (
            self._looks_like_bad_title(cleaned_html_title, file_title) or heading_overlap > html_overlap
        ):
            return heading_title

        if cleaned_html_title and not self._looks_like_bad_title(cleaned_html_title, file_title):
            return cleaned_html_title

        if heading_title and not self._looks_like_bad_title(heading_title, file_title):
            return heading_title

        return file_title or cleaned_html_title or Path(path).stem
