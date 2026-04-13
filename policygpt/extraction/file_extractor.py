import re
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup

from policygpt.config import Config
from policygpt.extraction.ocr import TextractOCR


class FileExtractor:
    HTML_SEMANTIC_TAGS = (
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "table", "tr", "td", "th",
        "blockquote", "pre", "dd", "dt",
    )
    HTML_BLOCK_TAGS = (
        "article", "section", "main", "aside", "header", "footer", "nav",
        "div", "p", "li", "table", "tr", "td", "th", "blockquote", "pre",
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

    def __init__(self, config: Config):
        self.config = config
        self._ocr: TextractOCR | None = (
            TextractOCR(
                region=config.bedrock_region,
                min_confidence=config.ocr_min_confidence,
            )
            if config.ocr_enabled
            else None
        )

    @staticmethod
    def read_text_file(path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def clean_whitespace(text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract(self, path: str) -> tuple[str, list[tuple[str, str]]]:
        suffix = Path(path).suffix.lower()
        if suffix in {".html", ".htm"}:
            return self.extract_from_html(path)
        if suffix == ".txt":
            return self.extract_from_plain_text(path)
        if suffix == ".pdf":
            return self.extract_from_pdf(path)
        return Path(path).stem, []

    def extract_from_html(self, path: str) -> tuple[str, list[tuple[str, str]]]:
        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        raw_text = self._extract_text_from_raw_html(html)

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception:
                soup = None

        if soup is None:
            title = self._select_text_document_title(
                path=path,
                lines=[line.strip() for line in raw_text.split("\n") if line.strip()],
            )
            return title, self._group_units_into_sections(self._build_html_text_fallback_units(raw_text))

        # OCR pass — collect text from all <img> tags before they are removed.
        # Each image becomes a "p" unit appended after the normal content units
        # so it is indexed and searchable.  The full OCR text is stored in the
        # section so the entire block is shown when a match is retrieved.
        ocr_units: list[tuple[str, str]] = []
        if self._ocr is not None:
            html_dir = Path(path).resolve().parent
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src") or img_tag.get("data-src") or ""
                ocr_text = self._ocr.extract_from_src(src, html_dir)
                if ocr_text:
                    alt = self.clean_whitespace(img_tag.get("alt") or "")
                    label = f"[Image{': ' + alt if alt else ''}]"
                    ocr_units.append(("p", f"{label}\n{ocr_text}"))

        for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link"]):
            tag.decompose()

        html_title = ""
        if soup.title and soup.title.text.strip():
            html_title = self.clean_whitespace(soup.title.text)

        primary_units = self._extract_html_semantic_units(soup)
        fallback_units = self._extract_html_block_units(soup)
        body_root = soup.body or soup
        body_text = self.clean_whitespace(body_root.get_text("\n", strip=True))
        fallback_text = body_text if len(body_text) >= len(raw_text) else raw_text

        units = primary_units
        if self._should_prefer_html_block_units(primary_units, fallback_units, body_text):
            units = fallback_units

        if not self._has_substantive_body_units(units) and fallback_text:
            units = self._build_html_text_fallback_units(fallback_text)

        # Append OCR units after the main content so they form their own
        # section(s) and are fully shown when their text matches a query.
        units = units + ocr_units

        title = self._select_document_title(path=path, html_title=html_title, units=units)
        return title, self._group_units_into_sections(units)

    def extract_from_plain_text(self, path: str) -> tuple[str, list[tuple[str, str]]]:
        text = self.clean_whitespace(self.read_text_file(path))
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        doc_title = Path(path).stem

        if lines and len(lines[0]) < 150:
            doc_title = lines[0]

        units: list[tuple[str, str]] = []
        buffer: list[str] = []

        def flush_paragraph() -> None:
            nonlocal buffer
            if buffer:
                paragraph = " ".join(buffer).strip()
                if paragraph:
                    units.append(("p", paragraph))
                buffer = []

        for line in lines:
            is_heading = self._looks_like_heading_line(line)
            is_bullet = re.match(r"^[-*\u2022]\s+.+", line) or re.match(r"^\d+[\.\)]\s+.+", line)

            if is_heading:
                flush_paragraph()
                units.append(("h2", line))
            elif is_bullet:
                flush_paragraph()
                units.append(("li", line))
            else:
                buffer.append(line)

        flush_paragraph()
        return doc_title, self._group_units_into_sections(units)

    def extract_from_pdf(self, path: str) -> tuple[str, list[tuple[str, str]]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF ingestion requires the 'pypdf' package. Install dependencies from requirements.txt."
            ) from exc

        reader = PdfReader(path)
        metadata_title = ""
        if reader.metadata and reader.metadata.title:
            metadata_title = self.clean_whitespace(str(reader.metadata.title))

        page_texts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append(extracted)

        combined_text = self.clean_whitespace("\n\n".join(page_texts))
        if not combined_text:
            return Path(path).stem, []

        lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
        units = self._build_text_units(lines)
        title = self._select_text_document_title(
            path=path,
            lines=lines,
            metadata_title=metadata_title,
        )
        return title, self._group_units_into_sections(units)

    def _build_text_units(self, lines: list[str]) -> list[tuple[str, str]]:
        units: list[tuple[str, str]] = []
        buffer: list[str] = []

        def flush_paragraph() -> None:
            nonlocal buffer
            if buffer:
                paragraph = " ".join(buffer).strip()
                if paragraph:
                    units.append(("p", paragraph))
                buffer = []

        for line in lines:
            is_heading = self._looks_like_heading_line(line)
            is_bullet = re.match(r"^[-*\u2022]\s+.+", line) or re.match(r"^\d+[\.\)]\s+.+", line)

            if is_heading:
                flush_paragraph()
                units.append(("h2", line))
            elif is_bullet:
                flush_paragraph()
                units.append(("li", line))
            else:
                buffer.append(line)

        flush_paragraph()
        return units

    def _select_text_document_title(
        self,
        path: str,
        lines: list[str],
        metadata_title: str = "",
    ) -> str:
        file_title = self._clean_title_candidate(Path(path).stem.replace("_", " "))
        first_line_title = ""

        for line in lines[:8]:
            candidate = self._clean_title_candidate(line)
            if not candidate:
                continue
            if len(candidate) > 180:
                continue
            first_line_title = candidate
            break

        cleaned_metadata_title = self._clean_title_candidate(metadata_title)
        metadata_overlap = self._title_overlap_score(cleaned_metadata_title, file_title)
        line_overlap = self._title_overlap_score(first_line_title, file_title)

        if first_line_title and not self._looks_like_bad_title(first_line_title, file_title) and (
            self._looks_like_bad_title(cleaned_metadata_title, file_title)
            or line_overlap > metadata_overlap
        ):
            return first_line_title

        if cleaned_metadata_title and not self._looks_like_bad_title(cleaned_metadata_title, file_title):
            return cleaned_metadata_title

        if first_line_title and not self._looks_like_bad_title(first_line_title, file_title):
            return first_line_title

        return file_title or cleaned_metadata_title or Path(path).stem

    def _extract_html_semantic_units(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        units: list[tuple[str, str]] = []
        seen_texts: set[str] = set()
        emitted_table_ids: set[int] = set()

        for node in (soup.body or soup).find_all(list(self.HTML_SEMANTIC_TAGS)):
            if self._is_html_noise_node(node) or self._has_html_noise_ancestor(node):
                continue

            node_name = node.name.lower()

            if node_name == "table":
                table_text = self._table_to_plain_text(node)
                if not table_text:
                    continue
                normalized = self._normalize_html_text(table_text)
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    emitted_table_ids.add(id(node))
                    units.append(("table", table_text))
                continue

            if node_name in {"tr", "td", "th"}:
                ancestor_table = node.find_parent("table")
                if ancestor_table is not None and id(ancestor_table) in emitted_table_ids:
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

    @staticmethod
    def _table_to_plain_text(table_node) -> str:
        rows: list[str] = []
        for tr in table_node.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                for cell in tr.find_all(["th", "td"])
            ]
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

        stripped = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
        stripped = re.sub(
            r"<(script|style|noscript|svg|img|meta|link)\b.*?>.*?</\1\s*>",
            " ",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        stripped = re.sub(
            r"<\s*(br|/p|/div|/section|/article|/li|/tr|/td|/th|/h[1-6]|hr)\b[^>]*>",
            "\n",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = unescape(stripped)
        return self.clean_whitespace(stripped)

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

    def _group_units_into_sections(self, units: list[tuple[str, str]]) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_title = "Introduction"
        current_parts: list[str] = []

        def flush() -> None:
            nonlocal current_parts
            content = "\n".join(current_parts).strip()
            if content:
                sections.append((current_title, self.clean_whitespace(content)))
            current_parts = []

        for tag, text in units:
            if tag.startswith("h"):
                flush()
                current_title = text[:200]
            elif tag == "table":
                flush()
                sections.append((current_title, self.clean_whitespace(text)))
            else:
                current_parts.append(text)

        flush()

        if len(sections) == 1:
            title, text = sections[0]
            if not text:
                return []
            if title and title != "Introduction":
                if len(text) <= self.config.max_section_chars:
                    return [(title, text)]
                return self._split_large_text_into_synthetic_sections(text, title_prefix=title)

        if len(sections) == 0:
            all_text = "\n".join(text for _, text in sections) if sections else ""
            if not all_text:
                return []
            return self._split_large_text_into_synthetic_sections(all_text)

        final_sections: list[tuple[str, str]] = []
        for title, text in sections:
            if len(text) <= self.config.max_section_chars:
                final_sections.append((title, text))
            else:
                final_sections.extend(
                    self._split_large_text_into_synthetic_sections(text, title_prefix=title)
                )

        return final_sections

    def _split_large_text_into_synthetic_sections(
        self,
        text: str,
        title_prefix: str = "Section",
    ) -> list[tuple[str, str]]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            paragraphs = [text]

        sections: list[tuple[str, str]] = []
        current: list[str] = []
        current_len = 0
        part_no = 1

        for paragraph in paragraphs:
            paragraph_parts = self._split_oversized_text_block(paragraph, self.config.target_section_chars)
            for paragraph_part in paragraph_parts:
                paragraph_len = len(paragraph_part)
                too_big = current_len + paragraph_len > self.config.target_section_chars

                if current and too_big and current_len >= self.config.min_section_chars:
                    sections.append((f"{title_prefix} - Part {part_no}", "\n\n".join(current)))
                    part_no += 1
                    current = [paragraph_part]
                    current_len = paragraph_len
                else:
                    current.append(paragraph_part)
                    current_len += paragraph_len

        if current:
            sections.append((f"{title_prefix} - Part {part_no}", "\n\n".join(current)))

        return sections

    def _split_oversized_text_block(self, text: str, target_chars: int) -> list[str]:
        text = text.strip()
        if not text or len(text) <= target_chars:
            return [text] if text else []

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if len(sentences) > 1:
            return self._pack_text_segments(sentences, target_chars, separator=" ")

        words = text.split()
        if not words:
            return []
        return self._pack_text_segments(words, target_chars, separator=" ")

    @staticmethod
    def _pack_text_segments(segments: list[str], target_chars: int, separator: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for segment in segments:
            projected_len = current_len + len(segment) + (len(separator) if current else 0)
            if current and projected_len > target_chars:
                chunks.append(separator.join(current).strip())
                current = [segment]
                current_len = len(segment)
            else:
                current.append(segment)
                current_len = projected_len if current_len else len(segment)

        if current:
            chunks.append(separator.join(current).strip())

        return chunks

    @staticmethod
    def _looks_like_heading_line(line: str) -> bool:
        compact = line.strip()
        if not compact or len(compact) >= 180:
            return False

        if re.match(r"^(\d+(\.\d+)*[\)\.]?)\s+.+", compact):
            return True
        if re.match(r"^[A-Z][A-Z0-9 /&\-,]{4,}$", compact):
            return True
        if compact.endswith((".", "?", "!", ";")):
            return False

        words = compact.split()
        if not (1 <= len(words) <= 12):
            return False

        alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
        if not alpha_words:
            return False

        title_like_words = sum(
            1
            for word in alpha_words
            if word[:1].isupper() or word.upper() == word
        )
        return title_like_words / len(alpha_words) >= 0.75

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

    @staticmethod
    def _clean_title_candidate(text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        return compact[:200]

    def _title_overlap_score(self, candidate: str, file_title: str) -> float:
        candidate_tokens = self._tokenize_title(candidate)
        file_tokens = self._tokenize_title(file_title)
        if not candidate_tokens or not file_tokens:
            return 0.0
        return len(candidate_tokens & file_tokens) / len(file_tokens)

    @staticmethod
    def _tokenize_title(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) >= 3
        }

    def _looks_like_bad_title(self, title: str, file_title: str) -> bool:
        if not title:
            return True

        normalized = title.casefold()
        if normalized in {
            "company car scheme",
            "policy summary",
            "document",
            "untitled",
            "table of contents",
            "contents",
        }:
            return True

        return self._title_overlap_score(title, file_title) == 0.0
