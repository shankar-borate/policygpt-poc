import re
from pathlib import Path

from bs4 import BeautifulSoup

from policygpt.config import Config


class FileExtractor:
    def __init__(self, config: Config):
        self.config = config

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
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link"]):
            tag.decompose()

        html_title = ""
        if soup.title and soup.title.text.strip():
            html_title = self.clean_whitespace(soup.title.text)

        candidates = soup.find_all([
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "table", "tr", "td", "th",
        ])

        units: list[tuple[str, str]] = []
        for node in candidates:
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            units.append((node.name.lower(), text))

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
