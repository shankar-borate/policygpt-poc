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

        heading_pattern = re.compile(r"^(\d+(\.\d+)*[\)\.]?)\s+.+|^[A-Z][A-Z0-9 /&\-,]{4,}$")

        for line in lines:
            is_heading = bool(heading_pattern.match(line)) and len(line) < 180
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

        if len(sections) <= 1:
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
            paragraph_len = len(paragraph)
            too_big = current_len + paragraph_len > self.config.target_section_chars

            if current and too_big and current_len >= self.config.min_section_chars:
                sections.append((f"{title_prefix} - Part {part_no}", "\n\n".join(current)))
                part_no += 1
                current = [paragraph]
                current_len = paragraph_len
            else:
                current.append(paragraph)
                current_len += paragraph_len

        if current:
            sections.append((f"{title_prefix} - Part {part_no}", "\n\n".join(current)))

        return sections

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
