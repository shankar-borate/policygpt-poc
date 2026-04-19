"""BaseExtractor — shared utilities for all file extraction strategies."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from policygpt.config import Config
from policygpt.models.extraction import ExtractedSection  # noqa: F401 — re-exported for subclasses


class BaseExtractor(ABC):
    """Abstract base for file extractors.

    Subclasses implement :meth:`extract` for a specific file type.
    Shared text-processing helpers live here so they are available to
    every concrete extractor without duplication.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    # ── Public interface ───────────────────────────────────────────────────────

    @abstractmethod
    def extract(self, path: str) -> tuple[str, list[ExtractedSection]]:
        """Return ``(document_title, sections)`` for *path*."""

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def read_text_file(path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def clean_whitespace(text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

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

    @staticmethod
    def _derive_section_title_from_content(text: str) -> str:
        """Derive a section title from content when the document has no heading tags."""
        first_line = text.split("\n")[0].strip()
        if not first_line:
            return "Introduction"
        if " | " in first_line:
            cells = [c.strip() for c in first_line.split(" | ") if c.strip()]
            candidate = cells[1] if len(cells) > 1 else cells[0]
            if candidate and len(candidate) <= 150:
                return candidate
        if len(first_line) <= 150:
            return first_line
        return "Introduction"

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

    # ── Text-splitting helpers (shared by PDF and plain-text extractors) ───────

    def _split_oversized_text_block(self, text: str, target_chars: int) -> list[str]:
        text = text.strip()
        if not text or len(text) <= target_chars:
            return [text] if text else []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 1:
            return self._pack_text_segments(sentences, target_chars, separator=" ")

        words = text.split()
        if not words:
            return []
        return self._pack_text_segments(words, target_chars, separator=" ")

    def _split_large_text_into_synthetic_sections(
        self,
        text: str,
        title_prefix: str = "Section",
    ) -> list[tuple[str, str]]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            paragraph_parts = self._split_oversized_text_block(paragraph, self.config.ingestion.target_section_chars)
            for paragraph_part in paragraph_parts:
                paragraph_len = len(paragraph_part)
                too_big = current_len + paragraph_len > self.config.ingestion.target_section_chars

                if current and too_big and current_len >= self.config.ingestion.min_section_chars:
                    chunks.append("\n\n".join(current))
                    current = [paragraph_part]
                    current_len = paragraph_len
                else:
                    current.append(paragraph_part)
                    current_len += paragraph_len

        if current:
            chunks.append("\n\n".join(current))

        if not chunks:
            return []

        # Single chunk — no suffix needed
        if len(chunks) == 1:
            return [(title_prefix, chunks[0])]

        # Multiple chunks — derive a unique title for each from its content
        return [
            (self._title_for_chunk(chunk, title_prefix, idx), chunk)
            for idx, chunk in enumerate(chunks, 1)
        ]

    def _title_for_chunk(self, text: str, title_prefix: str, part_no: int) -> str:
        """Build a title for a split chunk by extracting a content hint.

        Takes the first non-empty line of the chunk (clipped to ~60 chars at a
        word boundary) and appends it to *title_prefix* as a subtitle.  Falls
        back to ``"title_prefix - Part N"`` only when no distinctive hint can
        be found.
        """
        for line in text.split("\n"):
            hint = line.strip()
            if not hint:
                continue
            # Clip at a word boundary around 60 chars
            if len(hint) > 60:
                hint = hint[:60].rsplit(" ", 1)[0]
            hint = hint.rstrip(".,;:—")
            # Only use as subtitle if it genuinely differs from the parent title
            if hint and len(hint) >= 10 and hint.casefold() != title_prefix.casefold():
                return f"{title_prefix} — {hint}"
            break
        return f"{title_prefix} - Part {part_no}"

    def _group_units_into_sections(self, units: list[tuple[str, str]]) -> list[ExtractedSection]:
        """Group (tag, text) units into ExtractedSection objects."""
        sections: list[tuple[str, str]] = []
        table_texts: set[str] = set()
        current_title = "Introduction"
        current_parts: list[str] = []
        heading_seen = False

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
                heading_seen = True
            elif tag == "table":
                flush()
                clean_text = self.clean_whitespace(text)
                table_texts.add(clean_text)
                sections.append((current_title, clean_text))
            else:
                current_parts.append(text)

        flush()

        if not heading_seen and len(sections) > 1:
            sections = [
                (self._derive_section_title_from_content(text), text)
                for _, text in sections
            ]

        if len(sections) == 1:
            title, text = sections[0]
            if not text:
                return []
            if title and title != "Introduction":
                if len(text) <= self.config.ingestion.max_section_chars:
                    return [ExtractedSection(title=title, text=text)]
                return [
                    ExtractedSection(title=t, text=x)
                    for t, x in self._split_large_text_into_synthetic_sections(text, title_prefix=title)
                ]

        if len(sections) == 0:
            return []

        final_sections: list[ExtractedSection] = []
        for title, text in sections:
            if text in table_texts or len(text) <= self.config.ingestion.max_section_chars:
                final_sections.append(ExtractedSection(title=title, text=text))
            else:
                final_sections.extend(
                    ExtractedSection(title=t, text=x)
                    for t, x in self._split_large_text_into_synthetic_sections(text, title_prefix=title)
                )

        return final_sections

    def _build_text_units(self, lines: list[str]) -> list[tuple[str, str]]:
        """Convert a list of text lines into (tag, text) units."""
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

    # ── Title helpers (shared by PDF and plain-text extractors) ────────────────

    @staticmethod
    def _clean_title_candidate(text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        return compact[:200]

    @staticmethod
    def _tokenize_title(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) >= 3
        }

    def _title_overlap_score(self, candidate: str, file_title: str) -> float:
        candidate_tokens = self._tokenize_title(candidate)
        file_tokens = self._tokenize_title(file_title)
        if not candidate_tokens or not file_tokens:
            return 0.0
        return len(candidate_tokens & file_tokens) / len(file_tokens)

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
