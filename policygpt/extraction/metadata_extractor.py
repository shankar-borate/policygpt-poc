import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from policygpt.extraction.taxonomy import (
    AUDIENCE_KEYWORDS,
    DOCUMENT_TYPE_KEYWORDS,
    SECTION_TYPE_KEYWORDS,
    DOMAIN_TOPIC_SYNONYMS,
    detect_matching_labels,
    humanize_term,
    is_informative_term,
    keywordize_text,
    normalize_text,
    unique_preserving_order,
)


VERSION_PATTERN = re.compile(r"\b(?:v(?:ersion)?\.?\s*|ver\.?\s*)(\d+(?:\.\d+)*)\b", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-])?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,-]+\d{2,4}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentMetadata:
    normalized_title: str
    canonical_title: str
    document_type: str
    version: str
    effective_date: str
    tags: list[str] = field(default_factory=list)
    audiences: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    title_terms: list[str] = field(default_factory=list)
    token_counts: dict[str, int] = field(default_factory=dict)
    token_length: int = 0


@dataclass(frozen=True)
class SectionMetadata:
    normalized_title: str
    section_type: str
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    title_terms: list[str] = field(default_factory=list)
    token_counts: dict[str, int] = field(default_factory=dict)
    token_length: int = 0


class MetadataExtractor:
    def extract_document_metadata(self, source_path: str, title: str, full_text: str) -> DocumentMetadata:
        normalized_title = normalize_text(title)
        canonical_title = self._canonicalize_title(title)
        document_type = self._detect_document_type(title, full_text)
        version = self._extract_version(f"{title}\n{full_text[:600]}")
        effective_date = self._extract_effective_date(f"{title}\n{full_text[:1000]}")
        tags = self._extract_document_tags(
            source_path=source_path,
            title=title,
            canonical_title=canonical_title,
            full_text=full_text,
        )
        audiences = self._extract_audiences(f"{title}\n{full_text[:2000]}")

        title_terms = unique_preserving_order(keywordize_text(title))
        keywords = self._extract_keywords(
            parts=[
                title,
                canonical_title,
                Path(source_path).stem.replace("_", " "),
                *tags,
                *audiences,
                full_text[:2500],
            ]
        )
        token_counts = Counter(self._build_search_terms(title, full_text, tags, audiences, canonical_title))
        token_length = sum(token_counts.values())

        return DocumentMetadata(
            normalized_title=normalized_title,
            canonical_title=canonical_title,
            document_type=document_type,
            version=version,
            effective_date=effective_date,
            tags=tags,
            audiences=audiences,
            keywords=keywords,
            title_terms=title_terms,
            token_counts=dict(token_counts),
            token_length=token_length,
        )

    def extract_section_metadata(self, doc_title: str, section_title: str, section_text: str) -> SectionMetadata:
        normalized_title = normalize_text(section_title)
        section_type = self._detect_section_type(section_title, section_text)
        tags = self._extract_section_tags(
            doc_title=doc_title,
            section_title=section_title,
            section_text=section_text,
        )
        title_terms = unique_preserving_order(keywordize_text(section_title))
        keywords = self._extract_keywords([section_title, *tags, section_text[:1600]])
        token_counts = Counter(self._build_search_terms(section_title, section_text, tags, [], doc_title))
        token_length = sum(token_counts.values())

        return SectionMetadata(
            normalized_title=normalized_title,
            section_type=section_type,
            tags=tags,
            keywords=keywords,
            title_terms=title_terms,
            token_counts=dict(token_counts),
            token_length=token_length,
        )

    def _detect_document_type(self, title: str, full_text: str) -> str:
        matches = detect_matching_labels(f"{title}\n{full_text[:1200]}", DOCUMENT_TYPE_KEYWORDS)
        return matches[0] if matches else "document"

    def _detect_section_type(self, section_title: str, section_text: str) -> str:
        matches = detect_matching_labels(f"{section_title}\n{section_text[:900]}", SECTION_TYPE_KEYWORDS)
        return matches[0] if matches else "general"

    def _extract_document_tags(
        self,
        source_path: str,
        title: str,
        canonical_title: str,
        full_text: str,
    ) -> list[str]:
        auto_tags = self._extract_auto_tags(
            weighted_parts=[
                (title, 4),
                (canonical_title, 3),
                (Path(source_path).stem.replace("_", " "), 2),
            ],
            limit=6,
        )
        if auto_tags:
            return auto_tags
        auto_tags = self._extract_auto_tags(
            weighted_parts=[
                (title, 3),
                (full_text[:2500], 1),
            ],
            limit=6,
        )
        if auto_tags:
            return auto_tags
        return detect_matching_labels(f"{title}\n{full_text[:2000]}", DOMAIN_TOPIC_SYNONYMS)

    def _extract_section_tags(self, doc_title: str, section_title: str, section_text: str) -> list[str]:
        auto_tags = self._extract_auto_tags(
            weighted_parts=[
                (section_title, 4),
                (doc_title, 2),
            ],
            limit=5,
        )
        if auto_tags:
            return auto_tags
        auto_tags = self._extract_auto_tags(
            weighted_parts=[
                (section_title, 3),
                (section_text[:1400], 1),
            ],
            limit=5,
        )
        if auto_tags:
            return auto_tags
        return detect_matching_labels(f"{doc_title}\n{section_title}\n{section_text[:1200]}", DOMAIN_TOPIC_SYNONYMS)

    @staticmethod
    def _extract_audiences(text: str) -> list[str]:
        normalized = normalize_text(text)
        audiences = [keyword for keyword in AUDIENCE_KEYWORDS if keyword in normalized]
        return unique_preserving_order(audiences)

    def _extract_keywords(self, parts: list[str], limit: int = 16) -> list[str]:
        counter: Counter[str] = Counter()
        for part in parts:
            for term in keywordize_text(part):
                if len(term) < 3 or not is_informative_term(term):
                    continue
                counter[term] += 1

        ordered = [term for term, _ in counter.most_common(limit * 2)]
        return unique_preserving_order(ordered[:limit])

    def _extract_auto_tags(self, weighted_parts: list[tuple[str, int]], limit: int) -> list[str]:
        counter: Counter[str] = Counter()
        for part, weight in weighted_parts:
            if not part:
                continue
            for term in keywordize_text(part):
                if not is_informative_term(term):
                    continue
                term_weight = float(weight)
                if "_" in term:
                    term_weight += 0.6
                counter[term] += term_weight

        ordered_terms = [
            term
            for term, _ in sorted(
                counter.items(),
                key=lambda item: (item[1], item[0].count("_"), len(item[0])),
                reverse=True,
            )
        ]

        tags: list[str] = []
        seen_labels: set[str] = set()
        for term in ordered_terms:
            label = humanize_term(term)
            if not label or label in seen_labels:
                continue
            if "_" not in term and any(label in existing.split() for existing in tags):
                continue
            seen_labels.add(label)
            tags.append(label)
            if len(tags) >= limit:
                break
        return tags

    @staticmethod
    def _canonicalize_title(title: str) -> str:
        lowered = normalize_text(title)
        lowered = VERSION_PATTERN.sub("", lowered)
        lowered = DATE_PATTERN.sub("", lowered)
        lowered = re.sub(r"\bmajor version\b", "", lowered)
        lowered = re.sub(r"\bfinal\b", "", lowered)
        lowered = re.sub(r"[_\-]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    @staticmethod
    def _extract_version(text: str) -> str:
        match = VERSION_PATTERN.search(text or "")
        return match.group(1) if match else ""

    @staticmethod
    def _extract_effective_date(text: str) -> str:
        match = DATE_PATTERN.search(text or "")
        return match.group(0).strip() if match else ""

    def _build_search_terms(
        self,
        title: str,
        full_text: str,
        tags: list[str],
        audiences: list[str],
        canonical_title: str,
    ) -> list[str]:
        pieces = [
            title,
            canonical_title,
            *tags,
            *audiences,
            full_text,
        ]
        terms: list[str] = []
        for piece in pieces:
            terms.extend(keywordize_text(piece))
        return unique_preserving_order(terms) if not full_text else terms
