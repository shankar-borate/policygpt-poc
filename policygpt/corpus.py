import re
import traceback
import uuid
from collections import Counter
from math import log
from pathlib import Path
from typing import Callable

import numpy as np

from policygpt.config import Config
from policygpt.models import DocumentRecord, SectionRecord, utc_now_iso
from policygpt.services.base import AIRequestTooLargeError, AIService
from policygpt.services.debug_logging import write_llm_debug_log_pair
from policygpt.services.file_extractor import FileExtractor
from policygpt.services.metadata_extractor import MetadataExtractor
from policygpt.services.query_analyzer import QueryAnalysis
from policygpt.services.redaction import Redactor
from policygpt.services.taxonomy import keywordize_text, normalize_text, tokenize_text, unique_preserving_order


ProgressCallback = Callable[[int, int, str | None, int, int], None]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(vector)
    if denominator == 0:
        return vector
    return vector / denominator


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.dot(matrix, query)


class DocumentCorpus:
    def __init__(
        self,
        config: Config,
        extractor: FileExtractor,
        ai: AIService,
        redactor: Redactor,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.ai = ai
        self.redactor = redactor
        self.metadata_extractor = MetadataExtractor()

        self.documents: dict[str, DocumentRecord] = {}
        self.sections: dict[str, SectionRecord] = {}
        self.doc_ids: list[str] = []
        self.doc_embedding_matrix: np.ndarray | None = None
        self.section_ids: list[str] = []
        self.section_embedding_matrix: np.ndarray | None = None
        self.doc_term_doc_freq: dict[str, int] = {}
        self.section_term_doc_freq: dict[str, int] = {}
        self.avg_doc_token_length = 0.0
        self.avg_section_token_length = 0.0
        self._active_ingestion_run_id: str | None = None

    TOPIC_ALIGNMENT_IGNORED_TOKENS: set[str] = {
        "policy",
        "policies",
        "document",
        "documents",
        "process",
        "processes",
        "procedure",
        "procedures",
        "checklist",
        "checklists",
        "guideline",
        "guidelines",
        "manual",
        "matrix",
        "form",
        "forms",
    }

    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        file_paths = self.list_supported_policy_files(folder_path)
        if not file_paths:
            self._write_ingestion_failure_log(
                source_path=folder_path,
                stage="scan",
                reason="No .html/.htm/.txt/.pdf files found in folder.",
            )
            raise FileNotFoundError(f"No .html/.htm/.txt/.pdf files found in folder: {folder_path}")

        total_files = len(file_paths)
        self._emit_progress(progress_callback, 0, total_files, "Scanning policy files")
        self._active_ingestion_run_id = str(uuid.uuid4())
        self._write_ingestion_run_header(folder_path, file_paths)

        try:
            for processed_files, path in enumerate(file_paths, start=1):
                file_name = Path(path).name
                self._emit_progress(
                    progress_callback,
                    processed_files - 1,
                    total_files,
                    f"{file_name} - reading file",
                )
                try:
                    status, detail = self.ingest_file(
                        path=path,
                        progress_callback=progress_callback,
                        processed_files=processed_files - 1,
                        total_files=total_files,
                    )
                    if status == "ingested":
                        self._emit_progress(
                            progress_callback,
                            processed_files,
                            total_files,
                            f"{file_name} - completed",
                        )
                    else:
                        self._append_ingestion_index_entry(
                            source_path=path,
                            status=status,
                            detail=detail,
                        )
                        self._write_ingestion_failure_log(
                            source_path=path,
                            stage="ingest_file",
                            reason=detail,
                        )
                        self._emit_progress(
                            progress_callback,
                            processed_files,
                            total_files,
                            f"{file_name} - {detail}",
                        )
                except Exception as exc:
                    self._append_ingestion_index_entry(
                        source_path=path,
                        status="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                    self._write_ingestion_failure_log(
                        source_path=path,
                        stage="ingest_file",
                        reason=f"{type(exc).__name__}: {exc}",
                        exc=exc,
                    )
                    self._emit_progress(
                        progress_callback,
                        processed_files,
                        total_files,
                        f"{file_name} - skipped after error: {self._format_error_for_progress(exc)}",
                    )
                    if self.config.debug:
                        print(f"Skipped document after error: {path}")
                        print(f"  {type(exc).__name__}: {exc}")

            self._emit_progress(progress_callback, total_files, total_files, "Building retrieval indexes")
            self.rebuild_indexes()
        finally:
            self._active_ingestion_run_id = None

    def ingest_file(
        self,
        path: str,
        progress_callback: ProgressCallback | None = None,
        processed_files: int = 0,
        total_files: int = 0,
    ) -> tuple[str, str]:
        path_obj = Path(path)
        file_name = path_obj.name
        extension = path_obj.suffix.lower()

        self._emit_progress(
            progress_callback,
            processed_files,
            total_files,
            f"{file_name} - extracting sections",
        )

        if extension not in {".html", ".htm", ".txt", ".pdf"}:
            return ("skipped", f"unsupported file extension: {extension or 'none'}")

        title, sections = self.extractor.extract(path)
        full_text = "\n\n".join(text for _, text in sections).strip()
        if not full_text:
            skip_reason = "no extractable text found" if extension == ".pdf" else "skipped empty document"
            self._emit_progress(
                progress_callback,
                processed_files,
                total_files,
                f"{file_name} - {skip_reason}",
            )
            return ("skipped", skip_reason)

        masked_title = self.redactor.mask_text(title)
        masked_full_text = self.redactor.mask_text(full_text)
        document_metadata = self.metadata_extractor.extract_document_metadata(path, title, full_text)

        self._emit_progress(
            progress_callback,
            processed_files,
            total_files,
            f"{file_name} - summarizing document",
        )
        document_summary = self._create_document_summary(
            masked_title=masked_title,
            masked_text=masked_full_text,
            progress_callback=progress_callback,
            processed_files=processed_files,
            total_files=total_files,
            file_name=file_name,
        )
        document_embedding = self._build_enriched_embedding(title, document_summary, full_text)

        document_id = str(uuid.uuid4())
        document = DocumentRecord(
            doc_id=document_id,
            title=title,
            source_path=path,
            raw_text=full_text,
            masked_text=masked_full_text,
            summary=document_summary,
            summary_embedding=document_embedding,
            sections=[],
            normalized_title=document_metadata.normalized_title,
            canonical_title=document_metadata.canonical_title,
            document_type=document_metadata.document_type,
            version=document_metadata.version,
            effective_date=document_metadata.effective_date,
            metadata_tags=document_metadata.tags,
            audiences=document_metadata.audiences,
            keywords=document_metadata.keywords,
            title_terms=document_metadata.title_terms,
            token_counts=document_metadata.token_counts,
            token_length=document_metadata.token_length,
        )

        valid_sections = [(section_title, section_text) for section_title, section_text in sections if section_text.strip()]
        total_sections = len(valid_sections)

        for index, (section_title, section_text) in enumerate(valid_sections, start=1):
            section_label = self._format_progress_label(section_title)
            self._emit_progress(
                progress_callback,
                processed_files,
                total_files,
                f"{file_name} - section {index}/{total_sections}: {section_label}",
            )

            masked_section_title = self.redactor.mask_text(section_title)
            masked_section_text = self.redactor.mask_text(section_text)
            section_metadata = self.metadata_extractor.extract_section_metadata(title, section_title, section_text)
            try:
                if self.config.skip_section_summary:
                    section_summary = self._build_fallback_section_summary(
                        section_title=masked_section_title,
                        section_text=masked_section_text,
                    )
                else:
                    section_summary = self._create_section_summary(
                        doc_title=masked_title,
                        section_title=masked_section_title,
                        section_text=masked_section_text,
                    )
                section_embedding = self._build_enriched_embedding(section_title, section_summary, section_text)
            except Exception as exc:
                self._emit_progress(
                    progress_callback,
                    processed_files,
                    total_files,
                    (
                        f"{file_name} - skipped section {index}/{total_sections}: "
                        f"{self._format_error_for_progress(exc)}"
                    ),
                )
                if self.config.debug:
                    print(f"Skipped section in {path}: {section_title}")
                    print(f"  {type(exc).__name__}: {exc}")
                self._write_ingestion_failure_log(
                    source_path=path,
                    stage="section_summary",
                    reason=f"Skipped section '{section_title}': {type(exc).__name__}: {exc}",
                    exc=exc,
                )
                continue

            section_id = str(uuid.uuid4())
            section = SectionRecord(
                section_id=section_id,
                title=section_title,
                raw_text=section_text,
                masked_text=masked_section_text,
                summary=section_summary,
                summary_embedding=section_embedding,
                source_path=path,
                doc_id=document_id,
                order_index=index - 1,
                normalized_title=section_metadata.normalized_title,
                section_type=section_metadata.section_type,
                metadata_tags=section_metadata.tags,
                keywords=section_metadata.keywords,
                title_terms=section_metadata.title_terms,
                token_counts=section_metadata.token_counts,
                token_length=section_metadata.token_length,
            )
            document.sections.append(section)
            self.sections[section_id] = section

        self.documents[document_id] = document
        self._write_ingestion_log(document)
        self._emit_progress(
            progress_callback,
            processed_files,
            total_files,
            f"{file_name} - finished with {len(document.sections)} sections",
        )

        if self.config.debug:
            print(f"Ingested: {path}")
            print(f"  Title: {title}")
            print(f"  Sections: {len(document.sections)}")
        return ("ingested", f"ingested with {len(document.sections)} sections")

    def rebuild_indexes(self) -> None:
        self.doc_ids = []
        doc_vectors: list[np.ndarray] = []
        doc_token_lengths: list[int] = []
        self.doc_term_doc_freq = {}
        for doc_id, document in self.documents.items():
            self.doc_ids.append(doc_id)
            doc_vectors.append(document.summary_embedding)
            doc_token_lengths.append(document.token_length)
            for term in document.token_counts:
                self.doc_term_doc_freq[term] = self.doc_term_doc_freq.get(term, 0) + 1

        self.section_ids = []
        section_vectors: list[np.ndarray] = []
        section_token_lengths: list[int] = []
        self.section_term_doc_freq = {}
        for section_id, section in self.sections.items():
            self.section_ids.append(section_id)
            section_vectors.append(section.summary_embedding)
            section_token_lengths.append(section.token_length)
            for term in section.token_counts:
                self.section_term_doc_freq[term] = self.section_term_doc_freq.get(term, 0) + 1

        self.doc_embedding_matrix = np.vstack(doc_vectors) if doc_vectors else None
        self.section_embedding_matrix = np.vstack(section_vectors) if section_vectors else None
        self.avg_doc_token_length = float(sum(doc_token_lengths) / len(doc_token_lengths)) if doc_token_lengths else 0.0
        self.avg_section_token_length = (
            float(sum(section_token_lengths) / len(section_token_lengths))
            if section_token_lengths
            else 0.0
        )

    def retrieve_top_docs(
        self,
        query_vec: np.ndarray,
        query_analysis: QueryAnalysis,
        preferred_doc_ids: list[str] | None = None,
    ) -> list[tuple[DocumentRecord, float]]:
        if self.doc_embedding_matrix is None or not self.doc_ids:
            return []

        semantic_scores = {
            doc_id: float(score)
            for doc_id, score in zip(self.doc_ids, cosine_similarity(query_vec, self.doc_embedding_matrix))
        }
        lexical_scores = {
            doc_id: self._bm25_score(
                query_terms=query_analysis.expanded_terms or query_analysis.focus_terms,
                token_counts=self.documents[doc_id].token_counts,
                token_length=self.documents[doc_id].token_length,
                doc_freq=self.doc_term_doc_freq,
                total_items=len(self.doc_ids),
                average_length=self.avg_doc_token_length,
            )
            for doc_id in self.doc_ids
        }
        title_scores = {
            doc_id: self._term_overlap_score(
                query_terms=query_analysis.expanded_terms or query_analysis.focus_terms,
                candidate_terms=self.documents[doc_id].title_terms + self.documents[doc_id].keywords,
            )
            for doc_id in self.doc_ids
        }
        metadata_scores = {
            doc_id: self._document_metadata_score(query_analysis, self.documents[doc_id])
            for doc_id in self.doc_ids
        }
        lookup_scores = {
            doc_id: self._document_lookup_score(query_analysis, self.documents[doc_id])
            for doc_id in self.doc_ids
        }

        semantic_norm = self._normalize_score_map(semantic_scores)
        lexical_norm = self._normalize_score_map(lexical_scores)
        title_norm = self._normalize_score_map(title_scores)
        preferred_set = set(preferred_doc_ids or [])
        rescored: list[tuple[str, float]] = []

        for doc_id in self.doc_ids:
            score = (
                self.config.doc_semantic_weight * semantic_norm.get(doc_id, 0.0)
                + self.config.doc_lexical_weight * lexical_norm.get(doc_id, 0.0)
                + self.config.doc_title_weight * title_norm.get(doc_id, 0.0)
                + self.config.doc_metadata_weight * metadata_scores.get(doc_id, 0.0)
            )
            if "document_lookup" in query_analysis.intents:
                score += 0.22 * lookup_scores.get(doc_id, 0.0)
            if doc_id in preferred_set:
                score += 0.08
            rescored.append((doc_id, score))

        rescored.sort(key=lambda item: item[1], reverse=True)
        doc_limit = self._doc_limit_for_query(query_analysis)
        results = [(self.documents[doc_id], score) for doc_id, score in rescored[:doc_limit]]

        if self.config.debug:
            print("\nTop documents:")
            for document, score in results:
                print(f"  {score:.4f} | {document.title}")

        return results

    def retrieve_top_sections(
        self,
        query_vec: np.ndarray,
        query_analysis: QueryAnalysis,
        top_docs: list[tuple[DocumentRecord, float]],
        preferred_section_ids: list[str] | None = None,
    ) -> list[tuple[SectionRecord, float]]:
        if self.section_embedding_matrix is None or not self.section_ids:
            return []

        preferred_set = set(preferred_section_ids or [])
        doc_score_lookup = {document.doc_id: score for document, score in top_docs}
        candidate_scores: dict[str, float] = {}

        per_doc_candidate_limit = self._per_doc_section_limit_for_query(query_analysis)
        for document, _ in top_docs:
            local_pairs = self._score_sections(
                query_vec=query_vec,
                query_analysis=query_analysis,
                section_ids=[section.section_id for section in document.sections],
                preferred_section_ids=preferred_set,
                doc_score_lookup=doc_score_lookup,
            )
            for section, score in local_pairs[:per_doc_candidate_limit]:
                candidate_scores[section.section_id] = max(candidate_scores.get(section.section_id, score), score)

        rerank_limit = self._rerank_limit_for_query(query_analysis)
        result_limit = self._section_result_limit_for_query(query_analysis)
        global_candidate_limit = max(
            rerank_limit * 2,
            result_limit * 4,
        )
        global_pairs = self._score_sections(
            query_vec=query_vec,
            query_analysis=query_analysis,
            section_ids=self.section_ids,
            preferred_section_ids=preferred_set,
            doc_score_lookup=doc_score_lookup,
        )
        for section, score in global_pairs[:global_candidate_limit]:
            candidate_scores[section.section_id] = max(candidate_scores.get(section.section_id, score), score)

        candidate_sections = [
            (self.sections[section_id], score)
            for section_id, score in candidate_scores.items()
            if section_id in self.sections
        ]
        candidate_sections.sort(key=lambda item: item[1], reverse=True)
        reranked_sections = self._rerank_sections(
            query_analysis=query_analysis,
            candidate_sections=candidate_sections[:rerank_limit],
        )
        reranked_sections.sort(key=lambda item: item[1], reverse=True)
        results = self._select_diverse_sections(
            query_analysis=query_analysis,
            scored_sections=reranked_sections,
            limit=result_limit,
        )

        if self.config.debug:
            print("\nTop sections:")
            for section, score in results:
                print(f"  {score:.4f} | {self.documents[section.doc_id].title} :: {section.title}")

        return results

    def _doc_limit_for_query(self, query_analysis: QueryAnalysis) -> int:
        if query_analysis.multi_doc_expected:
            return max(self.config.broad_top_docs, self.config.top_docs)
        if query_analysis.exact_match_expected:
            return max(1, min(self.config.exact_top_docs, self.config.top_docs))
        return self.config.top_docs

    def _per_doc_section_limit_for_query(self, query_analysis: QueryAnalysis) -> int:
        base_limit = max(self.config.top_sections_per_doc, self.config.rerank_section_candidates)
        if query_analysis.multi_doc_expected:
            return max(base_limit, self.config.broad_top_sections_per_doc)
        if query_analysis.exact_match_expected:
            return max(self.config.exact_top_sections_per_doc, self.config.exact_max_sections_to_llm)
        return base_limit

    def _rerank_limit_for_query(self, query_analysis: QueryAnalysis) -> int:
        if query_analysis.multi_doc_expected:
            return max(self.config.broad_rerank_section_candidates, self.config.rerank_section_candidates)
        if query_analysis.exact_match_expected:
            return max(
                self.config.exact_rerank_section_candidates,
                self.config.exact_top_sections_per_doc * 2,
                self.config.exact_max_sections_to_llm * 2,
            )
        return self.config.rerank_section_candidates

    def _section_result_limit_for_query(self, query_analysis: QueryAnalysis) -> int:
        if query_analysis.multi_doc_expected:
            return max(self.config.broad_max_sections_to_llm, self.config.max_sections_to_llm)
        if query_analysis.exact_match_expected:
            return max(1, min(self.config.exact_max_sections_to_llm, self.config.max_sections_to_llm))
        return self.config.max_sections_to_llm

    def _score_sections(
        self,
        query_vec: np.ndarray,
        query_analysis: QueryAnalysis,
        section_ids: list[str],
        preferred_section_ids: set[str],
        doc_score_lookup: dict[str, float],
    ) -> list[tuple[SectionRecord, float]]:
        valid_section_ids = [section_id for section_id in section_ids if section_id in self.sections]
        if not valid_section_ids:
            return []

        if valid_section_ids == self.section_ids and self.section_embedding_matrix is not None:
            section_vectors = self.section_embedding_matrix
        else:
            section_vectors = np.vstack([self.sections[section_id].summary_embedding for section_id in valid_section_ids])

        semantic_scores = {
            section_id: float(score)
            for section_id, score in zip(valid_section_ids, cosine_similarity(query_vec, section_vectors))
        }
        lexical_scores = {
            section_id: self._bm25_score(
                query_terms=query_analysis.expanded_terms or query_analysis.focus_terms,
                token_counts=self.sections[section_id].token_counts,
                token_length=self.sections[section_id].token_length,
                doc_freq=self.section_term_doc_freq,
                total_items=len(self.section_ids),
                average_length=self.avg_section_token_length,
            )
            for section_id in valid_section_ids
        }
        title_scores = {
            section_id: self._term_overlap_score(
                query_terms=query_analysis.expanded_terms or query_analysis.focus_terms,
                candidate_terms=self.sections[section_id].title_terms + self.sections[section_id].keywords,
            )
            for section_id in valid_section_ids
        }
        metadata_scores = {
            section_id: self._section_metadata_score(
                query_analysis,
                self.sections[section_id],
                self.documents[self.sections[section_id].doc_id],
            )
            for section_id in valid_section_ids
        }

        semantic_norm = self._normalize_score_map(semantic_scores)
        lexical_norm = self._normalize_score_map(lexical_scores)
        title_norm = self._normalize_score_map(title_scores)

        scored_sections: list[tuple[SectionRecord, float]] = []
        for section_id in valid_section_ids:
            section = self.sections[section_id]
            focus_match_score = self._focus_term_match_score(
                query_analysis.focus_terms,
                [section.title, section.summary, section.raw_text],
            )
            precise_match_score = self._precise_focus_match_score(
                query_analysis.focus_terms,
                [section.title, section.raw_text],
            )
            parent_weight = self.config.section_parent_weight
            if query_analysis.exact_match_expected:
                parent_weight *= self.config.exact_query_section_parent_weight_scale
            score = (
                self.config.section_semantic_weight * semantic_norm.get(section_id, 0.0)
                + self.config.section_lexical_weight * lexical_norm.get(section_id, 0.0)
                + parent_weight * doc_score_lookup.get(section.doc_id, 0.0)
                + self.config.section_title_weight * title_norm.get(section_id, 0.0)
                + self.config.section_metadata_weight * metadata_scores.get(section_id, 0.0)
                + (0.08 * focus_match_score)
                + (0.12 * precise_match_score)
            )
            if section_id in preferred_section_ids:
                score += 0.08
            scored_sections.append((section, score))

        scored_sections.sort(key=lambda item: item[1], reverse=True)
        return scored_sections

    def _select_diverse_sections(
        self,
        query_analysis: QueryAnalysis,
        scored_sections: list[tuple[SectionRecord, float]],
        limit: int,
    ) -> list[tuple[SectionRecord, float]]:
        if limit <= 0 or not scored_sections:
            return []

        if query_analysis.exact_match_expected:
            return scored_sections[:limit]

        first_per_doc: list[tuple[SectionRecord, float]] = []
        overflow: list[tuple[SectionRecord, float]] = []
        seen_doc_ids: set[str] = set()
        for section, score in scored_sections:
            if section.doc_id in seen_doc_ids:
                overflow.append((section, score))
                continue
            seen_doc_ids.add(section.doc_id)
            first_per_doc.append((section, score))

        return (first_per_doc + overflow)[:limit]

    def extract_answer_evidence_blocks(
        self,
        section: SectionRecord,
        query_analysis: QueryAnalysis,
        limit: int | None = None,
    ) -> list[str]:
        raw_text = section.raw_text.strip()
        if not raw_text:
            return []

        block_limit = limit or self._answer_evidence_block_limit_for_query(query_analysis)
        char_limit = self._answer_evidence_char_limit_for_query(query_analysis)
        if len(raw_text) <= min(self.config.small_section_full_text_chars, char_limit):
            return [raw_text]

        units = self._split_text_into_evidence_units(raw_text)
        if not units:
            return [self._truncate_text(raw_text, char_limit)]

        scored_units: list[tuple[int, float]] = []
        for index, unit in enumerate(units):
            score = self._score_evidence_unit(unit, section, query_analysis)
            if score > 0:
                scored_units.append((index, score))

        if not scored_units:
            return [self._truncate_text(raw_text, char_limit)]

        scored_units.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        selected_spans: list[tuple[int, int]] = []
        for index, _ in scored_units:
            span_start = max(0, index - self.config.evidence_neighboring_units)
            span_end = min(len(units) - 1, index + self.config.evidence_neighboring_units)
            span = (span_start, span_end)
            if any(not (span_end < existing_start or span_start > existing_end) for existing_start, existing_end in selected_spans):
                continue
            selected_spans.append(span)
            if len(selected_spans) >= block_limit:
                break

        if not selected_spans:
            return [self._truncate_text(raw_text, char_limit)]

        merged_spans = self._merge_evidence_spans(selected_spans)
        blocks: list[str] = []
        for start, end in merged_spans:
            block = "\n".join(units[start : end + 1]).strip()
            if not block:
                continue
            blocks.append(self._truncate_text(block, char_limit))

        return blocks or [self._truncate_text(raw_text, char_limit)]

    def precise_evidence_match_count(
        self,
        section: SectionRecord,
        query_analysis: QueryAnalysis,
    ) -> int:
        if not query_analysis.focus_terms:
            return 0
        score = self._precise_focus_match_score(
            query_analysis.focus_terms,
            [section.title, section.raw_text],
        )
        return max(0, int(score + 0.5))

    def _answer_evidence_block_limit_for_query(self, query_analysis: QueryAnalysis) -> int:
        if query_analysis.multi_doc_expected:
            return max(self.config.answer_evidence_block_limit_broad, 2)
        return max(self.config.answer_evidence_block_limit_exact, 1)

    def _answer_evidence_char_limit_for_query(self, query_analysis: QueryAnalysis) -> int:
        if query_analysis.multi_doc_expected:
            return max(self.config.broad_answer_evidence_char_limit, self.config.evidence_chunk_char_limit)
        if query_analysis.exact_match_expected:
            return max(self.config.exact_answer_evidence_char_limit, self.config.small_section_full_text_chars)
        return max(self.config.broad_answer_evidence_char_limit, self.config.evidence_chunk_char_limit)

    def _score_evidence_unit(
        self,
        unit_text: str,
        section: SectionRecord,
        query_analysis: QueryAnalysis,
    ) -> float:
        normalized_unit = normalize_text(unit_text)
        if not normalized_unit:
            return 0.0

        score = 0.0
        expanded_terms = unique_preserving_order(query_analysis.expanded_terms or query_analysis.focus_terms)
        for term in unique_preserving_order(query_analysis.focus_terms):
            phrase = term.replace("_", " ").strip()
            if phrase and phrase in normalized_unit:
                score += 2.0 if "_" in term else 0.8
        for term in expanded_terms:
            if term in query_analysis.focus_terms:
                continue
            phrase = term.replace("_", " ").strip()
            if phrase and phrase in normalized_unit:
                score += 1.2 if "_" in term else 0.35

        score += 0.9 * self._precise_focus_match_score(query_analysis.focus_terms, [unit_text])
        if section.section_type in query_analysis.expected_section_types:
            score += 0.35
        score += 0.45 * self.topic_alignment_score(query_analysis.topic_hints, section.metadata_tags)
        return score

    def _split_text_into_evidence_units(self, text: str) -> list[str]:
        normalized_text = text.replace("\r", "\n").strip()
        if not normalized_text:
            return []

        paragraph_chunks = [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n", normalized_text)
            if chunk.strip()
        ]

        units: list[str] = []
        for chunk in paragraph_chunks:
            lines = [line.strip() for line in chunk.split("\n") if line.strip()]
            if len(lines) >= 2 and all(len(line) <= 180 for line in lines):
                units.extend(lines)
                continue

            bullet_like_lines = [
                line
                for line in lines
                if re.match(r"^[-*\u2022]|\d+[\.\)]\s+", line)
            ]
            if bullet_like_lines:
                units.extend(lines)
                continue

            units.append(chunk)

        if len(units) > 1:
            return units

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized_text)
            if sentence.strip()
        ]
        if len(sentences) > 1:
            return sentences

        lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]
        return lines if len(lines) > 1 else [normalized_text]

    @staticmethod
    def _merge_evidence_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not spans:
            return []

        ordered = sorted(spans, key=lambda item: item[0])
        merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
        for start, end in ordered[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def extract_evidence_snippets(
        self,
        section: SectionRecord,
        query_analysis: QueryAnalysis,
        limit: int | None = None,
    ) -> list[str]:
        snippet_limit = limit or self.config.max_evidence_snippets_per_section
        evidence_blocks = self.extract_answer_evidence_blocks(
            section,
            query_analysis,
            limit=snippet_limit,
        )
        if not evidence_blocks:
            return []
        return [
            self._truncate_text(block, self.config.evidence_snippet_char_limit)
            for block in evidence_blocks[:snippet_limit]
        ]

    def _rerank_sections(
        self,
        query_analysis: QueryAnalysis,
        candidate_sections: list[tuple[SectionRecord, float]],
    ) -> list[tuple[SectionRecord, float]]:
        reranked: list[tuple[SectionRecord, float]] = []
        for section, base_score in candidate_sections:
            document = self.documents[section.doc_id]
            snippets = self.extract_evidence_snippets(section, query_analysis, limit=2)
            snippet_terms = keywordize_text(" ".join(snippets))
            snippet_overlap = self._term_overlap_score(
                query_terms=query_analysis.expanded_terms or query_analysis.focus_terms,
                candidate_terms=snippet_terms,
            )
            focus_match_score = self._focus_term_match_score(
                query_analysis.focus_terms,
                [section.title, section.summary, " ".join(snippets), section.raw_text],
            )
            precise_match_score = self._precise_focus_match_score(
                query_analysis.focus_terms,
                [section.title, " ".join(snippets), section.raw_text],
            )
            section_type_boost = 1.0 if section.section_type in query_analysis.expected_section_types else 0.0
            tag_boost = self.topic_alignment_score(query_analysis.topic_hints, section.metadata_tags)
            title_alignment = self._term_overlap_score(
                query_terms=query_analysis.focus_terms + query_analysis.topic_hints,
                candidate_terms=section.title_terms + document.title_terms,
            )
            reranked_score = (
                (0.72 * base_score)
                + (0.14 * snippet_overlap)
                + (0.08 * section_type_boost)
                + (0.04 * tag_boost)
                + (0.02 * title_alignment)
                + (0.06 * focus_match_score)
                + (0.08 * precise_match_score)
            )
            reranked.append((section, reranked_score))
        return reranked

    def _document_metadata_score(self, query_analysis: QueryAnalysis, document: DocumentRecord) -> float:
        score = 0.0
        score += 0.55 * self.topic_alignment_score(query_analysis.topic_hints, document.metadata_tags)
        if set(document.audiences).intersection(query_analysis.focus_terms):
            score += 0.15
        if query_analysis.intents:
            intent_set = set(query_analysis.intents)
            if "approval" in intent_set and document.document_type in {"policy", "matrix"}:
                score += 0.15
            if intent_set.intersection({"checklist", "process"}) and document.document_type in {"policy", "process", "checklist"}:
                score += 0.15
        return min(score, 1.0)

    def _section_metadata_score(
        self,
        query_analysis: QueryAnalysis,
        section: SectionRecord,
        document: DocumentRecord,
    ) -> float:
        score = 0.0
        if section.section_type in query_analysis.expected_section_types:
            score += 0.45
        score += 0.25 * self.topic_alignment_score(query_analysis.topic_hints, section.metadata_tags)
        score += 0.15 * self.topic_alignment_score(query_analysis.topic_hints, document.metadata_tags)
        if set(document.audiences).intersection(query_analysis.focus_terms):
            score += 0.15
        return min(score, 1.0)

    def document_lookup_score(self, query_analysis: QueryAnalysis, document: DocumentRecord) -> float:
        return self._document_lookup_score(query_analysis, document)

    def _document_lookup_score(self, query_analysis: QueryAnalysis, document: DocumentRecord) -> float:
        if "document_lookup" not in query_analysis.intents:
            return 0.0

        query_text = normalize_text(query_analysis.original_question)
        title_basis = document.canonical_title or document.title
        normalized_title = normalize_text(title_basis)
        title_tokens = {
            token
            for token in tokenize_text(title_basis)
            if token not in {"policy", "policies", "document", "documents", "process", "processes"}
        }
        if not normalized_title and not title_tokens:
            return 0.0

        phrase_hit = 1.0 if normalized_title and normalized_title in query_text else 0.0
        if not title_tokens:
            return phrase_hit

        query_tokens = set(tokenize_text(query_analysis.original_question))
        token_coverage = len(query_tokens.intersection(title_tokens)) / len(title_tokens)
        term_overlap = self._term_overlap_score(
            query_terms=query_analysis.expanded_terms or query_analysis.focus_terms,
            candidate_terms=document.title_terms + document.keywords,
        )
        return min(1.0, (0.5 * phrase_hit) + (0.35 * token_coverage) + (0.15 * term_overlap))

    def topic_alignment_score(self, topic_hints: list[str], metadata_tags: list[str]) -> float:
        if not topic_hints or not metadata_tags:
            return 0.0

        best_score = 0.0
        for topic in topic_hints:
            normalized_topic = normalize_text(topic)
            topic_tokens = self._topic_alignment_tokens(topic)
            for tag in metadata_tags:
                normalized_tag = normalize_text(tag)
                tag_tokens = self._topic_alignment_tokens(tag)

                score = 0.0
                if normalized_topic and normalized_tag and normalized_topic == normalized_tag:
                    score = 1.0
                elif normalized_topic and normalized_tag and (
                    normalized_topic in normalized_tag or normalized_tag in normalized_topic
                ):
                    score = 0.82
                elif topic_tokens and tag_tokens:
                    overlap = len(topic_tokens.intersection(tag_tokens))
                    if overlap > 0:
                        score = overlap / max(len(topic_tokens), len(tag_tokens))
                        if overlap == min(len(topic_tokens), len(tag_tokens)):
                            score = max(score, 0.72)

                if score > best_score:
                    best_score = score
        return best_score

    @staticmethod
    def _focus_term_match_score(focus_terms: list[str], texts: list[str]) -> float:
        combined_text = normalize_text(" ".join(text for text in texts if text))
        if not combined_text:
            return 0.0

        score = 0.0
        for term in unique_preserving_order(focus_terms):
            phrase = term.replace("_", " ").strip()
            if not phrase or phrase not in combined_text:
                continue
            score += 1.0 if "_" in term else 0.45

        return min(score, 2.0)

    @staticmethod
    def _precise_focus_match_score(focus_terms: list[str], texts: list[str]) -> float:
        combined_text = normalize_text(" ".join(text for text in texts if text))
        if not combined_text:
            return 0.0

        score = 0.0
        for term in unique_preserving_order(focus_terms):
            phrase = term.replace("_", " ").strip()
            if not phrase or phrase not in combined_text:
                continue
            has_numeric_part = any(part.isdigit() for part in term.split("_"))
            if "_" in term or has_numeric_part:
                score += 1.0
            elif len(term) >= 3:
                score += 0.5

        return min(score, 3.0)

    @classmethod
    def _topic_alignment_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in tokenize_text(text)
            if token not in cls.TOPIC_ALIGNMENT_IGNORED_TOKENS
        }

    def _bm25_score(
        self,
        query_terms: list[str],
        token_counts: dict[str, int],
        token_length: int,
        doc_freq: dict[str, int],
        total_items: int,
        average_length: float,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        if not query_terms or not token_counts or total_items <= 0:
            return 0.0

        average_length = average_length or max(float(token_length), 1.0)
        score = 0.0
        for term in unique_preserving_order(query_terms):
            frequency = token_counts.get(term, 0)
            if frequency <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = log(1 + ((total_items - df + 0.5) / (df + 0.5))) if df >= 0 else 0.0
            denominator = frequency + k1 * (1 - b + b * (token_length / average_length))
            score += idf * ((frequency * (k1 + 1)) / denominator)
        return score

    @staticmethod
    def _term_overlap_score(query_terms: list[str], candidate_terms: list[str]) -> float:
        if not query_terms or not candidate_terms:
            return 0.0
        candidate_set = set(candidate_terms)
        if not candidate_set:
            return 0.0
        query_set = set(unique_preserving_order(query_terms))
        if not query_set:
            return 0.0
        return len(query_set.intersection(candidate_set)) / len(query_set)

    @staticmethod
    def _normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        min_value = min(values)
        max_value = max(values)
        if abs(max_value - min_value) < 1e-9:
            normalized_value = 1.0 if max_value > 0 else 0.0
            return {key: normalized_value for key in score_map}
        return {
            key: (value - min_value) / (max_value - min_value)
            for key, value in score_map.items()
        }

    def list_supported_policy_files(self, folder_path: str) -> list[str]:
        folder = Path(folder_path)
        file_paths: list[str] = []
        for pattern in self.config.supported_file_patterns:
            file_paths.extend(str(path) for path in folder.glob(pattern))

        filtered_paths: list[str] = []
        for file_path in sorted(file_paths):
            file_name = Path(file_path).name.lower()
            if any(part in file_name for part in self.config.excluded_file_name_parts):
                continue
            filtered_paths.append(file_path)
        return filtered_paths

    def _create_document_summary(
        self,
        masked_title: str,
        masked_text: str,
        progress_callback: ProgressCallback | None = None,
        processed_files: int = 0,
        total_files: int = 0,
        file_name: str = "",
    ) -> str:
        # Fix #5: scale summary output budget with document size so large
        # documents get richer summaries visible to retrieval.
        doc_tokens = self._estimate_tokens(masked_text)
        scaled_max_output = self._scaled_summary_budget(doc_tokens)

        input_budget = self.config.doc_summary_input_token_budget
        if doc_tokens <= input_budget:
            try:
                return self._summarize_document_text(masked_title, masked_text, max_output_tokens=scaled_max_output)
            except AIRequestTooLargeError:
                pass

        chunks = self._split_text_for_summary(masked_text, input_budget)
        if len(chunks) <= 1:
            smaller_budget = max(self.config.min_recursive_summary_token_budget, input_budget // 2)
            if smaller_budget < input_budget:
                chunks = self._split_text_for_summary(masked_text, smaller_budget)
                input_budget = smaller_budget

        if len(chunks) <= 1:
            raise RuntimeError("Document is too large to summarize safely within the configured token budget.")

        partial_summaries: list[str] = []
        for index, chunk_text in enumerate(chunks, start=1):
            self._emit_progress(
                progress_callback,
                processed_files,
                total_files,
                f"{file_name} - summarizing document chunk {index}/{len(chunks)}",
            )
            partial_summaries.append(
                self._summarize_document_chunk_with_fallback(
                    masked_title=masked_title,
                    chunk_text=chunk_text,
                    chunk_label=f"{index}/{len(chunks)}",
                    token_budget=input_budget,
                )
            )

        reduced_summary_input = self._reduce_summary_groups(
            summaries=partial_summaries,
            token_budget=self.config.doc_summary_combine_token_budget,
            combine_batch=lambda summary_batch: self._combine_document_summaries(masked_title, summary_batch),
            progress_callback=progress_callback,
            processed_files=processed_files,
            total_files=total_files,
            progress_prefix=f"{file_name} - combining document summaries",
        )

        self._emit_progress(
            progress_callback,
            processed_files,
            total_files,
            f"{file_name} - finalizing document summary",
        )
        try:
            return self._finalize_document_summary(masked_title, reduced_summary_input)
        except AIRequestTooLargeError:
            return reduced_summary_input

    def _summarize_document_text(self, masked_title: str, masked_text: str, max_output_tokens: int | None = None) -> str:
        system_prompt = (
            "You summarize enterprise documents for retrieval. "
            "Return a compact retrieval-oriented summary. "
            "Focus on purpose, scope, major topics, rules/processes, exceptions, and key entities. "
            "Do not add facts."
        )
        user_prompt = (
            f"Document title:\n{masked_title}\n\n"
            f"Document text:\n{masked_text}\n\n"
            "Return a concise summary suitable for semantic retrieval."
        )
        return self._llm_text_with_debug_log(
            purpose="document_summary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens or self.config.doc_summary_max_output_tokens,
        )

    def _summarize_document_chunk_with_fallback(
        self,
        masked_title: str,
        chunk_text: str,
        chunk_label: str,
        token_budget: int,
    ) -> str:
        effective_budget = max(token_budget, self.config.min_recursive_summary_token_budget)
        if self._estimate_tokens(chunk_text) <= effective_budget:
            try:
                return self._summarize_document_chunk(
                    masked_title=masked_title,
                    chunk_text=chunk_text,
                    chunk_label=chunk_label,
                )
            except AIRequestTooLargeError:
                if effective_budget <= self.config.min_recursive_summary_token_budget:
                    raise

        smaller_budget = max(self.config.min_recursive_summary_token_budget, effective_budget // 2)
        if smaller_budget >= effective_budget:
            raise RuntimeError("Document chunk is too large to summarize safely within the configured token budget.")

        smaller_chunks = self._split_text_for_summary(chunk_text, smaller_budget)
        if len(smaller_chunks) <= 1:
            smaller_chunks = self._force_split_text(chunk_text)
        if len(smaller_chunks) <= 1:
            raise RuntimeError("Document chunk could not be split further after a token-limit failure.")

        nested_summaries: list[str] = []
        for nested_index, smaller_chunk in enumerate(smaller_chunks, start=1):
            nested_summaries.append(
                self._summarize_document_chunk_with_fallback(
                    masked_title=masked_title,
                    chunk_text=smaller_chunk,
                    chunk_label=f"{chunk_label}.{nested_index}/{len(smaller_chunks)}",
                    token_budget=smaller_budget,
                )
            )

        return self._reduce_summary_groups(
            summaries=nested_summaries,
            token_budget=min(self.config.doc_summary_combine_token_budget, smaller_budget),
            combine_batch=lambda summary_batch: self._combine_document_summaries(masked_title, summary_batch),
        )

    def _summarize_document_chunk(
        self,
        masked_title: str,
        chunk_text: str,
        chunk_label: str,
    ) -> str:
        system_prompt = (
            "You summarize one chunk of a longer enterprise document for retrieval. "
            "Use only the provided chunk. "
            "Capture the chunk's main topics, rules, process steps, eligibility, approvals, exceptions, "
            "thresholds, dates, and named entities when present. "
            "Do not invent facts."
        )
        user_prompt = (
            f"Document title:\n{masked_title}\n\n"
            f"Chunk:\n{chunk_label}\n\n"
            f"Chunk text:\n{chunk_text}\n\n"
            "Return a concise chunk summary suitable for later merging."
        )
        return self._llm_text_with_debug_log(
            purpose="document_chunk_summary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.doc_summary_chunk_max_output_tokens,
        )

    def _combine_document_summaries(self, masked_title: str, summary_batch: str) -> str:
        system_prompt = (
            "You are combining multiple partial summaries of the same enterprise document for retrieval. "
            "Merge overlap, preserve concrete policy details, and keep the result compact. "
            "Retain the important rules, approvals, thresholds, dates, exceptions, and actors when present. "
            "Do not invent facts."
        )
        user_prompt = (
            f"Document title:\n{masked_title}\n\n"
            f"Partial summaries:\n{summary_batch}\n\n"
            "Return one merged summary that can be used for another reduction pass if needed."
        )
        return self._llm_text_with_debug_log(
            purpose="document_summary_combine",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.doc_summary_chunk_max_output_tokens,
        )

    def _finalize_document_summary(self, masked_title: str, summary_input: str) -> str:
        system_prompt = (
            "You summarize enterprise documents for retrieval. "
            "Return a compact retrieval-oriented summary. "
            "Focus on purpose, scope, major topics, rules/processes, exceptions, and key entities. "
            "Do not add facts."
        )
        user_prompt = (
            f"Document title:\n{masked_title}\n\n"
            f"Collected document notes:\n{summary_input}\n\n"
            "Return a concise summary suitable for semantic retrieval."
        )
        return self._llm_text_with_debug_log(
            purpose="document_summary_finalize",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.doc_summary_max_output_tokens,
        )

    def _create_section_summary(
        self,
        doc_title: str,
        section_title: str,
        section_text: str,
        token_budget: int | None = None,
    ) -> str:
        effective_budget = max(
            token_budget or self.config.section_summary_input_token_budget,
            self.config.min_recursive_summary_token_budget,
        )
        if self._estimate_tokens(section_text) <= effective_budget:
            try:
                return self._summarize_section_text(doc_title, section_title, section_text)
            except AIRequestTooLargeError:
                if effective_budget <= self.config.min_recursive_summary_token_budget:
                    raise

        smaller_budget = max(self.config.min_recursive_summary_token_budget, effective_budget // 2)
        if smaller_budget >= effective_budget:
            raise RuntimeError("Section is too large to summarize safely within the configured token budget.")

        section_chunks = self._split_text_for_summary(section_text, smaller_budget)
        if len(section_chunks) <= 1:
            section_chunks = self._force_split_text(section_text)
        if len(section_chunks) <= 1:
            raise RuntimeError("Section could not be split further after a token-limit failure.")

        chunk_summaries: list[str] = []
        for index, chunk_text in enumerate(section_chunks, start=1):
            chunk_summaries.append(
                self._create_section_summary(
                    doc_title=doc_title,
                    section_title=f"{section_title} (part {index}/{len(section_chunks)})",
                    section_text=chunk_text,
                    token_budget=smaller_budget,
                )
            )

        reduced_summary_input = self._reduce_summary_groups(
            summaries=chunk_summaries,
            token_budget=min(self.config.doc_summary_combine_token_budget, smaller_budget),
            combine_batch=lambda summary_batch: self._combine_section_summaries(
                doc_title,
                section_title,
                summary_batch,
            ),
        )

        try:
            return self._finalize_section_summary(doc_title, section_title, reduced_summary_input)
        except AIRequestTooLargeError:
            return reduced_summary_input

    def _summarize_section_text(self, doc_title: str, section_title: str, section_text: str) -> str:
        system_prompt = (
            "You summarize a section for retrieval. "
            "Return a concise summary that helps answer questions later. "
            "Capture the key topic, rules, eligibility, process, exceptions, or details present. "
            "Do not invent facts."
        )
        user_prompt = (
            f"Document title:\n{doc_title}\n\n"
            f"Section title:\n{section_title}\n\n"
            f"Section text:\n{section_text}\n\n"
            "Return a concise section summary suitable for semantic retrieval."
        )
        return self._llm_text_with_debug_log(
            purpose="section_summary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.section_summary_max_output_tokens,
        )

    def _combine_section_summaries(self, doc_title: str, section_title: str, summary_batch: str) -> str:
        system_prompt = (
            "You are combining multiple partial summaries of the same document section for retrieval. "
            "Merge overlap, keep policy details precise, and preserve approvals, eligibility, process steps, "
            "exceptions, thresholds, and dates when present. "
            "Do not invent facts."
        )
        user_prompt = (
            f"Document title:\n{doc_title}\n\n"
            f"Section title:\n{section_title}\n\n"
            f"Partial section summaries:\n{summary_batch}\n\n"
            "Return one merged section summary suitable for retrieval."
        )
        return self._llm_text_with_debug_log(
            purpose="section_summary_combine",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.section_summary_max_output_tokens,
        )

    def _finalize_section_summary(self, doc_title: str, section_title: str, summary_input: str) -> str:
        system_prompt = (
            "You summarize a section for retrieval. "
            "Return a concise summary that helps answer questions later. "
            "Capture the key topic, rules, eligibility, process, exceptions, or details present. "
            "Do not invent facts."
        )
        user_prompt = (
            f"Document title:\n{doc_title}\n\n"
            f"Section title:\n{section_title}\n\n"
            f"Collected section notes:\n{summary_input}\n\n"
            "Return a concise section summary suitable for semantic retrieval."
        )
        return self._llm_text_with_debug_log(
            purpose="section_summary_finalize",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.section_summary_max_output_tokens,
        )

    def _embed_one(self, text: str) -> np.ndarray:
        return l2_normalize(self.ai.embed_texts([text])[0])

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def _build_enriched_embedding(self, title: str, summary: str, raw_text: str) -> np.ndarray:
        """Embed title + unmasked summary + raw text excerpt so the vector
        captures original terminology users will search with, instead of
        redacted placeholders or LLM-rephrased terms."""
        unmasked_summary = self.redactor.unmask_text(summary)
        raw_excerpt = raw_text[:self.config.embedding_raw_excerpt_chars].strip() if raw_text else ""
        combined = f"{title}\n{unmasked_summary}\n{raw_excerpt}".strip()
        return self._embed_one(combined)

    def _build_fallback_section_summary(self, section_title: str, section_text: str) -> str:
        compact_text = re.sub(r"\s+", " ", (section_text or "").strip())
        compact_title = re.sub(r"\s+", " ", (section_title or "").strip())
        char_limit = max(120, self.config.evidence_snippet_char_limit)
        if compact_text:
            summary_body = self._truncate_text(compact_text, char_limit)
            if compact_title and not summary_body.casefold().startswith(compact_title.casefold()):
                return f"{compact_title}: {summary_body}"
            return summary_body
        return compact_title or "(empty)"

    def _emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        processed_files: int,
        total_files: int,
        current_step: str | None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            processed_files,
            total_files,
            current_step,
            len(self.documents),
            len(self.sections),
        )

    @staticmethod
    def _format_progress_label(text: str, limit: int = 72) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return "Untitled section"
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _reduce_summary_groups(
        self,
        summaries: list[str],
        token_budget: int,
        combine_batch,
        progress_callback: ProgressCallback | None = None,
        processed_files: int = 0,
        total_files: int = 0,
        progress_prefix: str = "",
    ) -> str:
        working = [summary.strip() for summary in summaries if summary and summary.strip()]
        if not working:
            return ""

        effective_budget = max(token_budget, self.config.min_recursive_summary_token_budget)
        round_no = 1

        while len(working) > 1:
            formatted = self._format_summary_chunks(working)
            if self._estimate_tokens(formatted) <= effective_budget:
                try:
                    return combine_batch(formatted)
                except AIRequestTooLargeError:
                    pass

            batch_budget = max(self.config.min_recursive_summary_token_budget, effective_budget // 2)
            summary_batches = self._split_summary_batches(working, batch_budget)
            if len(summary_batches) <= 1 and len(working) > 1:
                summary_batches = [[summary] for summary in working]

            reduced_summaries: list[str] = []
            for batch_index, summary_batch in enumerate(summary_batches, start=1):
                if len(summary_batch) == 1:
                    reduced_summaries.append(summary_batch[0])
                    continue

                if progress_callback is not None and progress_prefix:
                    self._emit_progress(
                        progress_callback,
                        processed_files,
                        total_files,
                        f"{progress_prefix} round {round_no}, batch {batch_index}/{len(summary_batches)}",
                    )

                batch_text = self._format_summary_chunks(summary_batch)
                try:
                    reduced_summaries.append(combine_batch(batch_text))
                except AIRequestTooLargeError:
                    next_budget = max(self.config.min_recursive_summary_token_budget, batch_budget // 2)
                    if next_budget >= batch_budget:
                        reduced_summaries.extend(summary_batch)
                    else:
                        reduced_summaries.append(
                            self._reduce_summary_groups(
                                summaries=summary_batch,
                                token_budget=next_budget,
                                combine_batch=combine_batch,
                                progress_callback=progress_callback,
                                processed_files=processed_files,
                                total_files=total_files,
                                progress_prefix=progress_prefix,
                            )
                        )

            if reduced_summaries == working:
                return self._format_summary_chunks(working)

            working = [summary.strip() for summary in reduced_summaries if summary and summary.strip()]
            effective_budget = batch_budget
            round_no += 1

        return working[0]

    def _split_summary_batches(self, summaries: list[str], target_tokens: int) -> list[list[str]]:
        target_tokens = max(target_tokens, 1)
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for summary in summaries:
            entry_text = f"Partial summary:\n{summary}"
            entry_tokens = self._estimate_tokens(entry_text)
            projected_tokens = current_tokens + entry_tokens + (12 if current_batch else 0)
            if current_batch and projected_tokens > target_tokens:
                batches.append(current_batch)
                current_batch = [summary]
                current_tokens = entry_tokens
            else:
                current_batch.append(summary)
                current_tokens = projected_tokens if current_tokens else entry_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _split_text_for_summary(self, text: str, target_tokens: int) -> list[str]:
        target_tokens = max(target_tokens, 1)
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_parts = self._split_oversized_summary_block(paragraph, target_tokens)
            for paragraph_part in paragraph_parts:
                paragraph_tokens = self._estimate_tokens(paragraph_part)
                projected_tokens = current_tokens + paragraph_tokens + (12 if current else 0)
                if current and projected_tokens > target_tokens:
                    chunks.append("\n\n".join(current).strip())
                    current = [paragraph_part]
                    current_tokens = paragraph_tokens
                else:
                    current.append(paragraph_part)
                    current_tokens = projected_tokens if current_tokens else paragraph_tokens

        if current:
            chunks.append("\n\n".join(current).strip())

        return [chunk for chunk in chunks if chunk]

    def _force_split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if len(paragraphs) > 1:
            return self._split_segments_in_half(paragraphs, separator="\n\n")

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if len(sentences) > 1:
            return self._split_segments_in_half(sentences, separator=" ")

        words = text.split()
        if len(words) > 1:
            return self._split_segments_in_half(words, separator=" ")

        return [text]

    def _split_oversized_summary_block(self, text: str, target_tokens: int) -> list[str]:
        text = text.strip()
        if not text or self._estimate_tokens(text) <= target_tokens:
            return [text] if text else []

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if len(sentences) > 1:
            return self._pack_text_segments(sentences, target_tokens, separator=" ")

        words = text.split()
        if not words:
            return []
        return self._pack_text_segments(words, target_tokens, separator=" ")

    @staticmethod
    def _pack_text_segments(segments: list[str], target_tokens: int, separator: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = DocumentCorpus._estimate_tokens_static(segment)
            projected_tokens = current_tokens + segment_tokens + (4 if current else 0)
            if current and projected_tokens > target_tokens:
                chunks.append(separator.join(current).strip())
                current = [segment]
                current_tokens = segment_tokens
            else:
                current.append(segment)
                current_tokens = projected_tokens if current_tokens else segment_tokens

        if current:
            chunks.append(separator.join(current).strip())

        return chunks

    def _split_segments_in_half(self, segments: list[str], separator: str) -> list[str]:
        if len(segments) <= 1:
            return [segments[0]] if segments else []

        total_tokens = sum(self._estimate_tokens(segment) for segment in segments)
        half_target = max(1, total_tokens // 2)

        first_half: list[str] = []
        running_tokens = 0
        for index, segment in enumerate(segments):
            remaining = len(segments) - index
            if first_half and running_tokens >= half_target and remaining >= 1:
                break
            first_half.append(segment)
            running_tokens += self._estimate_tokens(segment)

        split_index = len(first_half)
        if split_index <= 0 or split_index >= len(segments):
            split_index = max(1, len(segments) // 2)

        left = separator.join(segments[:split_index]).strip()
        right = separator.join(segments[split_index:]).strip()
        return [part for part in (left, right) if part]

    def _estimate_tokens(self, text: str) -> int:
        return self._estimate_tokens_static(
            text,
            chars_per_token=self.config.token_estimate_chars_per_token,
            tokens_per_word=self.config.token_estimate_tokens_per_word,
        )

    def _scaled_summary_budget(self, doc_tokens: int) -> int:
        """Scale summary output budget with document size.  Small docs
        (<3000 tokens) use the base budget; larger docs get proportionally
        more up to a cap."""
        base = self.config.doc_summary_max_output_tokens
        cap = self.config.doc_summary_max_output_tokens_cap
        if doc_tokens <= 3000:
            return base
        scale_factor = min(doc_tokens / 3000, cap / base)
        return min(int(base * scale_factor), cap)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        compact = re.sub(r"\s+", " ", (text or "").strip())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _write_ingestion_log(self, document: DocumentRecord) -> None:
        log_root = self._resolve_debug_log_dir()
        if log_root is None:
            return

        ingestion_dir = log_root / "ingestion"
        ingestion_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self._safe_log_name(Path(document.source_path).stem or document.title)}.txt"
        output_path = ingestion_dir / file_name
        self._append_ingestion_index_entry(
            source_path=document.source_path,
            status="ingested",
            detail=f"ingested with {len(document.sections)} sections",
            metadata_file_name=file_name,
            section_count=len(document.sections),
        )

        lines: list[str] = [
            f"Source path: {document.source_path}",
            f"Source file name: {Path(document.source_path).name}",
            f"Metadata file name: {file_name}",
            f"Title: {document.title}",
            f"Canonical title: {document.canonical_title or 'unknown'}",
            f"Document type: {document.document_type or 'document'}",
            f"Version: {document.version or 'unknown'}",
            f"Effective date: {document.effective_date or 'unknown'}",
            f"Tags: {', '.join(document.metadata_tags) or 'none'}",
            f"Audiences: {', '.join(document.audiences) or 'none'}",
            f"Keywords: {', '.join(document.keywords) or 'none'}",
            f"Section count: {len(document.sections)}",
            "",
            "=== Document Summary ===",
            self.redactor.unmask_text(document.summary).strip() or "(empty)",
            "",
        ]

        for index, section in enumerate(document.sections, start=1):
            lines.extend(
                [
                    f"=== Section {index}: {section.title} ===",
                    f"Type: {section.section_type or 'general'}",
                    f"Tags: {', '.join(section.metadata_tags) or 'none'}",
                    f"Keywords: {', '.join(section.keywords) or 'none'}",
                    "Summary:",
                    self.redactor.unmask_text(section.summary).strip() or "(empty)",
                    "",
                    "Raw text:",
                    section.raw_text.strip() or "(empty)",
                    "",
                ]
            )

        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _write_ingestion_run_header(self, folder_path: str, file_paths: list[str]) -> None:
        log_root = self._resolve_debug_log_dir()
        if log_root is None:
            return

        index_path = log_root / "ingestion_index.txt"
        lines = [
            f"=== Ingestion Run: {utc_now_iso()} ===",
            f"Run ID: {self._active_ingestion_run_id or 'unknown'}",
            f"Folder: {folder_path}",
            f"Discovered supported files: {len(file_paths)}",
            "Files:",
            *[f"- {Path(file_path).name} | {file_path}" for file_path in file_paths],
            "",
        ]
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _append_ingestion_index_entry(
        self,
        source_path: str,
        status: str,
        detail: str,
        metadata_file_name: str = "",
        section_count: int | None = None,
    ) -> None:
        log_root = self._resolve_debug_log_dir()
        if log_root is None:
            return

        index_path = log_root / "ingestion_index.txt"
        lines = [
            f"Run ID: {self._active_ingestion_run_id or 'unknown'}",
            f"Status: {status}",
            f"Source file name: {Path(source_path).name}",
            f"Source path: {source_path}",
            f"Detail: {detail}",
            "",
        ]
        if metadata_file_name:
            lines.insert(-1, f"Metadata file name: {metadata_file_name}")
        if section_count is not None:
            lines.insert(-1, f"Section count: {section_count}")
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def _write_ingestion_failure_log(
        self,
        source_path: str,
        stage: str,
        reason: str,
        exc: Exception | None = None,
    ) -> None:
        log_root = self._resolve_debug_log_dir()
        if log_root is None:
            return

        failures_dir = log_root / "ingestion_failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        output_path = failures_dir / f"{uuid.uuid4()}.txt"
        lines = [
            f"Run ID: {self._active_ingestion_run_id or 'unknown'}",
            f"Source file name: {Path(source_path).name}",
            f"Source path: {source_path}",
            f"Stage: {stage}",
            f"Reason: {reason}",
        ]
        if exc is not None:
            lines.extend(
                [
                    "",
                    "=== Traceback ===",
                    traceback.format_exc().strip() or f"{type(exc).__name__}: {exc}",
                ]
            )
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _llm_text_with_debug_log(
        self,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> str:
        try:
            response_text = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            self._write_llm_debug_log(
                purpose=purpose,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text="",
                max_output_tokens=max_output_tokens,
                error_text=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._write_llm_debug_log(
            purpose=purpose,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            max_output_tokens=max_output_tokens,
        )
        return response_text

    def _write_llm_debug_log(
        self,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        max_output_tokens: int,
        error_text: str = "",
    ) -> None:
        write_llm_debug_log_pair(
            log_root=self._resolve_debug_log_dir(),
            redactor=self.redactor,
            provider=self.config.ai_provider,
            model_name=self.config.chat_model,
            purpose=purpose,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            response_text=response_text,
            error_text=error_text,
        )

    def _resolve_debug_log_dir(self) -> Path | None:
        if not self.config.debug:
            return None
        raw_path = (self.config.debug_log_dir or "").strip()
        if not raw_path:
            return None
        base_path = Path(self.config.document_folder)
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = base_path / candidate
        return candidate

    @staticmethod
    def _safe_log_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
        return cleaned.strip("._") or "policygpt_log"

    @staticmethod
    def _estimate_tokens_static(
        text: str,
        chars_per_token: int = 4,
        tokens_per_word: float = 1.3,
    ) -> int:
        compact = (text or "").strip()
        if not compact:
            return 0
        char_based = max(1, (len(compact) + max(1, chars_per_token) - 1) // max(1, chars_per_token))
        word_count = len(re.findall(r"\S+", compact))
        word_based = max(1, int(word_count * max(tokens_per_word, 0.1)))
        return max(char_based, word_based)

    @staticmethod
    def _format_summary_chunks(summaries: list[str]) -> str:
        return "\n\n".join(
            f"Partial summary {index}:\n{summary}"
            for index, summary in enumerate(summaries, start=1)
        )

    @staticmethod
    def _format_error_for_progress(exc: Exception, limit: int = 120) -> str:
        compact = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
