import re
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from policygpt.config import Config
from policygpt.models import DocumentRecord, SectionRecord
from policygpt.services.base import AIRequestTooLargeError, AIService
from policygpt.services.file_extractor import FileExtractor
from policygpt.services.redaction import Redactor


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

        self.documents: dict[str, DocumentRecord] = {}
        self.sections: dict[str, SectionRecord] = {}
        self.doc_ids: list[str] = []
        self.doc_embedding_matrix: np.ndarray | None = None
        self.section_ids: list[str] = []
        self.section_embedding_matrix: np.ndarray | None = None

    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        file_paths = self.list_supported_policy_files(folder_path)
        if not file_paths:
            raise FileNotFoundError(f"No .html/.htm/.txt/.pdf files found in folder: {folder_path}")

        total_files = len(file_paths)
        self._emit_progress(progress_callback, 0, total_files, "Scanning policy files")

        for processed_files, path in enumerate(file_paths, start=1):
            file_name = Path(path).name
            self._emit_progress(
                progress_callback,
                processed_files - 1,
                total_files,
                f"{file_name} - reading file",
            )
            try:
                self.ingest_file(
                    path=path,
                    progress_callback=progress_callback,
                    processed_files=processed_files - 1,
                    total_files=total_files,
                )
                self._emit_progress(
                    progress_callback,
                    processed_files,
                    total_files,
                    f"{file_name} - completed",
                )
            except Exception as exc:
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

    def ingest_file(
        self,
        path: str,
        progress_callback: ProgressCallback | None = None,
        processed_files: int = 0,
        total_files: int = 0,
    ) -> None:
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
            return

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
            return

        masked_title = self.redactor.mask_text(title)
        masked_full_text = self.redactor.mask_text(full_text)

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
        document_embedding = self._embed_one(document_summary)

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
            try:
                section_summary = self._create_section_summary(
                    doc_title=masked_title,
                    section_title=masked_section_title,
                    section_text=masked_section_text,
                )
                section_embedding = self._embed_one(section_summary)
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
            )
            document.sections.append(section)
            self.sections[section_id] = section

        self.documents[document_id] = document
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

    def rebuild_indexes(self) -> None:
        self.doc_ids = []
        doc_vectors: list[np.ndarray] = []
        for doc_id, document in self.documents.items():
            self.doc_ids.append(doc_id)
            doc_vectors.append(document.summary_embedding)

        self.section_ids = []
        section_vectors: list[np.ndarray] = []
        for section_id, section in self.sections.items():
            self.section_ids.append(section_id)
            section_vectors.append(section.summary_embedding)

        self.doc_embedding_matrix = np.vstack(doc_vectors) if doc_vectors else None
        self.section_embedding_matrix = np.vstack(section_vectors) if section_vectors else None

    def retrieve_top_docs(
        self,
        query_vec: np.ndarray,
        preferred_doc_ids: list[str] | None = None,
    ) -> list[tuple[DocumentRecord, float]]:
        if self.doc_embedding_matrix is None or not self.doc_ids:
            return []

        scores = cosine_similarity(query_vec, self.doc_embedding_matrix)
        preferred_set = set(preferred_doc_ids or [])
        rescored: list[tuple[str, float]] = []

        for index, doc_id in enumerate(self.doc_ids):
            score = float(scores[index])
            if doc_id in preferred_set:
                score += 0.08
            rescored.append((doc_id, score))

        rescored.sort(key=lambda item: item[1], reverse=True)
        results = [(self.documents[doc_id], score) for doc_id, score in rescored[: self.config.top_docs]]

        if self.config.debug:
            print("\nTop documents:")
            for document, score in results:
                print(f"  {score:.4f} | {document.title}")

        return results

    def retrieve_top_sections(
        self,
        query_vec: np.ndarray,
        top_docs: list[tuple[DocumentRecord, float]],
        preferred_section_ids: list[str] | None = None,
    ) -> list[tuple[SectionRecord, float]]:
        preferred_set = set(preferred_section_ids or [])
        candidate_sections: list[tuple[SectionRecord, float]] = []

        for document, document_score in top_docs:
            if not document.sections:
                continue

            section_ids = [section.section_id for section in document.sections]
            section_vectors = np.vstack([self.sections[section_id].summary_embedding for section_id in section_ids])
            local_scores = cosine_similarity(query_vec, section_vectors)

            local_pairs: list[tuple[SectionRecord, float]] = []
            for index, section_id in enumerate(section_ids):
                section = self.sections[section_id]
                score = float(local_scores[index])
                score = (0.70 * score) + (0.30 * document_score)
                if section_id in preferred_set:
                    score += 0.08
                local_pairs.append((section, score))

            local_pairs.sort(key=lambda item: item[1], reverse=True)
            candidate_sections.extend(local_pairs[: self.config.top_sections_per_doc])

        candidate_sections.sort(key=lambda item: item[1], reverse=True)
        results = candidate_sections[: self.config.max_sections_to_llm]

        if self.config.debug:
            print("\nTop sections:")
            for section, score in results:
                print(f"  {score:.4f} | {self.documents[section.doc_id].title} :: {section.title}")

        return results

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
        input_budget = self.config.doc_summary_input_token_budget
        if self._estimate_tokens(masked_text) <= input_budget:
            try:
                return self._summarize_document_text(masked_title, masked_text)
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

    def _summarize_document_text(self, masked_title: str, masked_text: str) -> str:
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
        return self.ai.llm_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.doc_summary_max_output_tokens,
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
        return self.ai.llm_text(
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
        return self.ai.llm_text(
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
        return self.ai.llm_text(
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
        return self.ai.llm_text(
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
        return self.ai.llm_text(
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
        return self.ai.llm_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.config.section_summary_max_output_tokens,
        )

    def _embed_one(self, text: str) -> np.ndarray:
        return l2_normalize(self.ai.embed_texts([text])[0])

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_one(text)

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
