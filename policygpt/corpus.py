import re
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from policygpt.config import Config
from policygpt.models import DocumentRecord, SectionRecord
from policygpt.services.file_extractor import FileExtractor
from policygpt.services.openai_service import OpenAIService
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
        ai: OpenAIService,
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
            raise FileNotFoundError(f"No .html/.htm/.txt files found in folder: {folder_path}")

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

        if extension not in {".html", ".htm", ".txt"}:
            return

        title, sections = self.extractor.extract(path)
        full_text = "\n\n".join(text for _, text in sections).strip()
        if not full_text:
            self._emit_progress(
                progress_callback,
                processed_files,
                total_files,
                f"{file_name} - skipped empty document",
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
        document_summary = self._create_document_summary(masked_title, masked_full_text)
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
            section_summary = self._create_section_summary(
                doc_title=masked_title,
                section_title=masked_section_title,
                section_text=masked_section_text,
            )
            section_embedding = self._embed_one(section_summary)

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

    def _create_document_summary(self, masked_title: str, masked_text: str) -> str:
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

    def _create_section_summary(self, doc_title: str, section_title: str, section_text: str) -> str:
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
