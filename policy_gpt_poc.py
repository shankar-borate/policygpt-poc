import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

class Config:
    # Folder containing .html / .htm / .txt files
    DOCUMENT_FOLDER = r"D:\policy-mgmt\policies"
    SUPPORTED_FILE_PATTERNS = ("*.html", "*.htm", "*.txt")
    EXCLUDED_FILE_NAME_PARTS = ("_summary",)

    # Models
    CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # Retrieval
    TOP_DOCS = 3
    TOP_SECTIONS_PER_DOC = 3
    MAX_SECTIONS_TO_LLM = 4

    # Conversation memory
    MAX_RECENT_MESSAGES = 6
    SUMMARIZE_AFTER_TURNS = 8

    # Section sizing
    MIN_SECTION_CHARS = 300
    TARGET_SECTION_CHARS = 1800
    MAX_SECTION_CHARS = 3200

    # Redaction
    REDACTION_RULES = {
        "Kotak": "KKK",
        "kotak": "KKK",
        "KOTAK": "KKK",
    }

    # LLM generation limits
    DOC_SUMMARY_MAX_OUTPUT_TOKENS = 400
    SECTION_SUMMARY_MAX_OUTPUT_TOKENS = 220
    CHAT_MAX_OUTPUT_TOKENS = 900
    CONVERSATION_SUMMARY_MAX_OUTPUT_TOKENS = 250

    # Optional: if True, prints retrieval details
    DEBUG = os.getenv("POLICY_GPT_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class SectionRecord:
    section_id: str
    title: str
    raw_text: str
    masked_text: str
    summary: str
    summary_embedding: np.ndarray
    source_path: str
    doc_id: str
    order_index: int


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_path: str
    raw_text: str
    masked_text: str
    summary: str
    summary_embedding: np.ndarray
    sections: List[SectionRecord] = field(default_factory=list)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class SourceReference:
    document_title: str
    section_title: str
    source_path: str
    score: float


@dataclass
class ChatResult:
    thread_id: str
    answer: str
    sources: List[SourceReference]


@dataclass
class ThreadState:
    thread_id: str
    recent_messages: List[Message] = field(default_factory=list)
    display_messages: List[Message] = field(default_factory=list)
    conversation_summary: str = ""
    active_doc_ids: List[str] = field(default_factory=list)
    active_section_ids: List[str] = field(default_factory=list)
    current_topic: str = ""
    title: str = "New chat"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_answer_sources: List[SourceReference] = field(default_factory=list)


# ============================================================
# MASKING / UNMASKING
# ============================================================

class Redactor:
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping
        self.reverse_mapping = {}
        for k, v in mapping.items():
            self.reverse_mapping[v] = "Kotak"

    def mask_text(self, text: str) -> str:
        masked = text
        for original, replacement in self.mapping.items():
            masked = masked.replace(original, replacement)
        return masked

    def unmask_text(self, text: str) -> str:
        unmasked = text
        for replacement, original in self.reverse_mapping.items():
            unmasked = unmasked.replace(replacement, original)
        return unmasked


# ============================================================
# OPENAI CLIENT WRAPPER
# ============================================================

class OpenAIService:
    def __init__(self, chat_model: str, embedding_model: str):
        self.client = OpenAI()  # OPENAI_API_KEY must already be in env
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Batch embeddings for efficiency.
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return vectors

    def llm_text(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        """
        Stable text generation path using Chat Completions.
        """
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_output_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


# ============================================================
# FILE EXTRACTION
# ============================================================

class FileExtractor:
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
    def extract_from_html(path: str) -> Tuple[str, List[Tuple[str, str]]]:
        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link"]):
            tag.decompose()

        html_title = ""
        if soup.title and soup.title.text.strip():
            html_title = FileExtractor.clean_whitespace(soup.title.text)

        candidates = soup.find_all([
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "table", "tr", "td", "th"
        ])

        units: List[Tuple[str, str]] = []
        for node in candidates:
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            tag = node.name.lower()
            units.append((tag, text))

        doc_title = FileExtractor._select_document_title(
            path=path,
            html_title=html_title,
            units=units,
        )

        sections = FileExtractor._group_units_into_sections(units)
        return doc_title, sections

    @staticmethod
    def extract_from_plain_text(path: str) -> Tuple[str, List[Tuple[str, str]]]:
        text = FileExtractor.read_text_file(path)
        text = FileExtractor.clean_whitespace(text)

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        doc_title = Path(path).stem

        if lines:
            first = lines[0]
            if len(first) < 150:
                doc_title = first

        units: List[Tuple[str, str]] = []
        buffer: List[str] = []

        def flush_paragraph():
            nonlocal buffer
            if buffer:
                para = " ".join(buffer).strip()
                if para:
                    units.append(("p", para))
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
        sections = FileExtractor._group_units_into_sections(units)
        return doc_title, sections

    @staticmethod
    def _group_units_into_sections(units: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        sections: List[Tuple[str, str]] = []
        current_title = "Introduction"
        current_parts: List[str] = []

        def flush():
            nonlocal current_title, current_parts
            content = "\n".join(current_parts).strip()
            if content:
                sections.append((current_title, FileExtractor.clean_whitespace(content)))
            current_parts = []

        for tag, text in units:
            if tag.startswith("h"):
                flush()
                current_title = text[:200]
            else:
                current_parts.append(text)

        flush()

        if len(sections) <= 1:
            all_text = "\n".join([text for _, text in sections]) if sections else ""
            if not all_text:
                return []
            return FileExtractor._split_large_text_into_synthetic_sections(all_text)

        final_sections: List[Tuple[str, str]] = []
        for title, text in sections:
            if len(text) <= Config.MAX_SECTION_CHARS:
                final_sections.append((title, text))
            else:
                split_sections = FileExtractor._split_large_text_into_synthetic_sections(text, title_prefix=title)
                final_sections.extend(split_sections)

        return final_sections

    @staticmethod
    def _split_large_text_into_synthetic_sections(text: str, title_prefix: str = "Section") -> List[Tuple[str, str]]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        sections: List[Tuple[str, str]] = []
        current: List[str] = []
        current_len = 0
        part_no = 1

        for para in paragraphs:
            para_len = len(para)
            too_big = current_len + para_len > Config.TARGET_SECTION_CHARS

            if current and too_big and current_len >= Config.MIN_SECTION_CHARS:
                section_text = "\n\n".join(current)
                sections.append((f"{title_prefix} - Part {part_no}", section_text))
                part_no += 1
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            sections.append((f"{title_prefix} - Part {part_no}", "\n\n".join(current)))

        return sections

    @staticmethod
    def _select_document_title(path: str, html_title: str, units: List[Tuple[str, str]]) -> str:
        file_title = FileExtractor._clean_title_candidate(Path(path).stem.replace("_", " "))
        heading_title = ""

        for tag, text in units:
            if not tag.startswith("h"):
                continue
            heading_title = FileExtractor._clean_title_candidate(text)
            if heading_title:
                break

        cleaned_html_title = FileExtractor._clean_title_candidate(html_title)
        html_overlap = FileExtractor._title_overlap_score(cleaned_html_title, file_title)
        heading_overlap = FileExtractor._title_overlap_score(heading_title, file_title)

        if heading_title and not FileExtractor._looks_like_bad_title(heading_title, file_title) and (
            FileExtractor._looks_like_bad_title(cleaned_html_title, file_title)
            or heading_overlap > html_overlap
        ):
            return heading_title

        if cleaned_html_title and not FileExtractor._looks_like_bad_title(cleaned_html_title, file_title):
            return cleaned_html_title

        if heading_title and not FileExtractor._looks_like_bad_title(heading_title, file_title):
            return heading_title

        return file_title or cleaned_html_title or Path(path).stem

    @staticmethod
    def _clean_title_candidate(text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        return compact[:200]

    @staticmethod
    def _title_overlap_score(candidate: str, file_title: str) -> float:
        candidate_tokens = FileExtractor._tokenize_title(candidate)
        file_tokens = FileExtractor._tokenize_title(file_title)
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

    @staticmethod
    def _looks_like_bad_title(title: str, file_title: str) -> bool:
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

        return FileExtractor._title_overlap_score(title, file_title) == 0.0


# ============================================================
# RETRIEVAL HELPERS
# ============================================================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v)
    if denom == 0:
        return v
    return v / denom


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.dot(matrix, query)


def list_supported_policy_files(folder_path: str) -> List[str]:
    folder = Path(folder_path)
    file_paths: List[str] = []
    for pattern in Config.SUPPORTED_FILE_PATTERNS:
        file_paths.extend(str(path) for path in folder.glob(pattern))
    filtered_paths: List[str] = []
    for file_path in sorted(file_paths):
        file_name = Path(file_path).name.lower()
        if any(part in file_name for part in Config.EXCLUDED_FILE_NAME_PARTS):
            continue
        filtered_paths.append(file_path)
    return filtered_paths


ProgressCallback = Callable[[int, int, Optional[str], int, int], None]


# ============================================================
# MAIN BOT
# ============================================================

class PolicyGPTBot:
    def __init__(self):
        self.redactor = Redactor(Config.REDACTION_RULES)
        self.ai = OpenAIService(
            chat_model=Config.CHAT_MODEL,
            embedding_model=Config.EMBEDDING_MODEL,
        )

        self.documents: Dict[str, DocumentRecord] = {}
        self.sections: Dict[str, SectionRecord] = {}

        self.doc_ids: List[str] = []
        self.doc_embedding_matrix: Optional[np.ndarray] = None

        self.section_ids: List[str] = []
        self.section_embedding_matrix: Optional[np.ndarray] = None

        self.threads: Dict[str, ThreadState] = {}

    def new_thread(self) -> str:
        thread_id = str(uuid.uuid4())
        self.threads[thread_id] = ThreadState(thread_id=thread_id)
        return thread_id

    def reset_thread(self, thread_id: str) -> None:
        self.threads[thread_id] = ThreadState(thread_id=thread_id)

    def get_thread(self, thread_id: str) -> ThreadState:
        if thread_id not in self.threads:
            self.threads[thread_id] = ThreadState(thread_id=thread_id)
        return self.threads[thread_id]

    def list_threads(self) -> List[ThreadState]:
        return sorted(self.threads.values(), key=lambda thread: thread.updated_at, reverse=True)

    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        file_paths = list_supported_policy_files(folder_path)
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
                path,
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
        self._rebuild_indexes()

    def ingest_file(
        self,
        path: str,
        progress_callback: Optional[ProgressCallback] = None,
        processed_files: int = 0,
        total_files: int = 0,
    ) -> None:
        path_obj = Path(path)
        file_name = path_obj.name
        ext = path_obj.suffix.lower()

        self._emit_progress(
            progress_callback,
            processed_files,
            total_files,
            f"{file_name} - extracting sections",
        )

        if ext in [".html", ".htm"]:
            title, sections = FileExtractor.extract_from_html(path)
        elif ext == ".txt":
            title, sections = FileExtractor.extract_from_plain_text(path)
        else:
            return

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
        doc_summary = self._create_document_summary(masked_title, masked_full_text)
        doc_embedding = self._embed_one(doc_summary)

        doc_id = str(uuid.uuid4())
        doc_record = DocumentRecord(
            doc_id=doc_id,
            title=title,
            source_path=path,
            raw_text=full_text,
            masked_text=masked_full_text,
            summary=doc_summary,
            summary_embedding=doc_embedding,
            sections=[],
        )

        valid_sections = [
            (section_title, section_text)
            for section_title, section_text in sections
            if section_text.strip()
        ]
        total_sections = len(valid_sections)

        for idx, (section_title, section_text) in enumerate(valid_sections, start=1):
            section_label = self._format_progress_label(section_title)
            self._emit_progress(
                progress_callback,
                processed_files,
                total_files,
                f"{file_name} - section {idx}/{total_sections}: {section_label}",
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
            sec = SectionRecord(
                section_id=section_id,
                title=section_title,
                raw_text=section_text,
                masked_text=masked_section_text,
                summary=section_summary,
                summary_embedding=section_embedding,
                source_path=path,
                doc_id=doc_id,
                order_index=idx - 1,
            )
            doc_record.sections.append(sec)
            self.sections[section_id] = sec

        self.documents[doc_id] = doc_record
        self._emit_progress(
            progress_callback,
            processed_files,
            total_files,
            f"{file_name} - finished with {len(doc_record.sections)} sections",
        )

        if Config.DEBUG:
            print(f"Ingested: {path}")
            print(f"  Title: {title}")
            print(f"  Sections: {len(doc_record.sections)}")

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
            max_output_tokens=Config.DOC_SUMMARY_MAX_OUTPUT_TOKENS,
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
            max_output_tokens=Config.SECTION_SUMMARY_MAX_OUTPUT_TOKENS,
        )

    def _embed_one(self, text: str) -> np.ndarray:
        vec = self.ai.embed_texts([text])[0]
        return l2_normalize(vec)

    def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        processed_files: int,
        total_files: int,
        current_step: Optional[str],
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

    def _rebuild_indexes(self) -> None:
        self.doc_ids = []
        doc_vectors = []

        for doc_id, doc in self.documents.items():
            self.doc_ids.append(doc_id)
            doc_vectors.append(doc.summary_embedding)

        self.section_ids = []
        sec_vectors = []

        for sec_id, sec in self.sections.items():
            self.section_ids.append(sec_id)
            sec_vectors.append(sec.summary_embedding)

        self.doc_embedding_matrix = np.vstack(doc_vectors) if doc_vectors else None
        self.section_embedding_matrix = np.vstack(sec_vectors) if sec_vectors else None

    def chat(self, thread_id: str, user_question: str) -> ChatResult:
        if not self.documents:
            raise RuntimeError("No documents ingested. Call ingest_folder() first.")

        thread = self.get_thread(thread_id)
        first_user_message = not any(message.role == "user" for message in thread.display_messages)

        retrieval_query = self._build_retrieval_query(thread, user_question)
        masked_retrieval_query = self.redactor.mask_text(retrieval_query)
        query_vec = self._embed_one(masked_retrieval_query)

        top_docs = self._retrieve_top_docs(query_vec, preferred_doc_ids=thread.active_doc_ids)
        top_sections = self._retrieve_top_sections(query_vec, top_docs, preferred_section_ids=thread.active_section_ids)

        prompt_payload = self._build_answer_context(
            thread=thread,
            user_question=user_question,
            top_docs=top_docs,
            top_sections=top_sections,
        )

        masked_answer = self.ai.llm_text(
            system_prompt=self.redactor.mask_text(self._system_prompt()),
            user_prompt=self.redactor.mask_text(prompt_payload),
            max_output_tokens=Config.CHAT_MAX_OUTPUT_TOKENS,
        )

        sources = [
            SourceReference(
                document_title=self.documents[sec.doc_id].title,
                section_title=sec.title,
                source_path=sec.source_path,
                score=score,
            )
            for sec, score in top_sections[:3]
        ]
        final_answer = self._append_reference_file_names(
            self.redactor.unmask_text(masked_answer),
            sources,
        )

        thread.recent_messages.append(Message(role="user", content=user_question))
        thread.recent_messages.append(Message(role="assistant", content=final_answer))
        thread.recent_messages = thread.recent_messages[-Config.MAX_RECENT_MESSAGES:]
        thread.display_messages.append(Message(role="user", content=user_question))
        thread.display_messages.append(Message(role="assistant", content=final_answer))

        thread.active_doc_ids = [doc.doc_id for doc, _ in top_docs]
        thread.active_section_ids = [sec.section_id for sec, _ in top_sections]
        thread.current_topic = self._derive_thread_title(user_question, limit=90)
        thread.last_answer_sources = sources
        if first_user_message:
            thread.title = self._derive_thread_title(user_question)
        thread.updated_at = utc_now_iso()

        if len(thread.recent_messages) >= Config.SUMMARIZE_AFTER_TURNS:
            thread.conversation_summary = self._refresh_conversation_summary(thread)

        return ChatResult(
            thread_id=thread_id,
            answer=final_answer,
            sources=thread.last_answer_sources,
        )

    def ask(self, thread_id: str, user_question: str) -> str:
        return self.chat(thread_id, user_question).answer

    def _system_prompt(self) -> str:
        return (
            "You are a conversational enterprise document assistant.\n"
            "Rules:\n"
            "1. Answer only from the provided document evidence.\n"
            "2. If the answer is not clearly present, say that it is not clearly stated in the provided documents.\n"
            "3. Be conversational but precise.\n"
            "4. Prefer the current conversation context when interpreting follow-up questions like 'what about this', 'same policy', 'that section'.\n"
            "5. If multiple documents are relevant, separate them clearly.\n"
            "6. Mention section titles and file names when useful.\n"
            "7. Do not hallucinate.\n"
            "8. Format the answer in clean Markdown with short sections or bullets when helpful.\n"
            "9. Default to a sharp, concise answer that gets to the point quickly.\n"
            "10. Give more detail only when the user explicitly asks for it, such as 'in detail', 'detailed', 'step by step', or similar.\n"
            "11. Do not add unnecessary background, repetition, or long caveats.\n"
            "12. For direct questions, lead with the answer in the first line.\n"
            "13. Ignore retrieved text that is about a different policy or a different topic than the user's question, even if it appears semantically similar.\n"
            "14. If the evidence does not explicitly support the asked point, say it is not clearly stated instead of inferring."
        )

    @staticmethod
    def _append_reference_file_names(answer: str, sources: List[SourceReference]) -> str:
        file_names: List[str] = []
        seen = set()
        for source in sources:
            file_name = Path(source.source_path).name
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            file_names.append(file_name)

        if not file_names:
            return answer.strip()

        reference_line = f"Reference: {', '.join(file_names)}"
        clean_answer = answer.strip()
        if not clean_answer:
            return reference_line
        return f"{clean_answer}\n\n{reference_line}"

    @staticmethod
    def _derive_thread_title(text: str, limit: int = 56) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return "New chat"
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3].rstrip() + "..."

    def _build_retrieval_query(self, thread: ThreadState, user_question: str) -> str:
        parts = []
        if thread.conversation_summary:
            parts.append(f"Conversation summary: {thread.conversation_summary}")
        if thread.current_topic:
            parts.append(f"Current topic: {thread.current_topic}")
        if thread.active_doc_ids:
            active_titles = [self.documents[d].title for d in thread.active_doc_ids if d in self.documents]
            if active_titles:
                parts.append(f"Current documents: {', '.join(active_titles)}")
        parts.append(f"User question: {user_question}")
        return "\n".join(parts)

    def _retrieve_top_docs(
        self,
        query_vec: np.ndarray,
        preferred_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[DocumentRecord, float]]:
        if self.doc_embedding_matrix is None or not self.doc_ids:
            return []

        scores = cosine_similarity(query_vec, self.doc_embedding_matrix)
        preferred_doc_ids = preferred_doc_ids or []
        preferred_set = set(preferred_doc_ids)

        rescored = []
        for idx, doc_id in enumerate(self.doc_ids):
            score = float(scores[idx])
            if doc_id in preferred_set:
                score += 0.08
            rescored.append((doc_id, score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        top = rescored[:Config.TOP_DOCS]
        results = [(self.documents[doc_id], score) for doc_id, score in top]

        if Config.DEBUG:
            print("\nTop documents:")
            for doc, score in results:
                print(f"  {score:.4f} | {doc.title}")

        return results

    def _retrieve_top_sections(
        self,
        query_vec: np.ndarray,
        top_docs: List[Tuple[DocumentRecord, float]],
        preferred_section_ids: Optional[List[str]] = None
    ) -> List[Tuple[SectionRecord, float]]:
        preferred_section_ids = preferred_section_ids or []
        preferred_set = set(preferred_section_ids)

        candidate_sections: List[Tuple[SectionRecord, float]] = []

        for doc, doc_score in top_docs:
            if not doc.sections:
                continue

            section_ids = [s.section_id for s in doc.sections]
            section_vecs = np.vstack([self.sections[sid].summary_embedding for sid in section_ids])

            local_scores = cosine_similarity(query_vec, section_vecs)
            local_pairs = []

            for idx, sid in enumerate(section_ids):
                sec = self.sections[sid]
                score = float(local_scores[idx])
                score = (0.70 * score) + (0.30 * doc_score)
                if sid in preferred_set:
                    score += 0.08
                local_pairs.append((sec, score))

            local_pairs.sort(key=lambda x: x[1], reverse=True)
            candidate_sections.extend(local_pairs[:Config.TOP_SECTIONS_PER_DOC])

        candidate_sections.sort(key=lambda x: x[1], reverse=True)
        results = candidate_sections[:Config.MAX_SECTIONS_TO_LLM]

        if Config.DEBUG:
            print("\nTop sections:")
            for sec, score in results:
                print(f"  {score:.4f} | {self.documents[sec.doc_id].title} :: {sec.title}")

        return results

    def _build_answer_context(
        self,
        thread: ThreadState,
        user_question: str,
        top_docs: List[Tuple[DocumentRecord, float]],
        top_sections: List[Tuple[SectionRecord, float]],
    ) -> str:
        recent_chat = "\n".join(
            [f"{m.role.upper()}: {m.content}" for m in thread.recent_messages[-Config.MAX_RECENT_MESSAGES:]]
        ) or "None"

        doc_context_parts = []
        for doc, score in top_docs:
            doc_context_parts.append(
                f"[Document: {doc.title} | Score: {score:.4f} | File: {doc.source_path}]\n"
                f"Document summary:\n{self.redactor.unmask_text(doc.summary)}"
            )

        section_context_parts = []
        for sec, score in top_sections:
            doc = self.documents[sec.doc_id]
            section_context_parts.append(
                f"[Document: {doc.title} | Section: {sec.title} | Score: {score:.4f} | File: {sec.source_path}]\n"
                f"Section summary:\n{self.redactor.unmask_text(sec.summary)}\n\n"
                f"Original section text:\n{sec.raw_text}"
            )

        payload = (
            f"Conversation summary:\n{thread.conversation_summary or 'None'}\n\n"
            f"Recent chat:\n{recent_chat}\n\n"
            f"User question:\n{user_question}\n\n"
            f"Retrieved document context:\n{chr(10).join(doc_context_parts)}\n\n"
            f"Retrieved section evidence:\n{chr(10).join(section_context_parts)}\n\n"
            "Answer the user conversationally, but ground every claim in the provided evidence."
        )
        return payload

    def _refresh_conversation_summary(self, thread: ThreadState) -> str:
        recent_chat = "\n".join([f"{m.role.upper()}: {m.content}" for m in thread.recent_messages])
        system_prompt = (
            "Summarize this conversation for future retrieval and follow-up handling. "
            "Capture current document, current topic, key answered points, and unresolved questions. "
            "Be concise and factual."
        )
        summary = self.ai.llm_text(
            system_prompt=self.redactor.mask_text(system_prompt),
            user_prompt=self.redactor.mask_text(recent_chat),
            max_output_tokens=Config.CONVERSATION_SUMMARY_MAX_OUTPUT_TOKENS,
        )
        return self.redactor.unmask_text(summary)


# ============================================================
# SIMPLE CLI
# ============================================================

def print_help():
    print("""
Commands:
  /new               -> create a new thread and switch to it
  /reset             -> reset current thread memory
  /threads           -> list thread ids
  /use <thread_id>   -> switch to an existing thread
  /sources           -> show last answer sources
  /exit              -> quit

Anything else is treated as a user question.
""")


def create_ready_bot(
    folder: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> PolicyGPTBot:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

    resolved_folder = folder or Config.DOCUMENT_FOLDER
    bot = PolicyGPTBot()
    bot.ingest_folder(resolved_folder, progress_callback=progress_callback)
    return bot


def main():
    folder = Config.DOCUMENT_FOLDER
    print(f"Ingesting folder: {folder}")
    bot = create_ready_bot(folder)

    current_thread = bot.new_thread()
    print(f"\nReady. Current thread: {current_thread}")
    print_help()

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/new":
            current_thread = bot.new_thread()
            print(f"Switched to new thread: {current_thread}")
            continue

        if user_input == "/reset":
            bot.reset_thread(current_thread)
            print(f"Thread reset: {current_thread}")
            continue

        if user_input == "/threads":
            print("Threads:")
            for tid in bot.threads.keys():
                marker = " <current>" if tid == current_thread else ""
                print(f"  {tid}{marker}")
            continue

        if user_input.startswith("/use "):
            _, tid = user_input.split(" ", 1)
            tid = tid.strip()
            current_thread = tid
            bot.get_thread(current_thread)
            print(f"Switched to thread: {current_thread}")
            continue

        if user_input == "/sources":
            thread = bot.get_thread(current_thread)
            if not thread.last_answer_sources:
                print("No sources yet.")
            else:
                print("Last answer sources:")
                for source in thread.last_answer_sources:
                    print(
                        "  -",
                        f"{source.document_title} :: {source.section_title} ({source.source_path})",
                    )
            continue

        answer = bot.ask(current_thread, user_input)
        print(f"\nBot> {answer}")


if __name__ == "__main__":
    main()
