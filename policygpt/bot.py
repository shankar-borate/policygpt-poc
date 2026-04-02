import re
from pathlib import Path
from urllib.parse import quote

import numpy as np

from policygpt.config import Config
from policygpt.conversations import ConversationManager
from policygpt.corpus import DocumentCorpus, ProgressCallback
from policygpt.models import ChatResult, Message, SourceReference
from policygpt.models import utc_now_iso
from policygpt.services.base import AIService
from policygpt.services.bedrock_service import BedrockService
from policygpt.services.file_extractor import FileExtractor
from policygpt.services.openai_service import OpenAIService
from policygpt.services.redaction import Redactor


class PolicyGPTBot:
    def __init__(
        self,
        config: Config,
        ai: AIService | None = None,
        redactor: Redactor | None = None,
        extractor: FileExtractor | None = None,
        corpus: DocumentCorpus | None = None,
        conversations: ConversationManager | None = None,
    ) -> None:
        self.config = config
        self.redactor = redactor or Redactor(config.redaction_rules)
        self.ai = ai or self._build_ai_service()
        self.extractor = extractor or FileExtractor(config)
        self.corpus = corpus or DocumentCorpus(
            config=config,
            extractor=self.extractor,
            ai=self.ai,
            redactor=self.redactor,
        )
        self.conversations = conversations or ConversationManager()

    @property
    def documents(self):
        return self.corpus.documents

    @property
    def sections(self):
        return self.corpus.sections

    @property
    def threads(self):
        return self.conversations.threads

    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.corpus.ingest_folder(folder_path, progress_callback=progress_callback)

    def new_thread(self) -> str:
        return self.conversations.new_thread()

    def reset_thread(self, thread_id: str) -> None:
        self.conversations.reset_thread(thread_id)

    def get_thread(self, thread_id: str):
        return self.conversations.get_thread(thread_id)

    def list_threads(self):
        return self.conversations.list_threads()

    def chat(self, thread_id: str, user_question: str) -> ChatResult:
        if not self.documents:
            raise RuntimeError("No documents ingested. Call ingest_folder() first.")

        thread = self.get_thread(thread_id)
        first_user_message = not any(message.role == "user" for message in thread.display_messages)

        retrieval_query = self._build_retrieval_query(thread, user_question)
        masked_retrieval_query = self.redactor.mask_text(retrieval_query)
        query_vec = self._embed_one(masked_retrieval_query)

        top_docs = self.corpus.retrieve_top_docs(query_vec, preferred_doc_ids=thread.active_doc_ids)
        top_sections = self.corpus.retrieve_top_sections(
            query_vec,
            top_docs,
            preferred_section_ids=thread.active_section_ids,
        )

        prompt_payload = self._build_answer_context(
            thread=thread,
            user_question=user_question,
            top_docs=top_docs,
            top_sections=top_sections,
        )

        masked_answer = self.ai.llm_text(
            system_prompt=self.redactor.mask_text(self._system_prompt()),
            user_prompt=self.redactor.mask_text(prompt_payload),
            max_output_tokens=self.config.chat_max_output_tokens,
        )

        sources = [
            SourceReference(
                document_title=self.documents[section.doc_id].title,
                section_title=section.title,
                source_path=section.source_path,
                score=score,
            )
            for section, score in top_sections[:3]
        ]
        final_answer = self._append_reference_file_names(
            self._normalize_answer_markdown(self.redactor.unmask_text(masked_answer)),
            sources,
        )

        thread.recent_messages.append(Message(role="user", content=user_question))
        thread.recent_messages.append(Message(role="assistant", content=final_answer))
        thread.recent_messages = thread.recent_messages[-self.config.max_recent_messages :]
        thread.display_messages.append(Message(role="user", content=user_question))
        thread.display_messages.append(Message(role="assistant", content=final_answer))
        thread.active_doc_ids = [document.doc_id for document, _ in top_docs]
        thread.active_section_ids = [section.section_id for section, _ in top_sections]
        thread.current_topic = self._derive_thread_title(user_question, limit=90)
        thread.last_answer_sources = sources
        if first_user_message:
            thread.title = self._derive_thread_title(user_question)
        thread.updated_at = utc_now_iso()

        if len(thread.recent_messages) >= self.config.summarize_after_turns:
            thread.conversation_summary = self._refresh_conversation_summary(thread)

        return ChatResult(
            thread_id=thread_id,
            answer=final_answer,
            sources=thread.last_answer_sources,
        )

    def ask(self, thread_id: str, user_question: str) -> str:
        return self.chat(thread_id, user_question).answer

    def _embed_one(self, text: str) -> np.ndarray:
        return self.corpus.embed_text(text)

    def _build_ai_service(self) -> AIService:
        if self.config.ai_provider == "bedrock":
            return BedrockService(
                chat_model=self.config.chat_model,
                embedding_model=self.config.embedding_model,
                region_name=self.config.bedrock_region,
                rate_limit_retries=self.config.ai_rate_limit_retries,
                rate_limit_backoff_seconds=self.config.ai_rate_limit_backoff_seconds,
            )

        if self.config.ai_provider == "openai":
            return OpenAIService(
                self.config.chat_model,
                self.config.embedding_model,
                rate_limit_retries=self.config.ai_rate_limit_retries,
                rate_limit_backoff_seconds=self.config.ai_rate_limit_backoff_seconds,
            )

        raise ValueError(f"Unsupported AI provider: {self.config.ai_provider}")

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
            "14. If the evidence does not explicitly support the asked point, say it is not clearly stated instead of inferring.\n"
            "15. Rewrite policy language into plain, user-friendly business English. Do not paste long policy wording back to the user.\n"
            "16. For list, eligibility, process, approval, and comparison questions, prefer short bullets with bold labels instead of long paragraphs.\n"
            "17. Avoid horizontal rules, deep heading hierarchies, raw policy numbering, and document-style formatting.\n"
            "18. Do not add a separate source/citation section in the answer body. References are added separately."
        )

    def _append_reference_file_names(self, answer: str, sources: list[SourceReference]) -> str:
        reference_links: list[str] = []
        seen: set[str] = set()
        for source in sources:
            file_name = Path(source.source_path).name
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            encoded_path = quote(source.source_path, safe="")
            url = f"{self.config.public_base_url}/api/documents/open?path={encoded_path}"
            reference_links.append(f"[{file_name}]({url})")

        if not reference_links:
            return answer.strip()

        reference_line = f"Reference: {', '.join(reference_links)}"
        clean_answer = answer.strip()
        if not clean_answer:
            return reference_line
        return f"{clean_answer}\n\n{reference_line}"

    @staticmethod
    def _normalize_answer_markdown(answer: str) -> str:
        cleaned_lines: list[str] = []
        for raw_line in answer.splitlines():
            stripped = raw_line.strip()
            if re.fullmatch(r"[-*_]{3,}", stripped):
                continue
            if stripped.lower().startswith("source:") or stripped.lower().startswith("_source:"):
                continue

            heading_match = re.match(r"^#{4,}\s*(.+)$", raw_line)
            if heading_match:
                cleaned_lines.append(f"### {heading_match.group(1).strip()}")
                continue

            cleaned_lines.append(raw_line.rstrip())

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _derive_thread_title(text: str, limit: int = 56) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return "New chat"
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3].rstrip() + "..."

    def _build_retrieval_query(self, thread, user_question: str) -> str:
        parts: list[str] = []
        if thread.conversation_summary:
            parts.append(f"Conversation summary: {thread.conversation_summary}")
        if thread.current_topic:
            parts.append(f"Current topic: {thread.current_topic}")
        if thread.active_doc_ids:
            active_titles = [self.documents[doc_id].title for doc_id in thread.active_doc_ids if doc_id in self.documents]
            if active_titles:
                parts.append(f"Current documents: {', '.join(active_titles)}")
        parts.append(f"User question: {user_question}")
        return "\n".join(parts)

    def _build_answer_context(
        self,
        thread,
        user_question: str,
        top_docs,
        top_sections,
    ) -> str:
        recent_chat = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in thread.recent_messages[-self.config.max_recent_messages :]
        ) or "None"

        doc_context_parts = []
        for document, score in top_docs:
            doc_context_parts.append(
                f"[Document: {document.title} | Score: {score:.4f} | File: {document.source_path}]\n"
                f"Document summary:\n{self.redactor.unmask_text(document.summary)}"
            )

        section_context_parts = []
        for section, score in top_sections:
            document = self.documents[section.doc_id]
            section_context_parts.append(
                f"[Document: {document.title} | Section: {section.title} | Score: {score:.4f} | File: {section.source_path}]\n"
                f"Section summary:\n{self.redactor.unmask_text(section.summary)}\n\n"
                f"Original section text:\n{section.raw_text}"
            )

        return (
            f"Conversation summary:\n{thread.conversation_summary or 'None'}\n\n"
            f"Recent chat:\n{recent_chat}\n\n"
            f"User question:\n{user_question}\n\n"
            f"Retrieved document context:\n{chr(10).join(doc_context_parts)}\n\n"
            f"Retrieved section evidence:\n{chr(10).join(section_context_parts)}\n\n"
            "Answer the user conversationally, but ground every claim in the provided evidence. "
            "Synthesize the policy into a clean, readable answer instead of copying document structure or wording."
        )

    def _refresh_conversation_summary(self, thread) -> str:
        recent_chat = "\n".join(f"{message.role.upper()}: {message.content}" for message in thread.recent_messages)
        system_prompt = (
            "Summarize this conversation for future retrieval and follow-up handling. "
            "Capture current document, current topic, key answered points, and unresolved questions. "
            "Be concise and factual."
        )
        summary = self.ai.llm_text(
            system_prompt=self.redactor.mask_text(system_prompt),
            user_prompt=self.redactor.mask_text(recent_chat),
            max_output_tokens=self.config.conversation_summary_max_output_tokens,
        )
        return self.redactor.unmask_text(summary)
