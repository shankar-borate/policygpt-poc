import re
import traceback
import uuid
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
from policygpt.services.debug_logging import write_llm_debug_log_pair
from policygpt.services.file_extractor import FileExtractor
from policygpt.services.openai_service import OpenAIService
from policygpt.services.query_analyzer import QueryAnalysis, QueryAnalyzer
from policygpt.services.redaction import Redactor
from policygpt.services.taxonomy import unique_preserving_order


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
        self.query_analyzer = QueryAnalyzer()
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
        thread = None
        query_analysis = None
        retrieval_query = ""
        top_docs = []
        top_sections = []
        prompt_payload = ""
        sources: list[SourceReference] = []
        try:
            if not self.documents:
                raise RuntimeError("No documents ingested. Call ingest_folder() first.")

            thread = self.get_thread(thread_id)
            first_user_message = not any(message.role == "user" for message in thread.display_messages)
            active_document_titles = [
                self.documents[doc_id].title
                for doc_id in thread.active_doc_ids
                if doc_id in self.documents
            ]
            query_analysis = self.query_analyzer.analyze(
                user_question=user_question,
                active_document_titles=active_document_titles,
                candidate_documents=list(self.documents.values()),
            )

            retrieval_query = self._build_retrieval_query(thread, query_analysis)
            masked_retrieval_query = self.redactor.mask_text(retrieval_query)
            # Fix: embed only the user's question so the semantic vector is
            # not diluted by conversation context or metadata labels.  The
            # full retrieval_query feeds BM25/lexical channels via
            # query_analysis.expanded_terms already.  Use unmasked text so
            # the vector matches unmasked index vectors.
            query_vec = self._embed_one(query_analysis.canonical_question)
            preferred_doc_ids = thread.active_doc_ids if query_analysis.context_dependent else []
            preferred_section_ids = thread.active_section_ids if query_analysis.context_dependent else []

            top_docs = self.corpus.retrieve_top_docs(
                query_vec,
                query_analysis=query_analysis,
                preferred_doc_ids=preferred_doc_ids,
            )
            top_sections = self.corpus.retrieve_top_sections(
                query_vec,
                query_analysis,
                top_docs,
                preferred_section_ids=preferred_section_ids,
            )
            top_docs = self._merge_retrieved_documents(top_docs, top_sections)

            sources = [
                SourceReference(
                    document_title=self.documents[section.doc_id].title,
                    section_title=section.title,
                    source_path=section.source_path,
                    score=score,
                )
                for section, score in top_sections
            ]

            is_answerable = self._is_answerable(query_analysis, top_docs, top_sections)
            if is_answerable:
                prompt_payload = self._build_answer_context(
                    thread=thread,
                    query_analysis=query_analysis,
                    top_docs=top_docs,
                    top_sections=top_sections,
                )

                masked_answer = self._llm_text_with_debug_log(
                    purpose="chat_answer",
                    system_prompt=self.redactor.mask_text(self._system_prompt()),
                    user_prompt=self.redactor.mask_text(prompt_payload),
                    max_output_tokens=self.config.chat_max_output_tokens,
                )
                answer_text = self._normalize_answer_markdown(self.redactor.unmask_text(masked_answer))
            else:
                answer_text = self._build_unanswerable_response(query_analysis, top_docs)

            final_answer = self._append_reference_file_names(answer_text, sources)
            self._write_retrieval_log(
                thread_id=thread.thread_id,
                user_question=user_question,
                query_analysis=query_analysis,
                retrieval_query=retrieval_query,
                top_docs=top_docs,
                top_sections=top_sections,
                is_answerable=is_answerable,
                prompt_payload=prompt_payload,
                final_answer=final_answer,
                sources=sources,
            )

            thread.recent_messages.append(Message(role="user", content=user_question))
            thread.recent_messages.append(Message(role="assistant", content=final_answer))
            thread.recent_messages = thread.recent_messages[-self.config.max_recent_messages :]
            thread.display_messages.append(Message(role="user", content=user_question))
            thread.display_messages.append(Message(role="assistant", content=final_answer))
            thread.active_doc_ids = [document.doc_id for document, _ in top_docs]
            thread.active_section_ids = [section.section_id for section, _ in top_sections]
            topic_summary = ", ".join(query_analysis.topic_hints) if query_analysis.topic_hints else user_question
            thread.current_topic = self._derive_thread_title(topic_summary, limit=90)
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
        except Exception as exc:
            self._write_query_failure_log(
                thread_id=thread_id,
                user_question=user_question,
                query_analysis=query_analysis,
                retrieval_query=retrieval_query,
                top_docs=top_docs,
                top_sections=top_sections,
                prompt_payload=prompt_payload,
                sources=sources,
                exc=exc,
            )
            raise

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
            "4. Use current conversation context only when the user's wording is clearly referential, such as 'what about this', 'same policy', or 'that section'.\n"
            "5. When relevant evidence comes from multiple documents or sections, synthesize across them and call out any document-specific differences clearly.\n"
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
            "18. Do not add a separate source/citation section in the answer body. References are added separately.\n"
            "19. Treat the question analysis as a retrieval aid, but only state things that are clearly supported by the evidence snippets.\n"
            "20. If the evidence covers only part of the answer, answer only that part and say what is not clearly stated.\n"
            "21. Prefer evidence snippets over broad summaries when they conflict.\n"
            "22. When the user asks for a checklist, process, approval path, or timeline, present it in a scannable format.\n"
            "23. If recent chat suggests one document but the current retrieved evidence explicitly defines the asked term in another document, follow the current evidence and briefly note the difference if needed.\n"
            "24. Treat raw policy evidence blocks as the source of truth. Use summaries only for orientation. If a raw evidence block and a summary differ, trust the raw evidence."
        )

    def _answer_format_guidance(self, query_analysis: QueryAnalysis) -> str:
        detail_suffix = (
            " Expand slightly with supporting details because the user explicitly asked for more detail."
            if query_analysis.detail_requested
            else " Keep it tight and focused."
        )
        if "comparison" in query_analysis.intents:
            return (
                "Start with the direct difference, then use short bullets for each item being compared. "
                "Call out document-specific differences explicitly when the evidence comes from different sources."
                + detail_suffix
            )
        if query_analysis.exact_match_expected:
            return (
                "Start with the exact answer in the first line. "
                "Base labels, dates, numbers, thresholds, and definitions on the raw policy evidence blocks. "
                "If the exact wording is not clearly supported, say so instead of generalizing."
                + detail_suffix
            )
        if query_analysis.multi_doc_expected:
            return (
                "Start with the direct answer, then cover the relevant documents in short bullets. "
                "Call out differences or gaps across documents explicitly."
                + detail_suffix
            )
        if "document_lookup" in query_analysis.intents:
            return (
                "Start by naming the closest matching document. Then give a short, user-friendly summary of what it covers. "
                "If the user appears to be asking only to locate the policy, keep the summary brief and point them to the reference."
                + detail_suffix
            )
        if "approval" in query_analysis.intents:
            return (
                "Start with who approves, then use short bullets with bold labels for each approval path or threshold."
                + detail_suffix
            )
        if "eligibility" in query_analysis.intents:
            return "Start with who is eligible, then list conditions, exclusions, or exceptions in bullets." + detail_suffix
        if "documents_required" in query_analysis.intents:
            return (
                "Start with the required documents, then use short bullets. Include timing only if clearly stated."
                + detail_suffix
            )
        if "checklist" in query_analysis.intents or "process" in query_analysis.intents:
            return (
                "Start with a one-line answer, then use short action bullets in the order the user should follow."
                + detail_suffix
            )
        if "timeline" in query_analysis.intents:
            return "Lead with the timing or deadline, then list supporting conditions in bullets." + detail_suffix
        return "Start with a direct answer. Use short bullets only when they make the answer clearer." + detail_suffix

    def _is_answerable(
        self,
        query_analysis: QueryAnalysis,
        top_docs: list[tuple],
        top_sections: list[tuple],
    ) -> bool:
        if "document_lookup" in query_analysis.intents and top_docs:
            best_document = top_docs[0][0]
            if self.corpus.document_lookup_score(query_analysis, best_document) >= 0.55:
                return True

        if not top_sections:
            return False

        best_section, best_score = top_sections[0]
        best_document = self.documents[best_section.doc_id]
        if best_score < self.config.answerability_min_section_score:
            return False

        # Fix #6: if retrieval score is high enough, trust semantic
        # retrieval even when lexical term overlap is weak (user may use
        # synonyms or different phrasing).
        if best_score >= self.config.answerability_high_confidence_score:
            return True

        # Also trust strong topic/metadata alignment as an alternative
        # to exact term match in evidence blocks.
        section_topic_alignment = self.corpus.topic_alignment_score(query_analysis.topic_hints, best_section.metadata_tags)
        document_topic_alignment = self.corpus.topic_alignment_score(query_analysis.topic_hints, best_document.metadata_tags)
        if max(section_topic_alignment, document_topic_alignment) >= 0.55:
            return True

        support_match_count = 0
        supporting_doc_ids: set[str] = set()
        exact_evidence_match_count = 0
        for section, _ in top_sections:
            evidence_blocks = self.corpus.extract_answer_evidence_blocks(section, query_analysis, limit=2)
            if self._evidence_blocks_support_query(evidence_blocks, query_analysis):
                support_match_count += 1
                supporting_doc_ids.add(section.doc_id)
            if query_analysis.exact_match_expected:
                exact_evidence_match_count += self.corpus.precise_evidence_match_count(section, query_analysis)

        if support_match_count < self.config.answerability_min_support_matches:
            return False

        if query_analysis.exact_match_expected:
            if exact_evidence_match_count < self.config.answerability_min_exact_evidence_matches:
                return False

        if query_analysis.multi_doc_expected and len({document.doc_id for document, _ in top_docs}) > 1:
            if len(supporting_doc_ids) < self.config.answerability_min_support_matches_multi_doc:
                return False

        section_topic_alignment = self.corpus.topic_alignment_score(query_analysis.topic_hints, best_section.metadata_tags)
        document_topic_alignment = self.corpus.topic_alignment_score(query_analysis.topic_hints, best_document.metadata_tags)
        if query_analysis.topic_hints and max(section_topic_alignment, document_topic_alignment) == 0.0:
            # Allow this only if the lexical/title signals were still strong enough to put the section on top.
            title_overlap = len(
                set(best_section.title_terms + best_document.title_terms).intersection(
                    query_analysis.expanded_terms or query_analysis.focus_terms
                )
            )
            if title_overlap == 0:
                return False

        return True

    def _build_unanswerable_response(self, query_analysis: QueryAnalysis, top_docs) -> str:
        if top_docs:
            suggested_title = top_docs[0][0].title
            if query_analysis.topic_hints:
                topic_text = ", ".join(topic.replace("_", " ") for topic in query_analysis.topic_hints)
                return (
                    f"I couldn't find a clear statement for this in the indexed evidence. "
                    f"The closest policy area I found was {topic_text}, but the retrieved text does not clearly answer it. "
                    f"The nearest document was: {suggested_title}."
                )
            return (
                "I couldn't find a clear statement for this in the indexed evidence. "
                f"The nearest document was: {suggested_title}."
            )
        return "I couldn't find a clear statement for this in the indexed evidence."

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
    def _truncate_context_text(text: str, limit: int) -> str:
        compact = re.sub(r"\s+", " ", (text or "").strip())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    @staticmethod
    def _evidence_blocks_support_query(evidence_blocks: list[str], query_analysis: QueryAnalysis) -> bool:
        if not evidence_blocks:
            return False

        query_terms = unique_preserving_order(
            query_analysis.focus_terms if query_analysis.exact_match_expected else (query_analysis.expanded_terms or query_analysis.focus_terms)
        )
        for block in evidence_blocks:
            normalized_block = re.sub(r"\s+", " ", block).casefold()
            if any(term.replace("_", " ") in normalized_block for term in query_terms):
                return True
        return False

    @staticmethod
    def _document_context_limit(query_analysis: QueryAnalysis) -> int:
        if query_analysis.multi_doc_expected:
            return 4
        if query_analysis.exact_match_expected:
            return 2
        return 3

    @staticmethod
    def _derive_thread_title(text: str, limit: int = 56) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return "New chat"
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3].rstrip() + "..."

    def _build_retrieval_query(self, thread, query_analysis: QueryAnalysis) -> str:
        parts: list[str] = []
        if query_analysis.context_dependent and thread.conversation_summary:
            parts.append(f"Conversation summary: {thread.conversation_summary}")
        if query_analysis.context_dependent and thread.current_topic:
            parts.append(f"Current topic: {thread.current_topic}")
        if query_analysis.context_dependent and thread.active_doc_ids:
            active_titles = [self.documents[doc_id].title for doc_id in thread.active_doc_ids if doc_id in self.documents]
            if active_titles:
                parts.append(f"Current documents: {', '.join(active_titles)}")
        if query_analysis.topic_hints:
            parts.append(f"Inferred policy topics: {', '.join(query_analysis.topic_hints)}")
        if query_analysis.intents:
            parts.append(f"Inferred user intent: {', '.join(query_analysis.intents)}")
        if query_analysis.expanded_terms:
            parts.append(f"Expanded terms: {', '.join(query_analysis.expanded_terms[:24])}")
        parts.append(query_analysis.canonical_question)
        return "\n".join(parts)

    def _merge_retrieved_documents(self, top_docs: list[tuple], top_sections: list[tuple]) -> list[tuple]:
        doc_scores: dict[str, float] = {
            document.doc_id: score
            for document, score in top_docs
        }
        for section, score in top_sections:
            if section.doc_id not in self.documents:
                continue
            doc_scores[section.doc_id] = max(doc_scores.get(section.doc_id, score), score)

        ranked_doc_ids = sorted(doc_scores, key=lambda doc_id: doc_scores[doc_id], reverse=True)
        return [
            (self.documents[doc_id], doc_scores[doc_id])
            for doc_id in ranked_doc_ids
            if doc_id in self.documents
        ]

    def _build_answer_context(
        self,
        thread,
        query_analysis: QueryAnalysis,
        top_docs,
        top_sections,
    ) -> str:
        context_guidance = (
            "Treat recent chat as active context for resolving shorthand follow-ups."
            if query_analysis.context_dependent
            else "Treat recent chat as background only. If the current retrieved evidence points to a different document than the prior turn, follow the current evidence."
        )
        recent_chat = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in thread.recent_messages[-self.config.max_recent_messages :]
        ) or "None"

        doc_context_parts = []
        for document, score in top_docs[: self._document_context_limit(query_analysis)]:
            orientation_summary = self._truncate_context_text(
                self.redactor.unmask_text(document.summary),
                self.config.answer_context_doc_summary_char_limit,
            )
            doc_context_parts.append(
                f"[Document: {document.title} | Score: {score:.4f} | File: {document.source_path}]\n"
                f"Metadata: type={document.document_type or 'document'}; "
                f"tags={', '.join(document.metadata_tags) or 'none'}; "
                f"audience={', '.join(document.audiences) or 'none'}; "
                f"version={document.version or 'unknown'}; "
                f"effective_date={document.effective_date or 'unknown'}"
                + (
                    ""
                    if query_analysis.exact_match_expected
                    else f"\nDocument orientation:\n{orientation_summary or '(empty)'}"
                )
            )

        section_context_parts = []
        seen_evidence: set[str] = set()
        for section, score in top_sections:
            document = self.documents[section.doc_id]
            raw_evidence_blocks = self.corpus.extract_answer_evidence_blocks(section, query_analysis)

            # Fix #9: deduplicate evidence blocks across sections so the
            # LLM doesn't see the same raw text twice (common when two
            # sections from the same document are retrieved).
            unique_blocks: list[str] = []
            for block in raw_evidence_blocks:
                block_key = re.sub(r"\s+", " ", block).strip().casefold()
                if block_key not in seen_evidence:
                    seen_evidence.add(block_key)
                    unique_blocks.append(block)

            raw_evidence_text = (
                "\n\n".join(
                    f"Raw evidence block {index}:\n{block}"
                    for index, block in enumerate(unique_blocks, start=1)
                )
                or "Raw evidence block 1:\n(no focused raw evidence extracted)"
            )
            section_orientation = self._truncate_context_text(
                self.redactor.unmask_text(section.summary),
                self.config.answer_context_doc_summary_char_limit,
            )
            section_context_parts.append(
                f"[Document: {document.title} | Section: {section.title} | Score: {score:.4f} | File: {section.source_path}]\n"
                f"Section type: {section.section_type or 'general'} | Tags: {', '.join(section.metadata_tags) or 'none'}\n"
                + (
                    ""
                    if query_analysis.exact_match_expected
                    else f"Section orientation:\n{section_orientation or '(empty)'}\n\n"
                )
                + f"Raw policy evidence:\n{raw_evidence_text}"
            )

        return (
            f"Conversation summary:\n{thread.conversation_summary or 'None'}\n\n"
            f"Conversation handling:\n{context_guidance}\n\n"
            f"Recent chat:\n{recent_chat}\n\n"
            f"Question analysis:\n{query_analysis.canonical_question}\n\n"
            f"Answer format guidance:\n{self._answer_format_guidance(query_analysis)}\n\n"
            f"Retrieved section evidence:\n{chr(10).join(section_context_parts)}\n\n"
            f"Retrieved document context:\n{chr(10).join(doc_context_parts) or 'None'}\n\n"
            "Evidence priority:\n"
            "1. Raw policy evidence blocks\n"
            "2. Section orientation summaries\n"
            "3. Document orientation summaries\n\n"
            "Answer the user conversationally, but ground every claim in the provided evidence. "
            "Synthesize the policy into a clean, readable answer instead of copying document structure or wording. "
            "When multiple documents contribute, combine the overlapping evidence and point out any differences instead of flattening them into one rule. "
            "For exact labels, numbers, thresholds, dates, and definitions, cite only what the raw policy evidence blocks clearly support."
        )

    def _write_retrieval_log(
        self,
        thread_id: str,
        user_question: str,
        query_analysis: QueryAnalysis,
        retrieval_query: str,
        top_docs,
        top_sections,
        is_answerable: bool,
        prompt_payload: str,
        final_answer: str,
        sources: list[SourceReference],
    ) -> None:
        log_root = self._resolve_debug_log_dir()
        if log_root is None:
            return

        retrieval_dir = log_root / "retrieval"
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self._safe_log_name(utc_now_iso())
        file_name = f"{timestamp}_{self._safe_log_name(thread_id)}.txt"
        output_path = retrieval_dir / file_name

        lines: list[str] = [
            f"Thread ID: {thread_id}",
            f"User question: {user_question}",
            "",
            "=== Query Analysis ===",
            query_analysis.canonical_question,
            "",
            "=== Retrieval Query ===",
            retrieval_query,
            "",
            "=== Top Documents ===",
        ]

        if top_docs:
            for index, (document, score) in enumerate(top_docs, start=1):
                lines.extend(
                    [
                        f"{index}. {document.title} | score={score:.4f} | file={document.source_path}",
                        f"   type={document.document_type or 'document'} | tags={', '.join(document.metadata_tags) or 'none'} | audience={', '.join(document.audiences) or 'none'}",
                        f"   summary={self.redactor.unmask_text(document.summary).strip() or '(empty)'}",
                    ]
                )
        else:
            lines.append("(none)")

        lines.extend(["", "=== Top Sections ==="])
        if top_sections:
            for index, (section, score) in enumerate(top_sections, start=1):
                document = self.documents[section.doc_id]
                snippets = self.corpus.extract_evidence_snippets(section, query_analysis)
                lines.extend(
                    [
                        f"{index}. {document.title} :: {section.title} | score={score:.4f} | file={section.source_path}",
                        f"   type={section.section_type or 'general'} | tags={', '.join(section.metadata_tags) or 'none'}",
                        f"   summary={self.redactor.unmask_text(section.summary).strip() or '(empty)'}",
                        "   snippets:",
                    ]
                )
                if snippets:
                    lines.extend([f"   - {snippet}" for snippet in snippets])
                else:
                    lines.append("   - (none)")
        else:
            lines.append("(none)")

        lines.extend(
            [
                "",
                "=== Decision ===",
                f"Answerable: {'yes' if is_answerable else 'no'}",
                "",
            ]
        )

        if prompt_payload.strip():
            lines.extend(
                [
                    "=== Prompt Payload ===",
                    prompt_payload,
                    "",
                ]
            )

        lines.extend(
            [
                "=== Final Answer ===",
                final_answer,
                "",
                "=== Sources ===",
            ]
        )
        if sources:
            lines.extend(
                [
                    f"- {source.document_title} :: {source.section_title} | score={source.score:.4f} | file={source.source_path}"
                    for source in sources
                ]
            )
        else:
            lines.append("(none)")

        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _write_query_failure_log(
        self,
        thread_id: str,
        user_question: str,
        query_analysis: QueryAnalysis | None,
        retrieval_query: str,
        top_docs,
        top_sections,
        prompt_payload: str,
        sources: list[SourceReference],
        exc: Exception,
    ) -> None:
        log_root = self._resolve_debug_log_dir()
        if log_root is None:
            return

        failure_dir = log_root / "retrieval_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        output_path = failure_dir / f"{uuid.uuid4()}.txt"
        lines: list[str] = [
            f"Thread ID: {thread_id}",
            f"User question: {user_question}",
            f"Error: {type(exc).__name__}: {exc}",
            "",
        ]

        if query_analysis is not None:
            lines.extend(
                [
                    "=== Query Analysis ===",
                    query_analysis.canonical_question,
                    "",
                ]
            )

        if retrieval_query.strip():
            lines.extend(
                [
                    "=== Retrieval Query ===",
                    retrieval_query,
                    "",
                ]
            )

        lines.append("=== Top Documents ===")
        if top_docs:
            lines.extend(
                [
                    f"- {document.title} | score={score:.4f} | file={document.source_path}"
                    for document, score in top_docs
                ]
            )
        else:
            lines.append("(none)")

        lines.extend(["", "=== Top Sections ==="])
        if top_sections:
            lines.extend(
                [
                    f"- {self.documents[section.doc_id].title} :: {section.title} | score={score:.4f} | file={section.source_path}"
                    for section, score in top_sections
                    if section.doc_id in self.documents
                ]
            )
        else:
            lines.append("(none)")

        if prompt_payload.strip():
            lines.extend(
                [
                    "",
                    "=== Prompt Payload ===",
                    prompt_payload,
                ]
            )

        lines.extend(
            [
                "",
                "=== Sources ===",
            ]
        )
        if sources:
            lines.extend(
                [
                    f"- {source.document_title} :: {source.section_title} | score={source.score:.4f} | file={source.source_path}"
                    for source in sources
                ]
            )
        else:
            lines.append("(none)")

        lines.extend(
            [
                "",
                "=== Traceback ===",
                traceback.format_exc().strip() or f"{type(exc).__name__}: {exc}",
            ]
        )
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

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

    def _refresh_conversation_summary(self, thread) -> str:
        recent_chat = "\n".join(f"{message.role.upper()}: {message.content}" for message in thread.recent_messages)

        # Fix: include which documents and sections were referenced so
        # follow-up questions can retrieve the same context.
        source_context_parts: list[str] = []
        if thread.active_doc_ids:
            active_titles = [
                self.documents[doc_id].title
                for doc_id in thread.active_doc_ids
                if doc_id in self.documents
            ]
            if active_titles:
                source_context_parts.append(f"Documents referenced: {', '.join(active_titles)}")
        if thread.last_answer_sources:
            section_labels = [
                f"{src.document_title} > {src.section_title}"
                for src in thread.last_answer_sources[:6]
            ]
            source_context_parts.append(f"Sections referenced: {', '.join(section_labels)}")
        source_context = "\n".join(source_context_parts)

        system_prompt = (
            "Summarize this conversation for future retrieval and follow-up handling. "
            "Capture the specific document names, section titles, current topic, "
            "key answered points, and unresolved questions. "
            "Be concise and factual."
        )
        user_prompt = f"{recent_chat}\n\n{source_context}" if source_context else recent_chat
        summary = self._llm_text_with_debug_log(
            purpose="conversation_summary",
            system_prompt=self.redactor.mask_text(system_prompt),
            user_prompt=self.redactor.mask_text(user_prompt),
            max_output_tokens=self.config.conversation_summary_max_output_tokens,
        )
        return self.redactor.unmask_text(summary)
