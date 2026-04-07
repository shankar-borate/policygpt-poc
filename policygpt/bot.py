import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from policygpt.config import Config
from policygpt.conversations import ConversationManager
from policygpt.corpus import DocumentCorpus, ProgressCallback
from policygpt.document_links import build_document_view_url
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
from policygpt.services.usage_metrics import LLMUsageTracker


class PolicyGPTBot:
    def __init__(
        self,
        config: Config,
        ai: AIService | None = None,
        usage_tracker: LLMUsageTracker | None = None,
        redactor: Redactor | None = None,
        extractor: FileExtractor | None = None,
        corpus: DocumentCorpus | None = None,
        conversations: ConversationManager | None = None,
    ) -> None:
        self.config = config
        self.usage_tracker = usage_tracker
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
                    section_order_index=section.order_index,
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

            final_answer = self._sanitize_answer_for_user(answer_text.strip())
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

            recent_message_limit = max(0, self.config.max_recent_messages)
            thread.recent_messages.append(Message(role="user", content=user_question))
            thread.recent_messages.append(
                Message(
                    role="assistant",
                    content=self._compact_history_message(answer_text),
                )
            )
            thread.recent_messages = (
                thread.recent_messages[-recent_message_limit:]
                if recent_message_limit
                else []
            )
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
                usage_tracker=self.usage_tracker,
            )

        if self.config.ai_provider == "openai":
            return OpenAIService(
                self.config.chat_model,
                self.config.embedding_model,
                rate_limit_retries=self.config.ai_rate_limit_retries,
                rate_limit_backoff_seconds=self.config.ai_rate_limit_backoff_seconds,
                usage_tracker=self.usage_tracker,
            )

        raise ValueError(f"Unsupported AI provider: {self.config.ai_provider}")

    def _system_prompt(self) -> str:
        if self._uses_open_weight_prompt_profile():
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
                "24. Treat raw policy evidence blocks as the source of truth. Use summaries only for orientation. If a raw evidence block and a summary differ, trust the raw evidence.\n"
                "25. Never show internal IDs or evidence labels in the final answer body. Sources are shown separately."
            )
        return (
            "You are an enterprise document assistant.\n"
            "Answer only from the provided evidence; if unclear, say it is not clearly stated.\n"
            "Use recent chat only for referential follow-ups.\n"
            "Prefer raw evidence blocks over summaries; summaries are orientation only.\n"
            "Ignore off-topic retrieved text.\n"
            "When multiple documents apply, combine them and note differences.\n"
            "State only supported points.\n"
            "Reply in concise Markdown. Lead with the answer. Use short bullets for list, eligibility, process, approval, comparison, and timeline questions.\n"
            "Use plain English. Avoid long quotes and repetition.\n"
            "No Evidence:, Source:, or Reference: lines in the answer body.\n"
            "Never show internal IDs or evidence labels. Sources are shown separately."
        )

    def _uses_open_weight_prompt_profile(self) -> bool:
        config = getattr(self, "config", None)
        if config is None:
            return False
        return config.ai_provider == "bedrock" and config.bedrock_gpt_model_size in {"20b", "120b"}

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
        if "aggregate" in query_analysis.intents:
            return (
                "Start with the direct answer, then list each relevant item in short bullets. "
                "For every bullet, use 'Name - short description' and keep the description to one concise phrase based on the evidence. "
                "Prefer standalone contest names from document titles or sections that explicitly name the contest. "
                "Do not treat comparison-only mentions, incomplete-sentence mentions, or verification notes as main contest items. "
                "Do not output bare names without context."
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
            if not file_name:
                continue
            section_title = self._derive_thread_title(source.section_title, limit=48)
            label = file_name if not section_title or section_title.casefold() == "introduction" else f"{file_name} | {section_title}"
            url = build_document_view_url(
                self.config.public_base_url,
                source_path=source.source_path,
                section_index=source.section_order_index,
                section_title=source.section_title,
            )
            dedupe_key = f"{label}|{url}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            reference_links.append(f"[{label}]({url})")

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

    def _compact_history_message(self, text: str) -> str:
        compact = self._sanitize_answer_for_user((text or "").strip())
        compact = re.sub(r"\n{2,}Reference:\s.*$", "", compact, flags=re.DOTALL | re.IGNORECASE)
        limit = max(0, self.config.recent_chat_message_char_limit)
        if limit:
            return self._truncate_context_text(compact, limit)
        return compact

    @staticmethod
    def _sanitize_answer_for_user(answer: str) -> str:
        cleaned = re.sub(r"\*?\((?:D\d+(?::S\d+)?)\)\*?", "", (answer or "").strip())
        cleaned = re.sub(r"(?m)^\s*\[(?:D\d+(?::S\d+)?)\]\s*", "", cleaned)
        cleaned = re.sub(r"(?m)^(\s*[-*]\s*)?\*{0,2}(?:D\d+(?::S\d+)?)\*{0,2}\s*(?:-|:|—)\s*", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _sanitize_answer_for_user(answer: str) -> str:
        cleaned = (answer or "").strip()
        for pattern in (
            r"<reasoning\b[^>]*>[\s\S]*?</reasoning>",
            r"<thinking\b[^>]*>[\s\S]*?</thinking>",
            r"<think\b[^>]*>[\s\S]*?</think>",
            r"<analysis\b[^>]*>[\s\S]*?</analysis>",
        ):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"</?(?:reasoning|thinking|think|analysis)\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?mi)^\s*(?:evidence|source|sources|reference|references)\s*:.*$", "", cleaned)
        cleaned = re.sub(r"(?mi)^\s*(?:evidence|source|sources|reference|references)\s*$", "", cleaned)
        cleaned = re.sub(r"\*?\((?:(?:D\d+:)?S\d+(?:\.E\d+)?|D\d+(?::S\d+(?:\.E\d+)?)?)\)\*?", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*\[(?:(?:D\d+:)?S\d+(?:\.E\d+)?|D\d+(?::S\d+(?:\.E\d+)?)?)\]\s*", "", cleaned)
        cleaned = re.sub(r"(?m)^(\s*[-*]\s*)?\*{0,2}(?:(?:D\d+:)?S\d+(?:\.E\d+)?|D\d+(?::S\d+(?:\.E\d+)?)?)\*{0,2}\s*(?:-|:)\s*", r"\1", cleaned)
        cleaned = re.sub(r"\b(?:(?:D\d+:)?S\d+(?:\.E\d+)?|D\d+(?::S\d+(?:\.E\d+)?)?)\b", "", cleaned)
        cleaned = re.sub(r"[【\[]\s*[】\]]", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*[-*]\s*$", "", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(r"\n\s*\n(?=[-*]\s)", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _compact_query_brief(query_analysis: QueryAnalysis) -> str:
        lines = [query_analysis.original_question.strip()]
        modes: list[str] = []
        if query_analysis.exact_match_expected:
            modes.append("exact")
        if query_analysis.context_dependent:
            modes.append("follow-up")
        if query_analysis.multi_doc_expected:
            modes.append("multi-doc")
        if "document_lookup" in query_analysis.intents:
            modes.append("lookup")
        if modes:
            lines.append(f"Mode: {', '.join(modes)}")
        if query_analysis.topic_hints:
            lines.append(f"Topics: {', '.join(query_analysis.topic_hints[:4])}")
        return "\n".join(lines)

    def _build_document_aliases(self, top_docs: list[tuple], top_sections: list[tuple]) -> dict[str, str]:
        ordered_doc_ids: list[str] = []
        seen: set[str] = set()
        for document, _ in top_docs:
            if document.doc_id in seen:
                continue
            seen.add(document.doc_id)
            ordered_doc_ids.append(document.doc_id)
        for section, _ in top_sections:
            if section.doc_id in seen:
                continue
            seen.add(section.doc_id)
            ordered_doc_ids.append(section.doc_id)
        return {doc_id: f"D{index}" for index, doc_id in enumerate(ordered_doc_ids, start=1)}

    def _use_minimal_answer_context(self, query_analysis: QueryAnalysis) -> bool:
        return (
            not self._uses_open_weight_prompt_profile()
            and "aggregate" in query_analysis.intents
            and not query_analysis.exact_match_expected
        )

    @staticmethod
    def _needs_current_date_context(query_analysis: QueryAnalysis) -> bool:
        if {"eligibility", "timeline"} & set(query_analysis.intents):
            return True

        normalized_question = query_analysis.normalized_question.casefold()
        relative_date_markers = (
            "today",
            "current",
            "currently",
            "now",
            "as of",
            "right now",
            "at present",
        )
        return any(marker in normalized_question for marker in relative_date_markers)

    @staticmethod
    def _current_date_prompt_line() -> str:
        return (
            f"Current date: {datetime.now().astimezone().date().isoformat()}\n"
            "Interpret relative dates such as today, current, and now against this date."
        )

    @staticmethod
    def _aggregate_section_signal(section) -> int:
        text = " ".join(
            part for part in (
                section.title or "",
                section.summary or "",
                section.raw_text[:500] if section.raw_text else "",
            )
            if part
        ).casefold()

        positive_markers = (
            "contest name",
            "name and purpose",
            "contest identity",
            "contest overview",
            "contest structure",
            "structure summary",
            "contest is named",
            "contest is titled",
            "the contest is named",
            "the contest is titled",
        )
        negative_markers = (
            "incomplete sentence",
            "requiring verification",
            "items requiring verification",
            "higher-of",
            "higher of",
            "qualification rule",
            "allocation",
            "eligibility",
            "audience",
            "persistency",
            "tax",
            "net off all applicable taxes",
        )

        signal = 0
        if any(marker in text for marker in positive_markers):
            signal += 5
        if "component" in text or "components" in text:
            signal += 2
        if section.section_type == "general":
            signal += 1
        if any(marker in text for marker in negative_markers):
            signal -= 6
        return signal

    def _sections_for_answer_context(
        self,
        query_analysis: QueryAnalysis,
        top_sections: list[tuple],
        top_docs: list[tuple] | None = None,
    ) -> list[tuple]:
        if query_analysis.exact_match_expected:
            best_score = top_sections[0][1] if top_sections else 0.0
            score_floor = max(best_score * 0.65, 0.55) if best_score else 0.0
            filtered_sections: list[tuple] = []
            noisy_markers = (
                "requiring verification",
                "not explicitly defined",
                "specifics not provided",
                "incomplete sentence",
            )

            for section, score in top_sections:
                section_text = " ".join(
                    part for part in (
                        section.title or "",
                        section.summary or "",
                    )
                    if part
                ).casefold()
                if score < score_floor and filtered_sections:
                    continue
                if score < (best_score * 0.8 if best_score else 0.0) and any(marker in section_text for marker in noisy_markers):
                    continue
                filtered_sections.append((section, score))

            return filtered_sections or top_sections[:3]

        if "aggregate" not in query_analysis.intents:
            return top_sections

        ranked_sections = sorted(
            top_sections,
            key=lambda item: (self._aggregate_section_signal(item[0]), item[1]),
            reverse=True,
        )

        selected: list[tuple] = []
        seen_section_ids: set[str] = set()
        seen_doc_ids: set[str] = set()

        for section, score in ranked_sections:
            signal = self._aggregate_section_signal(section)
            if signal <= 0:
                continue
            if section.doc_id in seen_doc_ids and signal < 6:
                continue
            selected.append((section, score))
            seen_section_ids.add(section.section_id)
            seen_doc_ids.add(section.doc_id)

        for section, score in ranked_sections:
            if len(selected) >= len(top_sections):
                break
            if section.section_id in seen_section_ids:
                continue
            if self._aggregate_section_signal(section) < 4:
                continue
            selected.append((section, score))
            seen_section_ids.add(section.section_id)

        for document, _ in top_docs or []:
            if document.doc_id in seen_doc_ids:
                continue
            fallback = next(
                (
                    (section, score)
                    for section, score in ranked_sections
                    if section.doc_id == document.doc_id and section.section_id not in seen_section_ids
                ),
                None,
            )
            if fallback is None:
                continue
            selected.append(fallback)
            seen_section_ids.add(fallback[0].section_id)
            seen_doc_ids.add(document.doc_id)

        return selected or top_sections

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

    def _document_context_limit(self, query_analysis: QueryAnalysis) -> int:
        if self._uses_open_weight_prompt_profile():
            if "document_lookup" in query_analysis.intents:
                return 2
            if query_analysis.multi_doc_expected or "aggregate" in query_analysis.intents:
                return 3
            return 1

        if query_analysis.exact_match_expected or query_analysis.context_dependent:
            return 0
        if "document_lookup" in query_analysis.intents:
            return 2
        if query_analysis.multi_doc_expected:
            return 3
        return 1

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
        minimal_context = self._use_minimal_answer_context(query_analysis)
        doc_aliases = self._build_document_aliases(top_docs, top_sections)
        doc_index_lines = [
            f"{alias}={self.documents[doc_id].title}"
            for doc_id, alias in doc_aliases.items()
            if doc_id in self.documents
        ]
        recent_message_limit = max(0, self.config.max_recent_messages)
        recent_chat = "\n".join(
            f"{message.role.upper()}: {self._compact_history_message(message.content)}"
            for message in (
                thread.recent_messages[-recent_message_limit:]
                if recent_message_limit
                else []
            )
        ) or "None"

        doc_context_parts = []
        for document, _ in top_docs[: self._document_context_limit(query_analysis)]:
            doc_alias = doc_aliases.get(document.doc_id, "D?")
            orientation_summary = self._truncate_context_text(
                self.redactor.unmask_text(document.summary),
                self.config.answer_context_doc_summary_char_limit,
            )
            doc_context_lines = [
                f"[{doc_alias}]",
            ]
            if not minimal_context and self.config.include_document_metadata_in_answers:
                doc_context_lines.append(
                    f"Metadata: type={document.document_type or 'document'}; "
                    f"tags={', '.join(document.metadata_tags) or 'none'}; "
                    f"audience={', '.join(document.audiences) or 'none'}; "
                    f"version={document.version or 'unknown'}; "
                    f"effective_date={document.effective_date or 'unknown'}"
                )
            if (
                not minimal_context
                and not query_analysis.exact_match_expected
                and self.config.include_document_orientation_in_answers
            ):
                doc_context_lines.append(f"Summary:\n{orientation_summary or '(empty)'}")
            if len(doc_context_lines) > 1:
                doc_context_parts.append("\n".join(doc_context_lines))

        prompt_sections = self._sections_for_answer_context(query_analysis, top_sections, top_docs)
        section_context_parts = []
        seen_evidence: set[str] = set()
        section_alias_counts: dict[str, int] = {}
        for section, _ in prompt_sections:
            doc_alias = doc_aliases.get(section.doc_id, "D?")
            section_alias_counts[section.doc_id] = section_alias_counts.get(section.doc_id, 0) + 1
            section_alias = f"S{section_alias_counts[section.doc_id]}"
            evidence_tag = f"{doc_alias}:{section_alias}"
            if minimal_context:
                raw_evidence_blocks = self.corpus.extract_evidence_snippets(section, query_analysis, limit=1)
            else:
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
                    f"{evidence_tag}.E{index}\n{block}"
                    for index, block in enumerate(unique_blocks, start=1)
                )
                or f"{evidence_tag}.E1\n(no focused raw evidence extracted)"
            )
            section_orientation = self._truncate_context_text(
                self.redactor.unmask_text(section.summary),
                self.config.answer_context_doc_summary_char_limit,
            )
            section_context_lines = [
                f"[{evidence_tag} {section.title}]",
            ]
            if not minimal_context and self.config.include_section_metadata_in_answers:
                section_context_lines.append(
                    f"Section type: {section.section_type or 'general'} | Tags: {', '.join(section.metadata_tags) or 'none'}"
                )
            if (
                not minimal_context
                and not query_analysis.exact_match_expected
                and self.config.include_section_orientation_in_answers
            ):
                section_context_lines.append(f"Summary:\n{section_orientation or '(empty)'}")
            section_context_lines.append(f"Evidence:\n{raw_evidence_text}")
            section_context_parts.append("\n".join(section_context_lines))

        prompt_parts: list[str] = []
        if thread.conversation_summary.strip():
            prompt_parts.append(f"Summary:\n{thread.conversation_summary.strip()}")
        if recent_chat != "None":
            prompt_parts.append(f"Recent:\n{recent_chat}")
        prompt_parts.append(f"Question:\n{self._compact_query_brief(query_analysis)}")
        if self._needs_current_date_context(query_analysis):
            prompt_parts.append(self._current_date_prompt_line())
        prompt_parts.append(f"Style:\n{self._answer_format_guidance(query_analysis)}")
        if doc_index_lines:
            prompt_parts.append(f"Docs:\n{chr(10).join(doc_index_lines)}")
        prompt_parts.append(f"Evidence:\n{chr(10).join(section_context_parts)}")
        if doc_context_parts and ("document_lookup" in query_analysis.intents or self._uses_open_weight_prompt_profile()):
            prompt_parts.append(f"Doc notes:\n{chr(10).join(doc_context_parts)}")
        prompt_parts.append("Answer only from the evidence. Prefer raw evidence over summaries. Never mention internal IDs.")
        return "\n\n".join(part for part in prompt_parts if part.strip())

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
