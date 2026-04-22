import re
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from policygpt.config import Config
from policygpt.constants.enums import (
    AIProvider,
    BedRockModelSize,
    ConfidenceLevel,
    ConversationalIntent,
    QueryIntent,
)
from policygpt.core.conversations import ConversationManager
from policygpt.core.corpus import DocumentCorpus, ProgressCallback
from policygpt.core.document_links import build_document_open_url
from policygpt.models import ChatResult, Message, SourceReference
from policygpt.models import utc_now_iso
from policygpt.core.ai.base import AIService
from policygpt.core.ai.providers.bedrock_provider import BedrockService
from policygpt.observability.debug_logging import write_llm_debug_log_pair
from policygpt.ingestion.extraction.file_extractor import FileExtractor
from policygpt.core.ai.providers.openai_provider import OpenAIService
from policygpt.core.retrieval.query_analyzer import QueryAnalysis, QueryAnalyzer, detect_conversational_intent
from policygpt.ingestion.extraction.redaction import Redactor
from policygpt.ingestion.extraction.taxonomy import unique_preserving_order
from policygpt.observability.usage_metrics import LLMUsageTracker
from policygpt.config.user_profiles import UserProfile
from policygpt.cache import CacheManager, build_cache_backend


def _log_retrieval_console(top_sections, top_docs, corpus) -> None:
    """Print a concise retrieval summary to stdout for real-time tracking."""
    score_details = getattr(corpus, "last_retrieval_score_details", {})
    doc_count = len(top_docs)
    sec_count = len(top_sections)
    top_score = max((s for _, s in top_sections), default=0.0) if top_sections else 0.0
    print(
        f"  [Retrieval] {sec_count} section(s) from {doc_count} doc(s) | top_final={top_score:.3f}",
        flush=True,
    )
    for sec, final_score in top_sections:
        doc = next((d for d, _ in top_docs if any(s.doc_id == d.doc_id for s, _ in top_sections if s.section_id == sec.section_id)), None)
        doc_title = doc.title if doc else sec.doc_id
        sd = score_details.get(sec.section_id, {})
        matched = ",".join(sd.get("matched_by") or []) or "?"
        llm_part = (
            f"llm={sd['llm_score']:.3f}" if sd.get("llm_score") is not None
            else ("llm=skip" if sd.get("llm_skipped") else "")
        )
        print(
            f"    {final_score:.3f} | {doc_title} :: {sec.title}"
            f" | matched_by={matched}"
            f" kw={sd.get('kw_score',0.0):.2f}"
            f" sim={sd.get('sim_score',0.0):.2f}"
            f" vec={sd.get('vec_score',0.0):.2f}"
            f" os={sd.get('os_hybrid_score',0.0):.3f}"
            f" h={sd.get('h_score',0.0):.3f}"
            + (f" {llm_part}" if llm_part else ""),
            flush=True,
        )


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
        thread_repo=None,
        cache: CacheManager | None = None,
    ) -> None:
        self.config = config
        self.usage_tracker = usage_tracker
        self.redactor = redactor or Redactor(config.ingestion.redaction_rules)
        self.ai = ai or self._build_ai_service()
        self.extractor = extractor or FileExtractor(config)
        self.query_analyzer = QueryAnalyzer(ai=self.ai)
        self.cache = cache or CacheManager(
            backend=build_cache_backend(
                provider=config.cache.cache_provider,
                redis_host=config.cache.redis_host,
                redis_port=config.cache.redis_port,
                redis_db=config.cache.redis_db,
                redis_password=config.cache.redis_password,
                redis_ssl=config.cache.redis_ssl,
                redis_ssl_ca_certs=config.cache.redis_ssl_ca_certs,
                redis_key_prefix=config.cache.redis_key_prefix,
            )
        )
        self.corpus = corpus or DocumentCorpus(
            config=config,
            extractor=self.extractor,
            ai=self.ai,
            redactor=self.redactor,
            cache=self.cache,
        )
        self.conversations = conversations or ConversationManager(repo=thread_repo)
        self._supplementary_facts: str = self._load_supplementary_facts()

    @property
    def documents(self):
        return self.corpus.documents

    @property
    def sections(self):
        return self.corpus.sections

    def _get_doc_for_section(self, section):
        """Return the DocumentRecord for *section*, handling stale doc_ids.

        OpenSearch may return sections whose doc_id was generated in a previous
        ingestion run (different UUID than the current in-memory documents).
        When that happens, fall back to matching on source_path so we can still
        return a sensible document title rather than raising KeyError.
        Returns None only when no match can be found at all.
        """
        doc = self.documents.get(section.doc_id)
        if doc is not None:
            return doc
        # Fallback: scan by normalized source_path.
        norm = section.source_path.lower().replace("\\", "/")
        for d in self.documents.values():
            if d.source_path.lower().replace("\\", "/") == norm:
                return d
        return None

    def _get_doc_title_for_section(self, section) -> str:
        """Return the document title for *section*, with a safe fallback."""
        from pathlib import Path as _Path
        doc = self._get_doc_for_section(section)
        if doc is not None:
            return doc.title
        # Last resort: derive a readable title from the file name.
        return _Path(section.source_path).stem.replace("_", " ")

    def _get_original_source_path_for_section(self, section) -> str:
        """Return original_source_path for *section*'s document (pre-conversion path).

        When a document was converted from PDF/XLSX/etc. to HTML before ingestion,
        this returns the pre-conversion path so reference links open the source file
        rather than the generated HTML.  Falls back to "" when not applicable.
        """
        doc = self._get_doc_for_section(section)
        if doc is not None:
            return getattr(doc, "original_source_path", "") or ""
        return ""

    @property
    def threads(self):
        return self.conversations.threads

    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: ProgressCallback | None = None,
        user_ids: list[str | int] | None = None,
        domain: str = "",
    ) -> None:
        self.corpus.ingest_folder(
            folder_path,
            progress_callback=progress_callback,
            user_ids=user_ids,
            domain=domain,
        )

    def new_thread(self, user_id: str = "") -> str:
        return self.conversations.new_thread(user_id=user_id)

    def reset_thread(self, thread_id: str) -> None:
        self.conversations.reset_thread(thread_id)

    def get_thread(self, thread_id: str):
        return self.conversations.get_thread(thread_id)

    def list_threads(self, user_id: str = ""):
        return self.conversations.list_threads(user_id=user_id)

    def chat(
        self,
        thread_id: str,
        user_question: str,
        user_id: str | int | None = None,
        user_profile: UserProfile | None = None,
    ) -> ChatResult:
        thread = None
        query_analysis = None
        retrieval_query = ""
        top_docs = []
        top_sections = []
        prompt_payload = ""
        sources: list[SourceReference] = []
        try:
            if not self.documents and self.corpus._os_retriever is None:
                raise RuntimeError("No documents ingested. Call ingest_folder() first.")

            thread = self.get_thread(thread_id)

            conversational_intent = detect_conversational_intent(user_question)
            if conversational_intent:
                if conversational_intent == ConversationalIntent.SELF_REFERENTIAL:
                    reply = self._self_referential_reply(thread, user_question)
                else:
                    reply = self._conversational_reply(conversational_intent, user_question)
                thread.display_messages.append(Message(role="user", content=user_question))
                thread.display_messages.append(Message(role="assistant", content=reply))
                thread.updated_at = utc_now_iso()
                return ChatResult(thread_id=thread_id, answer=reply, sources=[], thread=thread)

            # Use recent_messages (always in memory) to detect first turn; display_messages
            # may be empty after OS save even when the thread already has history.
            first_user_message = not any(message.role == "user" for message in thread.recent_messages)
            active_document_titles = [
                self.documents[doc_id].title
                for doc_id in thread.active_doc_ids
                if doc_id in self.documents
            ]
            query_analysis = self.query_analyzer.analyze(
                user_question=user_question,
                active_document_titles=active_document_titles,
                candidate_documents=list(self.documents.values()),
                entity_lookup=self.corpus.entity_lookup,
                user_profile=user_profile,
            )

            retrieval_query = self._build_retrieval_query(thread, query_analysis)
            masked_retrieval_query = self.redactor.mask_text(retrieval_query)
            # Embed only the raw user question so the semantic vector is not
            # diluted by injected metadata labels ("Inferred policy topics:",
            # "Expanded search terms:", etc.).  The canonical_question drives
            # BM25/lexical channels via expanded_terms already; mixing it
            # into the embedding vector hurts cosine similarity against the
            # clean document embeddings.
            # Exception: short context-dependent timeline queries (e.g. "is this
            # open") embed as generic phrases that don't semantically match date
            # sections.  Augment with temporal keywords so the vector points at
            # date/period content rather than unrelated sections.
            query_vec = self._embed_one(self._build_embedding_text(query_analysis))

            # FAQ fast-path: if the question nearly exactly matches a stored
            # FAQ question (cosine ≥ faq_fastpath_min_score), return the FAQ
            # answer directly without running the full RAG pipeline.
            faq_hit = self.corpus.faq_fastpath_lookup(
                query_vec,
                min_score=self.config.retrieval.faq_fastpath_min_score,
                user_id=user_id,
            )
            if faq_hit:
                final_faq = self._normalize_answer_markdown(faq_hit)
                thread.display_messages.append(Message(role="user", content=user_question))
                thread.display_messages.append(Message(role="assistant", content=final_faq))
                thread.recent_messages.append(Message(role="user", content=user_question))
                thread.recent_messages.append(Message(role="assistant", content=self._compact_history_message(final_faq)))
                if first_user_message:
                    thread.title = self._derive_thread_title(user_question)
                thread.updated_at = utc_now_iso()
                return ChatResult(thread_id=thread_id, answer=final_faq, sources=[], thread=thread)
            preferred_doc_ids = thread.active_doc_ids if query_analysis.context_dependent else []
            preferred_section_ids = thread.active_section_ids if query_analysis.context_dependent else []

            # Answer cache — skip for context-dependent follow-ups where the
            # answer may differ based on which doc was last referenced.
            _cache_key: tuple | None = None
            if not query_analysis.context_dependent:
                _norm_q = re.sub(r"\s+", " ", user_question.strip().casefold())
                _cache_key = (_norm_q, frozenset(thread.active_doc_ids))
                _cached = self.cache.get_answer(_norm_q, frozenset(thread.active_doc_ids))
                if _cached is not None:
                    cached_answer, cached_sources = _cached
                    thread.display_messages.append(Message(role="user", content=user_question))
                    thread.display_messages.append(Message(role="assistant", content=cached_answer))
                    thread.recent_messages.append(Message(role="user", content=user_question))
                    thread.recent_messages.append(
                        Message(role="assistant", content=self._compact_history_message(cached_answer))
                    )
                    thread.last_answer_sources = cached_sources
                    thread.updated_at = utc_now_iso()
                    return ChatResult(thread_id=thread_id, answer=cached_answer, sources=cached_sources, thread=thread)

            answer_text = ""
            is_answerable = False

            # Compound question detection — if the user asked two distinct
            # questions in one message, retrieve separately for each and answer
            # them together.  Skip when the question is a simple follow-up or
            # a direct aggregate query (those are handled by the FAQ path).
            sub_questions = self._split_compound_question(user_question)
            is_compound = len(sub_questions) > 1 and not query_analysis.context_dependent

            if is_compound:
                answer_text, sources = self._answer_compound_question(
                    thread=thread,
                    user_question=user_question,
                    sub_questions=sub_questions,
                    active_document_titles=active_document_titles,
                    preferred_doc_ids=preferred_doc_ids,
                    preferred_section_ids=preferred_section_ids,
                    user_id=user_id,
                    user_profile=user_profile,
                )
                top_docs = []
                top_sections = []
                is_answerable = True
                prompt_payload = ""
            else:
                top_sections = self.corpus.retrieve_top_sections(
                    query_vec,
                    query_analysis,
                    preferred_section_ids=preferred_section_ids,
                    user_id=user_id,
                    user_profile=user_profile,
                )
                top_docs = self._merge_retrieved_documents(
                    self._docs_from_sections(top_sections, preferred_doc_ids),
                    top_sections,
                )
                _log_retrieval_console(top_sections, top_docs, self.corpus)

                # ── Full-document injection for describe/explain queries ───────
                # When the query explicitly names a document (high lookup score),
                # replace top_sections with ALL sections from that document so
                # the LLM can produce a complete description rather than
                # reconstructing from scattered top-k chunks.
                if (
                    "document_lookup" in query_analysis.intents
                    and top_docs
                ):
                    best_doc, _ = top_docs[0]
                    lookup_score = self.corpus.document_lookup_score(query_analysis, best_doc)
                    if lookup_score >= self.config.retrieval.document_lookup_score_threshold:
                        full_sections = self.corpus.get_all_sections_for_document(best_doc.doc_id)
                        if full_sections:
                            top_sections = full_sections
                            top_docs = self._merge_retrieved_documents(
                                self._docs_from_sections(top_sections, preferred_doc_ids),
                                top_sections,
                            )

                # One SourceReference per document (highest-scoring section wins).
                # top_sections is already sorted by score descending so the first
                # occurrence of each source_path is the best match.
                _seen_doc_paths: set[str] = set()
                sources = []
                for section, score in top_sections:
                    _key = section.source_path.lower().replace("\\", "/")
                    if _key in _seen_doc_paths:
                        continue
                    _seen_doc_paths.add(_key)
                    sources.append(SourceReference(
                        document_title=self._get_doc_title_for_section(section),
                        section_title=section.title,
                        source_path=section.source_path,
                        score=score,
                        section_order_index=section.order_index,
                        original_source_path=self._get_original_source_path_for_section(section),
                    ))
                # Drop very low-scoring sources that barely contributed — keeps
                # the reference list focused on genuinely relevant sections.
                if len(sources) > 1:
                    best_src_score = sources[0].score
                    min_src = max(self.config.retrieval.answerability_min_section_score, best_src_score * self.config.retrieval.source_score_min_scaling)
                    sources = [s for s in sources if s.score >= min_src] or sources[:1]

                is_answerable = self._is_answerable(query_analysis, top_docs, top_sections)

                # ── Layer 1: keyword-title fallback ───────────────────────────
                # When semantic search fails, retry using focus-term matching
                # against section/document titles and keywords.  This catches
                # cases where the section title ("Deployment Diagram") doesn't
                # embed close to the query ("which ports are used") even though
                # the content is exactly what was asked.
                if not is_answerable:
                    fallback_sections = self._keyword_title_fallback_sections(query_analysis)
                    if fallback_sections:
                        top_sections = fallback_sections
                        top_docs = self._merge_retrieved_documents(
                            self._docs_from_sections(top_sections, preferred_doc_ids),
                            top_sections,
                        )
                        is_answerable = self._is_answerable(query_analysis, top_docs, top_sections)
                        if is_answerable:
                            # Rebuild sources from the fallback sections
                            _seen_doc_paths_fb: set[str] = set()
                            sources = []
                            for section, score in top_sections:
                                _key = section.source_path.lower().replace("\\", "/")
                                if _key in _seen_doc_paths_fb:
                                    continue
                                _seen_doc_paths_fb.add(_key)
                                sources.append(SourceReference(
                                    document_title=self._get_doc_title_for_section(section),
                                    section_title=section.title,
                                    source_path=section.source_path,
                                    score=score,
                                    section_order_index=section.order_index,
                                    original_source_path=self._get_original_source_path_for_section(section),
                                ))

                if not is_answerable:
                    sources = []
                if is_answerable:
                    # ── Feature 2: dual answer when two docs have close scores ─
                    if self._is_dual_answer_candidate(top_docs, top_sections, query_analysis):
                        dual = self._build_dual_answer(
                            thread=thread,
                            query_analysis=query_analysis,
                            top_docs=top_docs,
                            top_sections=top_sections,
                            query_vec=query_vec,
                            user_id=user_id,
                            user_profile=user_profile,
                        )
                        if dual:
                            answer_text = dual
                            # Keep sources from both docs
                        else:
                            # Dual generation failed — fall through to single answer
                            pass

                    if not answer_text:
                        prompt_payload = self._build_answer_context(
                            thread=thread,
                            query_analysis=query_analysis,
                            top_docs=top_docs,
                            top_sections=top_sections,
                            query_vec=query_vec,
                            user_id=user_id,
                        )
                        masked_answer = self._llm_text_with_debug_log(
                            purpose="chat_answer",
                            system_prompt=self.redactor.mask_text(self._system_prompt(user_profile)),
                            user_prompt=self.redactor.mask_text(prompt_payload),
                            max_output_tokens=self.config.output.chat_max_output_tokens,
                        )
                        answer_text = self._normalize_answer_markdown(self.redactor.unmask_text(masked_answer))
                        if not self._check_answer_grounding(answer_text, prompt_payload):
                            answer_text += (
                                "\n\n_Note: some details in this answer could not be fully verified "
                                "against the retrieved evidence. Please cross-check with the source document._"
                            )

                    confidence = self._compute_confidence(top_sections)
                    if confidence != ConfidenceLevel.HIGH:
                        answer_text += f"\n\n_Confidence: {confidence}_"

                    # ── Feature 1b: follow-up nudge on LOW confidence ──────────
                    if (
                        confidence == ConfidenceLevel.LOW
                        and self.config.retrieval.followup_on_low_confidence
                        and not self._is_dual_answer_candidate(top_docs, top_sections, query_analysis)
                    ):
                        followup = self._generate_low_confidence_followup(
                            user_question, answer_text, query_analysis
                        )
                        if followup:
                            answer_text += f"\n\n{followup}"

                    # Suggest related questions from the FAQ corpus
                    related = self._find_related_questions(query_vec, user_question, user_id=user_id)
                    if related:
                        answer_text += "\n\n**You might also ask:**\n" + "\n".join(f"- {q}" for q in related)
                else:
                    # Pattern pre-filter missed this (e.g. "looks nice.. thanks").
                    # Ask the LLM to classify intent before showing a retrieval
                    # failure — cheap single call, avoids confusing non-policy
                    # messages with "I couldn't find a clear statement".
                    llm_intent = self._llm_classify_intent(user_question)
                    if llm_intent == ConversationalIntent.SELF_REFERENTIAL:
                        answer_text = self._self_referential_reply(thread, user_question)
                    elif llm_intent != ConversationalIntent.POLICY:
                        answer_text = self._conversational_reply(llm_intent, user_question)
                    else:
                        # ── Feature 1a: always ask a clarifying question before
                        # giving up — the bot has docs nearby, ask the user to
                        # narrow down rather than showing a dead-end message.
                        clarifying = self._generate_clarifying_question(user_question, query_analysis, top_docs)
                        answer_text = clarifying if clarifying else self._build_unanswerable_response(
                            query_analysis, top_docs, query_vec=query_vec, user_id=user_id
                        )

            final_answer = self._sanitize_answer_for_user(answer_text.strip())
            # Populate cache for eligible queries (non-context-dependent + answerable)
            if _cache_key is not None and is_answerable:
                self.cache.set_answer(_norm_q, frozenset(thread.active_doc_ids), final_answer, sources)
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

            recent_message_limit = max(0, self.config.conversation.max_recent_messages)
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

            if len(thread.recent_messages) >= self.config.conversation.summarize_after_turns:
                thread.conversation_summary = self._refresh_conversation_summary(thread)

            return ChatResult(
                thread_id=thread_id,
                answer=final_answer,
                sources=thread.last_answer_sources,
                thread=thread,
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

    @staticmethod
    def _build_embedding_text(query_analysis: QueryAnalysis) -> str:
        """Return the text to embed for vector retrieval.

        For most queries this is just the original question.
        For short context-dependent timeline queries (e.g. "is this open",
        "is this still active") the phrase is too generic to semantically
        match date/period sections, so we append temporal expansion terms
        to steer the vector toward contest-period / deadline content.
        """
        q = query_analysis.original_question
        is_short = len(q.split()) <= 6
        is_timeline = "timeline" in query_analysis.intents
        is_ctx = query_analysis.context_dependent
        if is_short and is_timeline and is_ctx:
            return (
                q + " start date end date closing date deadline valid from valid till "
                "effective date expiry open closed active inactive expired ongoing"
            )
        return q

    def _embed_one(self, text: str) -> np.ndarray:
        cached = self.cache.get_embedding(text)
        if cached is not None:
            return cached
        vector = self.corpus.embed_text(text)
        self.cache.set_embedding(text, vector)
        return vector

    def _build_ai_service(self) -> AIService:
        if self.config.ai.ai_provider == AIProvider.BEDROCK:
            return BedrockService(
                chat_model=self.config.ai.chat_model,
                embedding_model=self.config.ai.embedding_model,
                region_name=self.config.ai.bedrock_region,
                rate_limit_retries=self.config.ai.ai_rate_limit_retries,
                rate_limit_backoff_seconds=self.config.ai.ai_rate_limit_backoff_seconds,
                usage_tracker=self.usage_tracker,
            )

        if self.config.ai.ai_provider == AIProvider.OPENAI:
            return OpenAIService(
                self.config.ai.chat_model,
                self.config.ai.embedding_model,
                rate_limit_retries=self.config.ai.ai_rate_limit_retries,
                rate_limit_backoff_seconds=self.config.ai.ai_rate_limit_backoff_seconds,
                usage_tracker=self.usage_tracker,
            )

        raise ValueError(f"Unsupported AI provider: {self.config.ai.ai_provider}")

    def _conversational_reply(self, intent: str, user_message: str = "") -> str:
        """Return a natural LLM-generated reply for non-policy conversational messages.

        Falls back to a canned reply if the LLM call fails so the user always
        gets a response.
        """
        system_prompt = (
            f"You are a warm, friendly policy assistant for {self.config.domain_profile.persona_description}. "
            "The user has sent you a social or conversational message — not a policy question. "
            "Respond naturally and briefly (1-2 sentences max) as a helpful colleague would. "
            "Be warm, human, and in context with what the user said. "
            "Do not lecture them about what you can do unless they specifically asked. "
            "If they seem to be wrapping up, wish them well. "
            "If they expressed an emotion or made a social gesture, acknowledge it naturally. "
            "If the message is unclear or gibberish, respond with light humour and let them know you're ready to help."
        )
        try:
            reply = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=user_message.strip() or intent,
                max_output_tokens=80,
            ).strip()
            if reply:
                return reply
        except Exception:
            pass

        # Canned fallback in case the LLM call fails
        _fallback: dict[str, str] = {
            ConversationalIntent.GREETING: self.config.domain_profile.greeting_reply,
            ConversationalIntent.FAREWELL: "Goodbye! Come back anytime.",
            ConversationalIntent.THANKS:   "You're welcome!",
            ConversationalIntent.IDENTITY: self.config.domain_profile.identity_reply,
        }
        return _fallback.get(intent, "Happy to help — just ask!")

    def _self_referential_reply(self, thread: object, user_question: str) -> str:
        """Handle meta-questions about the bot's own previous answer.

        Looks at the last assistant reply and the user question before it, then
        uses the LLM to explain what was (or wasn't) covered and why, without
        triggering a fresh RAG search that would wrongly say "the documents
        don't explain why I missed something".
        """
        # Extract the most recent Q-A pair from conversation history
        recent = getattr(thread, "recent_messages", [])
        prev_user_q = ""
        prev_bot_answer = ""
        for msg in reversed(recent):
            if not prev_bot_answer and msg.role == "assistant":
                prev_bot_answer = msg.content or ""
            elif prev_bot_answer and not prev_user_q and msg.role == "user":
                prev_user_q = msg.content or ""
                break

        if not prev_bot_answer:
            # No prior context — treat as a clarification request
            return (
                "I don't have a previous answer to refer back to in this conversation. "
                "Could you re-ask the original question so I can give you a complete answer?"
            )

        system_prompt = (
            f"You are a helpful policy assistant for {self.config.domain_profile.persona_description}. "
            "The user is asking a follow-up question about your PREVIOUS answer — they feel you "
            "missed something or didn't cover a point they expected. "
            "Your job is to:\n"
            "1. Acknowledge the gap honestly (don't be defensive).\n"
            "2. If the missing detail can be inferred from your previous answer, explain it now.\n"
            "3. If the detail is simply not in the policy documents you have, say so clearly.\n"
            "4. Keep the response concise (3-5 sentences max).\n"
            "Do NOT say 'the provided documents do not explain why I missed it' — that is unhelpful. "
            "Respond directly to what the user is asking."
        )
        context_block = ""
        if prev_user_q:
            context_block += f"Original question: {prev_user_q}\n\n"
        context_block += f"Your previous answer:\n{prev_bot_answer}\n\nFollow-up: {user_question}"

        try:
            reply = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=context_block,
                max_output_tokens=300,
            ).strip()
            if reply:
                return reply
        except Exception:
            pass

        return (
            "I apologise for missing that. Could you tell me specifically what you were "
            "expecting, and I'll look it up properly?"
        )

    def _llm_classify_intent(self, text: str) -> str:
        """Use a cheap LLM call to classify intent when the pattern pre-filter misses.

        Returns one of: "policy" | "greeting" | "farewell" | "thanks" |
        "identity" | "chitchat" | "self_referential".

        Defaults to "policy" on any failure so real questions are never
        suppressed by a classification error.
        """
        system_prompt = (
            f"You are an intent classifier for a policy Q&A assistant used by {self.config.domain_profile.intent_user_description}. "
            "Classify the user message into exactly one of these categories:\n"
            f"  {ConversationalIntent.POLICY}           — a genuine question about {self.config.domain_profile.intent_policy_description}\n"
            f"  {ConversationalIntent.GREETING}         — hello, hi, good morning, good evening, etc.\n"
            f"  {ConversationalIntent.FAREWELL}         — bye, goodbye, see you, take care, etc.\n"
            f"  {ConversationalIntent.THANKS}           — thank you, thanks, great, nice, looks good, cheers, etc.\n"
            f"  {ConversationalIntent.IDENTITY}         — who are you, what can you do, what is this bot, etc.\n"
            f"  {ConversationalIntent.CHITCHAT}         — any other off-topic or social message that is not a policy question\n"
            f"  {ConversationalIntent.SELF_REFERENTIAL} — asking why the bot missed/forgot/omitted something in its previous answer\n\n"
            "Reply with ONLY the single category word. No punctuation, no explanation."
        )
        user_prompt = f"Message: {text.strip()}"
        try:
            raw = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=self.config.retrieval.intent_classification_max_tokens,
            )
            category = raw.strip().lower().split()[0] if raw.strip() else ConversationalIntent.POLICY
            valid = {i.value for i in ConversationalIntent}
            return category if category in valid else ConversationalIntent.POLICY
        except Exception:
            return ConversationalIntent.POLICY

    def _check_answer_grounding(self, answer: str, evidence_context: str) -> bool:
        """Return True if the answer appears grounded in the evidence.

        Makes a single cheap LLM call asking the model to verify that every
        factual claim in the answer is supported by the provided evidence.
        Returns True (grounded) on any failure so the guard never suppresses
        a valid answer due to an LLM error.
        """
        if not self.config.retrieval.grounding_guard_enabled:
            return True
        if not answer.strip() or not evidence_context.strip():
            return True
        # Skip the expensive LLM check for answers that contain no verifiable
        # numeric or date claims — pure narrative answers can't be "ungrounded"
        # in the sense of wrong numbers.
        if not re.search(r"\d", answer[:600]):
            return True
        system_prompt = (
            "You are a factual grounding checker. "
            "Given an answer and the evidence it was derived from, decide whether "
            "every specific factual claim in the answer (numbers, names, dates, "
            "thresholds, eligibility rules) is directly supported by the evidence. "
            "Reply with exactly one word: GROUNDED or UNGROUNDED."
        )
        user_prompt = (
            f"Evidence:\n{evidence_context[:3000]}\n\n"
            f"Answer:\n{answer[:1500]}\n\n"
            "Is every factual claim in the answer supported by the evidence above?"
        )
        try:
            raw = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=self.config.retrieval.grounding_guard_max_output_tokens,
            )
            return "ungrounded" not in raw.strip().lower()
        except Exception:
            return True

    # Patterns that signal the user asked two or more distinct questions.
    _COMPOUND_SPLITTERS = re.compile(
        r"\band\s+(what|who|when|where|how|which|is|are|can|does|do|will)\b"
        r"|\balso\s+(what|who|when|where|how|which|is|are|can|does|do|will)\b"
        r"|\?[^?]+\?",  # two or more question marks
        re.IGNORECASE,
    )

    @classmethod
    def _split_compound_question(cls, text: str) -> list[str]:
        """Return sub-questions if *text* contains a compound question, else [text].

        Splits on "... and what/who/how/..." or on multiple "?" marks. Returns
        the original question unchanged when no compound pattern is detected or
        when the question is too short to split meaningfully.
        """
        text = text.strip()
        # Two or more explicit question marks → split there.
        parts_by_qmark = [p.strip() for p in text.split("?") if p.strip()]
        if len(parts_by_qmark) >= 2:
            return [p + "?" for p in parts_by_qmark]

        # Detect "and what/who/how/..." mid-sentence connector.
        match = cls._COMPOUND_SPLITTERS.search(text)
        if match:
            split_pos = match.start()
            first = text[:split_pos].strip().rstrip(",;")
            second = text[split_pos:].strip().lstrip("and ").lstrip("also ").strip()
            if first and second and len(first) >= 12 and len(second) >= 12:
                return [first + "?", second + "?"]

        return [text]

    def _is_dual_answer_candidate(
        self,
        top_docs: list,
        top_sections: list,
        query_analysis: QueryAnalysis,
    ) -> bool:
        """Return True when two distinct documents have close enough scores to
        warrant showing both answers and letting the user choose.

        Skipped for: multi-doc queries (already expects many sources), exact-match
        queries (single clear answer expected), context-dependent follow-ups.
        """
        if not self.config.retrieval.dual_answer_enabled:
            return False
        if query_analysis.multi_doc_expected or query_analysis.context_dependent:
            return False
        if query_analysis.exact_match_expected:
            return False
        if len(top_docs) < 2:
            return False

        score_a = top_docs[0][1]
        score_b = top_docs[1][1]
        if score_a <= 0:
            return False
        # Only trigger when both docs are genuinely good (not just both mediocre)
        if score_a < self.config.retrieval.confidence_medium_score:
            return False
        return (score_b / score_a) >= self.config.retrieval.dual_answer_score_ratio

    def _build_dual_answer(
        self,
        thread,
        query_analysis: QueryAnalysis,
        top_docs: list,
        top_sections: list,
        query_vec,
        user_id,
        user_profile: "UserProfile | None" = None,
    ) -> str:
        """Generate two answers from the top two documents and present them for
        the user to choose the more relevant one.

        Each answer is generated from the sections belonging to its document only.
        """
        doc_a, _ = top_docs[0]
        doc_b, _ = top_docs[1]

        def _sections_for_doc(doc_id: str) -> list[tuple]:
            return [(s, sc) for s, sc in top_sections if s.doc_id == doc_id]

        sections_a = _sections_for_doc(doc_a.doc_id) or top_sections[:3]
        sections_b = _sections_for_doc(doc_b.doc_id) or top_sections[3:6]

        def _answer_from_sections(doc, secs) -> str:
            if not secs:
                return ""
            docs_for_context = [(doc, 1.0)]
            payload = self._build_answer_context(
                thread=thread,
                query_analysis=query_analysis,
                top_docs=docs_for_context,
                top_sections=secs,
                query_vec=query_vec,
                user_id=user_id,
            )
            try:
                raw = self._llm_text_with_debug_log(
                    purpose="dual_answer",
                    system_prompt=self.redactor.mask_text(self._system_prompt(user_profile)),
                    user_prompt=self.redactor.mask_text(payload),
                    max_output_tokens=self.config.output.chat_max_output_tokens,
                )
                return self._normalize_answer_markdown(self.redactor.unmask_text(raw))
            except Exception:
                return ""

        answer_a = _answer_from_sections(doc_a, sections_a)
        answer_b = _answer_from_sections(doc_b, sections_b)

        if not answer_a and not answer_b:
            return ""
        if not answer_b:
            return answer_a
        if not answer_a:
            return answer_b

        return (
            "I found relevant information in two different documents. "
            "Which of these better answers your question?\n\n"
            f"**Option A — {doc_a.title}:**\n{answer_a}\n\n"
            f"**Option B — {doc_b.title}:**\n{answer_b}"
        )

    def _answer_compound_question(
        self,
        thread,
        user_question: str,
        sub_questions: list[str],
        active_document_titles: list[str],
        preferred_doc_ids: list[str],
        preferred_section_ids: list[str],
        user_id: str | int | None = None,
        user_profile=None,
    ) -> tuple[str, list[SourceReference]]:
        """Retrieve evidence for each sub-question separately and answer together.

        Returns (combined_answer_text, merged_sources).
        """
        sub_contexts: list[str] = []
        all_sources: list[SourceReference] = []
        seen_section_ids: set[str] = set()
        seen_doc_paths: set[str] = set()

        for sub_q in sub_questions:
            sub_analysis = self.query_analyzer.analyze(
                user_question=sub_q,
                active_document_titles=active_document_titles,
                candidate_documents=list(self.documents.values()),
                entity_lookup=self.corpus.entity_lookup,
            )
            sub_vec = self._embed_one(sub_q)
            sub_sections = self.corpus.retrieve_top_sections(
                sub_vec,
                sub_analysis,
                preferred_section_ids=preferred_section_ids,
                user_id=user_id,
                user_profile=user_profile,
            )
            sub_docs = self._docs_from_sections(sub_sections, preferred_doc_ids)
            for section, score in sub_sections:
                if section.section_id in seen_section_ids:
                    continue
                seen_section_ids.add(section.section_id)
                _doc_key = section.source_path.lower().replace("\\", "/")
                if _doc_key in seen_doc_paths:
                    continue
                seen_doc_paths.add(_doc_key)
                all_sources.append(SourceReference(
                    document_title=self._get_doc_title_for_section(section),
                    section_title=section.title,
                    source_path=section.source_path,
                    score=score,
                    section_order_index=section.order_index,
                    original_source_path=self._get_original_source_path_for_section(section),
                ))
            sub_context = self._build_answer_context(
                thread=thread,
                query_analysis=sub_analysis,
                top_docs=sub_docs,
                top_sections=sub_sections,
                user_id=user_id,
            )
            sub_contexts.append(f"Sub-question: {sub_q}\n{sub_context}")

        combined_payload = (
            f"The user asked a compound question with {len(sub_questions)} parts. "
            "Answer each part in sequence, clearly labelled.\n\n"
            + "\n\n---\n\n".join(sub_contexts)
        )
        masked = self._llm_text_with_debug_log(
            purpose="chat_answer",
            system_prompt=self.redactor.mask_text(self._system_prompt()),
            user_prompt=self.redactor.mask_text(combined_payload),
            max_output_tokens=self.config.output.chat_max_output_tokens,
        )
        answer = self._normalize_answer_markdown(self.redactor.unmask_text(masked))
        return answer, all_sources

    def _compute_confidence(self, top_sections: list[tuple]) -> ConfidenceLevel:
        """Return HIGH / MEDIUM / LOW based on the best section retrieval score."""
        if not top_sections:
            return ConfidenceLevel.LOW
        best_score = top_sections[0][1]
        if best_score >= self.config.retrieval.confidence_high_score:
            return ConfidenceLevel.HIGH
        if best_score >= self.config.retrieval.confidence_medium_score:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _find_related_questions(
        self,
        query_vec,
        asked_question: str,
        top_k: int = 3,
        user_id: str | int | None = None,
    ) -> list[str]:
        """Return up to top_k FAQ questions related to the current query.

        Filters out the question that was just asked so suggestions are
        genuinely new.  Returns an empty list when FAQ embeddings are absent
        or when no related questions score above a basic threshold.
        """
        candidates = self.corpus.search_faq_questions(query_vec, top_k=top_k + 5, user_id=user_id)
        asked_lower = asked_question.strip().casefold()
        seen: set[str] = set()
        related: list[str] = []
        for score, q, _, _ in candidates:
            if score < self.config.retrieval.related_questions_min_score:
                break  # Results are sorted descending — stop when score drops off
            q_lower = q.strip().casefold()
            if q_lower == asked_lower or q_lower in seen:
                continue
            seen.add(q_lower)
            related.append(q)
            if len(related) >= top_k:
                break
        return related

    def _generate_clarifying_question(
        self,
        user_question: str,
        query_analysis: QueryAnalysis,
        top_docs: list,
    ) -> str:
        """Ask ONE clarifying question when retrieval fails to find an answer.

        Always attempts a clarifying question when there are candidate documents
        nearby — the question is the most conversational response to a failed
        lookup.  Falls back to the unanswerable response only when no documents
        exist to reference.

        Returns the clarifying question string, or empty string when no docs
        are available to anchor the question.
        """
        if not top_docs:
            return ""
        doc_titles = [doc.title for doc, _ in top_docs[:3]]
        system_prompt = (
            "You are a helpful policy assistant. The user asked a question but the "
            "answer was not found clearly in the available documents. "
            "Ask ONE short, specific clarifying question that would help narrow down "
            "the answer — for example, asking which document, which role, which scenario, "
            "or which specific aspect they mean. "
            "Reference the available documents by name if that helps focus the question. "
            "Do not explain why you are asking — just ask naturally."
        )
        user_prompt = (
            f"User question: {user_question}\n"
            f"Documents available: {', '.join(doc_titles)}\n\n"
            "Ask one clarifying question."
        )
        try:
            reply = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=self.config.retrieval.clarifying_question_max_tokens,
            ).strip()
            return reply if reply else ""
        except Exception:
            return ""

    def _generate_low_confidence_followup(
        self,
        user_question: str,
        answer_text: str,
        query_analysis: QueryAnalysis,
    ) -> str:
        """Generate a short follow-up nudge appended to a LOW-confidence answer.

        Invites the user to add context that would help produce a sharper answer.
        Returns empty string on failure so the caller can skip it silently.
        """
        system_prompt = (
            "You are a helpful policy assistant. You have just given an answer but "
            "you are not fully confident it covers what the user needs. "
            "Write ONE short follow-up question (one sentence) that invites the user "
            "to provide more context — for example: which department, which role, "
            "which scenario, or whether a specific document is in scope. "
            "Do NOT repeat the answer. Do NOT explain. Just ask the question naturally."
        )
        topic_hint = (
            f"Topic area: {', '.join(query_analysis.topic_hints)}\n" if query_analysis.topic_hints else ""
        )
        user_prompt = (
            f"User question: {user_question}\n"
            f"{topic_hint}"
            f"Answer given: {answer_text[:400]}\n\n"
            "Write one follow-up question."
        )
        try:
            reply = self.ai.llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=60,
            ).strip()
            return reply if reply else ""
        except Exception:
            return ""

    def _load_supplementary_facts(self) -> str:
        """Load supplementary facts from file if configured and present.

        Content is injected verbatim into every LLM prompt as background
        context.  It is never returned to the user as a source or citation.
        Returns empty string when the file is absent or unreadable.
        """
        facts_path = (self.config.storage.supplementary_facts_file or "").strip()
        if not facts_path:
            return ""
        path = Path(facts_path)
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def reload_supplementary_facts(self) -> None:
        """Re-read the supplementary facts file without restarting the bot.

        Call this if the file is updated while the bot is running.
        """
        self._supplementary_facts = self._load_supplementary_facts()

    def _system_prompt(self, user_profile: "UserProfile | None" = None) -> str:
        domain = self.config.domain_context
        profile_line = ""
        if user_profile and not user_profile.is_empty():
            ctx = user_profile.context_line()
            profile_line = (
                f"\nUser context: {ctx}\n"
                "When the retrieved evidence contains role-specific or grade-specific rules, "
                "prioritise the rules that apply to this user's role and context. "
                "If a rule applies to all users, state it normally."
            )
        if self._uses_open_weight_prompt_profile():
            return (
                f"Domain: {domain}\n"
                f"{profile_line}\n"
                "You are a conversational assistant helping users in this domain "
                f"find answers from their {self.config.domain_profile.doc_type_label}.\n"
                "Rules:\n"
                "1. Answer only from the provided document evidence.\n"
                "2. If the answer is not clearly present, say that it is not clearly stated in the provided documents.\n"
                "3. Be conversational but precise.\n"
                "4. Use current conversation context only when the user's wording is clearly referential, such as 'what about this', 'same policy', or 'that section'.\n"
                "5. When relevant evidence comes from multiple documents or sections, synthesize across them and call out any document-specific differences clearly.\n"
                "6. Mention section titles and file names when useful.\n"
                "7. Do not hallucinate.\n"
                "8. Format the answer in clean Markdown with short sections or bullets when helpful.\n"
                "9. Give a complete, descriptive answer that covers all relevant points found in the evidence. Do not truncate or summarise away useful detail.\n"
                "10. Open with the answer itself in the first sentence — no label, no preamble — then elaborate with supporting detail, context, and steps from the evidence.\n"
                "11. Do not add unnecessary background, repetition, or long caveats that are not in the evidence.\n"
                "12. Never prefix a sentence with labels like 'Direct answer:', 'Summary:', 'Answer:' or similar headings.\n"
                "13. Ignore retrieved text that is about a different policy or a different topic than the user's question, even if it appears semantically similar.\n"
                "14. If the evidence does not explicitly support the asked point, say it is not clearly stated instead of inferring.\n"
                "15. Rewrite policy language into plain, user-friendly business English. Do not paste long policy wording back to the user.\n"
                "16. For list, eligibility, process, approval, and comparison questions, prefer short bullets with bold labels instead of long paragraphs. When the user explicitly asks for a table or tabular format, output a proper HTML table using <table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody></table> — never use Markdown pipe syntax for tables as it does not render correctly.\n"
                "17. Avoid horizontal rules, deep heading hierarchies, raw policy numbering, and document-style formatting.\n"
                "18. Do not add a separate source/citation section in the answer body. References are added separately.\n"
                "19. Treat the question analysis as a retrieval aid, but only state things that are clearly supported by the evidence snippets.\n"
                "20. If the evidence covers only part of the answer, answer only that part and say what is not clearly stated.\n"
                "21. Prefer evidence snippets over broad summaries when they conflict.\n"
                "21a. Never introduce specific named entities, numbers, dates, versions, technology names, product names, or vendor names that do not appear in the evidence — even if you believe them to be true from your training data. You may freely elaborate, explain, and synthesise what IS in the evidence using your own words, but every concrete claim must be traceable to the evidence.\n"
                "22. When the user asks for a checklist, process, approval path, or timeline, present it in a scannable format.\n"
                "23. If recent chat suggests one document but the current retrieved evidence explicitly defines the asked term in another document, follow the current evidence and briefly note the difference if needed.\n"
                "24. Treat raw policy evidence blocks as the source of truth. Use summaries only for orientation. If a raw evidence block and a summary differ, trust the raw evidence.\n"
                "25. Never show internal IDs or evidence labels in the final answer body. Sources are shown separately.\n"
                "26. Never use the pipe character | anywhere in the response — not in tables, not as a text separator, not in any context. Use HTML tables for tabular data and commas or bullets for inline lists."
            )
        return (
            f"Domain: {domain}\n"
            f"{profile_line}\n"
            "You are a document assistant helping users in this domain find answers from their policy documents.\n"
            "Answer only from the provided evidence; if unclear, say it is not clearly stated.\n"
            "Never introduce specific named entities, numbers, dates, versions, technology names, product names, or vendor names that do not appear in the evidence — even if you believe them to be true from training. You may elaborate and explain using your own words, but every concrete claim must be traceable to the evidence.\n"
            "Give a complete, descriptive answer covering all relevant points in the evidence. Open with the answer itself in the first sentence, then elaborate with supporting detail and context. Do not add labels like 'Direct answer:' or 'Summary:'.\n"
            "Use recent chat only for referential follow-ups.\n"
            "Prefer raw evidence blocks over summaries; summaries are orientation only.\n"
            "Ignore off-topic retrieved text.\n"
            "When multiple documents apply, combine them and note differences.\n"
            "Reply in clear Markdown. Lead with the answer. Use bullets for list, eligibility, process, approval, comparison, and timeline questions.\n"
            "Use plain English. Avoid long quotes and repetition.\n"
            "No Evidence:, Source:, or Reference: lines in the answer body.\n"
            "Never show internal IDs or evidence labels. Sources are shown separately.\n"
            "Never use the pipe character | anywhere — not in tables, not as a separator. Use HTML tables for tabular data."
        )

    def _uses_open_weight_prompt_profile(self) -> bool:
        config = getattr(self, "config", None)
        if config is None:
            return False
        return (
            config.ai.ai_provider == AIProvider.BEDROCK
            and config.ai.bedrock_gpt_model_size in {BedRockModelSize.SMALL, BedRockModelSize.LARGE}
        )

    # Keywords that signal the user wants output as a table.
    _TABLE_REQUEST_PHRASES: tuple[str, ...] = (
        "in table",
        "in a table",
        "as a table",
        "show table",
        "show in table",
        "show me table",
        "show me in table",
        "display table",
        "display as table",
        "tabular",
        "table format",
        "in tabular",
    )

    @staticmethod
    def _user_wants_table(query_analysis: QueryAnalysis) -> bool:
        normalized = query_analysis.normalized_question
        return any(phrase in normalized for phrase in PolicyGPTBot._TABLE_REQUEST_PHRASES)

    def _answer_format_guidance(self, query_analysis: QueryAnalysis) -> str:
        detail_suffix = (
            " Expand slightly with supporting details because the user explicitly asked for more detail."
            if query_analysis.detail_requested
            else " Keep it tight and focused."
        )

        # Structured format for explicitly detailed questions — avoids walls of text
        if query_analysis.detail_requested and not self._user_wants_table(query_analysis):
            return (
                "The user has asked for a detailed explanation. "
                "Structure your answer as follows (skip any section that has no relevant content): "
                "1. **Summary** — one or two sentences answering the question directly. "
                "2. **Details** — bullet points with exact values, thresholds, role names, and conditions from the evidence. "
                "3. **Exceptions / Conditions** — any exclusions, disqualifications, or special rules if stated in the evidence. "
                "Keep each section tight. Lead with facts, not preamble."
            )

        if self._user_wants_table(query_analysis):
            return (
                "The user explicitly asked for a table. "
                "Respond using a proper HTML table — not Markdown pipe syntax. "
                "Structure: <table><thead><tr><th>Col</th>...</tr></thead><tbody><tr><td>val</td>...</tr></tbody></table>. "
                "Choose column headers that match the data (e.g. Name, Eligibility, Reward, Threshold, Amount). "
                "Keep each cell concise — one line of text per cell, no bullet lists or newlines inside cells. "
                "If the data has multiple logical groups, output one <table> per group with a short <b>heading</b> above it. "
                "Do not mix Markdown pipe rows with HTML tables. "
                "Do not add any extra Markdown decoration around the table."
                + detail_suffix
            )

        if QueryIntent.COMPARISON in query_analysis.intents:
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
        if QueryIntent.AGGREGATE in query_analysis.intents:
            return (
                "Open with the answer itself, then list each relevant item in short bullets. "
                "For every bullet, use 'Name - short description' and keep the description to one concise phrase based on the evidence. "
                + self.config.domain_profile.aggregate_response_hint
                + detail_suffix
            )
        if query_analysis.multi_doc_expected:
            return (
                "Open with the answer itself, then cover the relevant documents in short bullets. "
                "Call out differences or gaps across documents explicitly."
                + detail_suffix
            )
        if QueryIntent.DOCUMENT_LOOKUP in query_analysis.intents:
            return (
                "Start by naming the closest matching document. Then give a short, user-friendly summary of what it covers. "
                "If the user appears to be asking only to locate the policy, keep the summary brief and point them to the reference."
                + detail_suffix
            )
        if QueryIntent.APPROVAL in query_analysis.intents:
            return (
                "Start with who approves, then use short bullets with bold labels for each approval path or threshold."
                + detail_suffix
            )
        if QueryIntent.ELIGIBILITY in query_analysis.intents:
            return "Start with who is eligible, then list conditions, exclusions, or exceptions in bullets." + detail_suffix
        if QueryIntent.DOCUMENTS_REQUIRED in query_analysis.intents:
            return (
                "Start with the required documents, then use short bullets. Include timing only if clearly stated."
                + detail_suffix
            )
        if QueryIntent.CHECKLIST in query_analysis.intents or QueryIntent.PROCESS in query_analysis.intents:
            return (
                "Start with a one-line answer, then use short action bullets in the order the user should follow."
                + detail_suffix
            )
        if QueryIntent.TIMELINE in query_analysis.intents:
            return "Lead with the timing or deadline, then list supporting conditions in bullets." + detail_suffix
        return "Open with the answer itself in the first sentence. Use short bullets only when they make the answer clearer." + detail_suffix

    def _is_answerable(
        self,
        query_analysis: QueryAnalysis,
        top_docs: list[tuple],
        top_sections: list[tuple],
    ) -> bool:
        if QueryIntent.DOCUMENT_LOOKUP in query_analysis.intents and top_docs:
            best_document = top_docs[0][0]
            if self.corpus.document_lookup_score(query_analysis, best_document) >= self.config.retrieval.document_lookup_score_threshold:
                return True

        if not top_sections:
            return False

        best_section, best_score = top_sections[0]
        best_document = self._get_doc_for_section(best_section)
        if best_score < self.config.retrieval.answerability_min_section_score:
            return False

        # Fix #6: if retrieval score is high enough, trust semantic
        # retrieval even when lexical term overlap is weak (user may use
        # synonyms or different phrasing).
        if best_score >= self.config.retrieval.answerability_high_confidence_score:
            return True

        # Also trust strong topic/metadata alignment as an alternative
        # to exact term match in evidence blocks.
        section_topic_alignment = self.corpus.topic_alignment_score(query_analysis.topic_hints, best_section.metadata_tags)
        document_topic_alignment = self.corpus.topic_alignment_score(query_analysis.topic_hints, best_document.metadata_tags) if best_document else 0.0
        if max(section_topic_alignment, document_topic_alignment) >= self.config.retrieval.topic_alignment_threshold:
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

        if support_match_count < self.config.retrieval.answerability_min_support_matches:
            return False

        if query_analysis.exact_match_expected:
            if exact_evidence_match_count < self.config.retrieval.answerability_min_exact_evidence_matches:
                return False

        if query_analysis.multi_doc_expected and len({document.doc_id for document, _ in top_docs}) > 1:
            if len(supporting_doc_ids) < self.config.retrieval.answerability_min_support_matches_multi_doc:
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

    def _keyword_title_fallback_sections(
        self,
        query_analysis: QueryAnalysis,
        top_k: int = 12,
    ) -> list[tuple]:
        """Second-pass retrieval by keyword matching against section/document titles.

        Used when semantic search returns nothing answerable — catches cases
        where the query ("which ports are used") doesn't embed close to the
        section title ("Deployment Diagram") even though the content is relevant.

        Scores each section by how many focus/expanded terms appear in its
        title_terms, keywords, and metadata_tags.  Returns the top_k sections
        sorted by score descending, or an empty list when no matches found.
        """
        query_terms: set[str] = set(query_analysis.focus_terms + query_analysis.expanded_terms)
        if not query_terms:
            return []

        scored: list[tuple] = []
        for section in self.sections.values():
            section_signals: set[str] = set(
                section.title_terms
                + section.keywords
                + section.metadata_tags
            )
            # Also tokenize the raw title for short titles like "Deployment Diagram"
            # that may not have title_terms populated
            for word in re.findall(r"[a-z0-9]+", section.normalized_title or section.title.lower()):
                section_signals.add(word)

            doc = self.documents.get(section.doc_id)
            if doc:
                section_signals.update(doc.title_terms + doc.keywords + doc.metadata_tags)
                for word in re.findall(r"[a-z0-9]+", doc.normalized_title or doc.title.lower()):
                    section_signals.add(word)

            hits = len(query_terms.intersection(section_signals))
            if hits > 0:
                # Normalise: hits relative to query length so shorter queries
                # don't get an unfair advantage over longer ones.
                score = hits / max(len(query_terms), 1)
                scored.append((section, score))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _build_unanswerable_response(
        self,
        query_analysis: QueryAnalysis,
        top_docs: list,
        query_vec=None,
        user_id: str | int | None = None,
    ) -> str:
        """Build a helpful 'not found' response.

        Instead of a dead-end message, the response:
        1. States what couldn't be found.
        2. Lists specific sections/documents that are topically related and
           might contain the answer (so the user doesn't have to guess).
        3. Suggests follow-up questions drawn from the FAQ corpus.
        """
        # ── 1. Find topically-related sections via title keyword match ─────────
        hint_lines: list[str] = []
        seen_doc_titles: set[str] = set()
        query_terms: set[str] = set(query_analysis.focus_terms + query_analysis.expanded_terms)

        if query_terms:
            candidate_hits: list[tuple[float, str, str]] = []  # (score, section_title, doc_title)
            for section in self.sections.values():
                section_signals: set[str] = set(
                    section.title_terms + section.keywords + section.metadata_tags
                )
                for word in re.findall(r"[a-z0-9]+", section.normalized_title or section.title.lower()):
                    section_signals.add(word)
                doc = self.documents.get(section.doc_id)
                doc_title = doc.title if doc else ""
                if doc:
                    section_signals.update(doc.title_terms + doc.keywords + doc.metadata_tags)

                hits = len(query_terms.intersection(section_signals))
                if hits > 0:
                    score = hits / max(len(query_terms), 1)
                    candidate_hits.append((score, section.title, doc_title))

            candidate_hits.sort(key=lambda x: x[0], reverse=True)
            for score, sec_title, doc_title in candidate_hits[:5]:
                if score < 0.1:
                    break
                label = f"**{sec_title}**"
                if doc_title and doc_title not in seen_doc_titles:
                    label += f" (in {doc_title})"
                    seen_doc_titles.add(doc_title)
                if label not in hint_lines:
                    hint_lines.append(label)

        # ── 2. Related questions from FAQ corpus ───────────────────────────────
        suggested_questions: list[str] = []
        if query_vec is not None:
            try:
                candidates = self.corpus.search_faq_questions(
                    query_vec, top_k=8, user_id=user_id
                )
                asked_lower = query_analysis.original_question.strip().casefold()
                for score, q, _, _ in candidates:
                    if score < self.config.retrieval.related_questions_min_score * 0.85:
                        break
                    if q.strip().casefold() == asked_lower:
                        continue
                    suggested_questions.append(q)
                    if len(suggested_questions) >= 3:
                        break
            except Exception:
                pass

        # ── 3. Compose the response ────────────────────────────────────────────
        parts: list[str] = []

        if query_analysis.topic_hints:
            topic_text = ", ".join(t.replace("_", " ") for t in query_analysis.topic_hints)
            parts.append(
                f"I couldn't find a clear answer to this in the indexed documents "
                f"(closest topic area: {topic_text})."
            )
        else:
            parts.append("I couldn't find a clear answer to this in the indexed documents.")

        if hint_lines:
            parts.append(
                "These sections may have what you're looking for:\n"
                + "\n".join(f"- {h}" for h in hint_lines)
            )

        if suggested_questions:
            parts.append(
                "**You might also try asking:**\n"
                + "\n".join(f"- {q}" for q in suggested_questions)
            )

        return "\n\n".join(parts)

    def _append_reference_file_names(self, answer: str, sources: list[SourceReference]) -> str:
        reference_links: list[str] = []
        seen: set[str] = set()
        for source in sources:
            source_path = source.source_path
            if not source_path or source_path in seen:
                continue
            seen.add(source_path)
            label = source.document_title or Path(source_path).stem
            url = build_document_open_url(self.config.storage.public_base_url, source_path)
            reference_links.append(f"[{label}]({url})")

        if not reference_links:
            return answer.strip()

        reference_line = f"Reference: {', '.join(reference_links)}"
        clean_answer = answer.strip()
        if not clean_answer:
            return reference_line
        return f"{clean_answer}\n\n{reference_line}"

    @staticmethod
    def _is_table_separator_row(line: str) -> bool:
        """Return True if the line is a markdown table separator (e.g. | --- | :--- |)."""
        stripped = line.strip()
        return (
            "|" in stripped
            and "-" in stripped
            and bool(re.match(r"^[\|\-\:\s]+$", stripped))
        )

    @staticmethod
    def _parse_table_row(line: str) -> list[str]:
        """Split a markdown table row into cell strings."""
        stripped = line.strip().strip("|")
        return [cell.strip() for cell in stripped.split("|")]

    @staticmethod
    def _markdown_table_to_html(lines: list[str]) -> str:
        """Convert a list of markdown table lines into a compact HTML table string."""
        if len(lines) < 2:
            return "\n".join(lines)
        headers = PolicyGPTBot._parse_table_row(lines[0])
        # lines[1] is the separator row — skip it
        data_rows = [
            PolicyGPTBot._parse_table_row(row)
            for row in lines[2:]
            if "|" in row
        ]
        parts: list[str] = ["<table><thead><tr>"]
        parts += [f"<th>{h}</th>" for h in headers if h]
        parts += ["</tr></thead><tbody>"]
        for row in data_rows:
            parts.append("<tr>")
            parts += [f"<td>{cell}</td>" for cell in row if cell or len(row) > 1]
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    @staticmethod
    def _normalize_answer_markdown(answer: str) -> str:
        # ── pass 1: convert any markdown pipe-tables to HTML ──────────────────
        raw_lines = answer.splitlines()
        converted: list[str] = []
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            # A markdown table starts with a pipe row whose *next* line is a
            # separator row (| --- | --- |).
            if (
                "|" in line
                and i + 1 < len(raw_lines)
                and PolicyGPTBot._is_table_separator_row(raw_lines[i + 1])
            ):
                table_block: list[str] = [line]
                j = i + 1
                while j < len(raw_lines) and "|" in raw_lines[j]:
                    table_block.append(raw_lines[j])
                    j += 1
                converted.append(PolicyGPTBot._markdown_table_to_html(table_block))
                i = j
                continue
            converted.append(line)
            i += 1

        # ── pass 2: clean up non-table lines ──────────────────────────────────
        cleaned_lines: list[str] = []
        for raw_line in converted:
            stripped = raw_line.strip()

            # Drop standalone horizontal rules (---, ***, ___).
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

        # ── pass 3: strip any remaining stray pipe characters ─────────────────
        # Replace " | " style inline separators with " / " so they read
        # naturally.  Pipes inside HTML tag content are extremely unlikely but
        # we guard against matching inside angle-bracket spans.
        cleaned = re.sub(r"(?<![<\w])\s*\|\s*(?![>\w/])", " / ", cleaned)
        # Catch any residual lone pipes (e.g. at line start/end).
        cleaned = re.sub(r"\|", "", cleaned)

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
        limit = max(0, self.config.conversation.recent_chat_message_char_limit)
        if limit:
            return self._truncate_context_text(compact, limit)
        return compact

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
        if QueryIntent.DOCUMENT_LOOKUP in query_analysis.intents:
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

    def _role_clarification_instruction(
        self, query_analysis: QueryAnalysis, user_question: str
    ) -> str:
        """Return a role-clarification instruction for the LLM when the answer
        likely varies by role and the user has not stated their role.

        Returns an empty string when no clarification is needed.
        """
        # Only applies to domains that define roles (contest, policy).
        role_categories = getattr(self.config.domain_profile, "entity_categories", frozenset())
        if "role" not in role_categories:
            return ""

        # Only trigger for eligibility or reward queries (not generic listing).
        role_sensitive_intents = {"eligibility", "threshold", "reward", "benefit"}
        if not role_sensitive_intents & set(query_analysis.intents):
            # Also trigger if the question wording implies role-specific detail
            q_lower = user_question.casefold()
            role_keywords = ("eligible", "qualify", "qualify for", "threshold", "reward", "earn")
            if not any(kw in q_lower for kw in role_keywords):
                return ""

        # Check whether the user already mentioned a role in the question.
        q_lower = user_question.casefold()
        known_roles = [
            e.name.casefold()
            for e in (self.corpus.entity_lookup or {}).values()
            if getattr(e, "category", "") == "role"
        ]
        # Fallback role list when entity map hasn't been built yet.
        if not known_roles:
            known_roles = ["fc", "eim", "ach", "fls", "esl", "manager", "agent"]

        role_mentioned = any(role in q_lower for role in known_roles)
        if role_mentioned:
            return ""

        # Build comma-separated role list from known roles (de-dup, title-case).
        role_display = ", ".join(sorted({r.upper() for r in known_roles[:8]}))
        return (
            f"Role note: The question does not mention a specific role. "
            f"If the answer differs by role ({role_display}), answer for each role separately "
            f"and end with: '**Could you share your role so I can focus on what applies to you?**'"
        )

    @staticmethod
    def _needs_current_date_context(query_analysis: QueryAnalysis) -> bool:
        # Always inject the date for eligibility, timeline, and aggregate queries.
        # Aggregate queries (list all X, active contests, ongoing rewards) depend
        # on the date to determine what is currently active vs expired.
        if {QueryIntent.ELIGIBILITY, QueryIntent.TIMELINE, QueryIntent.AGGREGATE} & set(query_analysis.intents):
            return True

        normalized_question = query_analysis.normalized_question.casefold()
        date_sensitive_markers = (
            "today",
            "current",
            "currently",
            "now",
            "as of",
            "right now",
            "at present",
            "still open",
            "still active",
            "still running",
            "open for",
            "open now",
            "enrollment",
            "enroll",
            "login window",
            "registration",
            "last date",
            "deadline",
            "expired",
            "ended",
            "closed",
            # Active/ongoing listing — user wants to know what is running today
            "active",
            "ongoing",
            "running",
            "available",
            "live",
            "which contest",
            "what contest",
            "list contest",
            "list all",
        )
        return any(marker in normalized_question for marker in date_sensitive_markers)

    @staticmethod
    def _current_date_prompt_line() -> str:
        today = datetime.now().astimezone().date().isoformat()
        return (
            f"Current date: {today}\n"
            "Interpret relative dates such as today, current, and now against this date.\n"
            "If the evidence mentions a specific date range (e.g. contest window, login window, "
            "enrollment period) and today's date falls AFTER the end of that range, explicitly "
            "note in your answer that the window has already passed. "
            "If the date range has not yet started, note that it has not begun. "
            "If today falls within the range, note that it is currently active. "
            "Do this only when the evidence clearly states the dates — do not infer dates."
        )

    def _aggregate_section_signal(self, section) -> int:
        text = " ".join(
            part for part in (
                section.title or "",
                section.summary or "",
                section.raw_text[:500] if section.raw_text else "",
            )
            if part
        ).casefold()

        positive_markers = self.config.domain_profile.aggregate_positive_markers
        negative_markers = (
            "incomplete sentence",
            "requiring verification",
            "items requiring verification",
            "higher-of",
            "higher of",
            "qualification rule",
            "allocation",
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
            score_floor = max(best_score * self.config.retrieval.exact_score_floor_scale, self.config.retrieval.exact_score_floor_min) if best_score else 0.0
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
            # Only hard-drop strongly negative sections once we already have
            # enough coverage; never drop the first representative per doc.
            if signal <= -4 and section.doc_id in seen_doc_ids:
                continue
            # Allow a second section from the same doc if its signal is at
            # least mildly positive (was 6, now 2 to include eligibility etc.)
            if section.doc_id in seen_doc_ids and signal < 2:
                continue
            selected.append((section, score))
            seen_section_ids.add(section.section_id)
            seen_doc_ids.add(section.doc_id)

        for section, score in ranked_sections:
            if len(selected) >= len(top_sections):
                break
            if section.section_id in seen_section_ids:
                continue
            # Lowered from 4 to 1 so boundary sections aren't silently dropped
            if self._aggregate_section_signal(section) < 1:
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
            if QueryIntent.DOCUMENT_LOOKUP in query_analysis.intents:
                return 2
            if query_analysis.multi_doc_expected or QueryIntent.AGGREGATE in query_analysis.intents:
                return 3
            return 1

        if query_analysis.exact_match_expected or query_analysis.context_dependent:
            return 0
        if QueryIntent.DOCUMENT_LOOKUP in query_analysis.intents:
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

        # Always inject the titles of documents referenced in the last answer so
        # follow-up questions resolve correctly even before a conversation summary
        # exists (first few turns where summarize_after_turns hasn't fired yet).
        if query_analysis.context_dependent:
            active_titles: list[str] = [
                self.documents[doc_id].title
                for doc_id in thread.active_doc_ids
                if doc_id in self.documents
            ]
            # Fall back to last-answer sources when active_doc_ids is empty
            if not active_titles and thread.last_answer_sources:
                seen: set[str] = set()
                for src in thread.last_answer_sources:
                    if src.document_title not in seen:
                        seen.add(src.document_title)
                        active_titles.append(src.document_title)
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

    def _docs_from_sections(
        self,
        top_sections: list[tuple],
        preferred_doc_ids: list[str] | None = None,
    ) -> list[tuple]:
        """Derive (document, score) pairs from retrieved sections.

        Groups sections by doc_id, taking the max section score per document.
        Preferred docs get a small boost so context-dependent follow-ups surface
        the same documents as the previous turn.
        """
        doc_scores: dict[str, float] = {}
        preferred_set = set(preferred_doc_ids or [])
        for section, score in top_sections:
            if section.doc_id not in self.documents:
                continue
            doc_scores[section.doc_id] = max(doc_scores.get(section.doc_id, 0.0), score)
        for doc_id in preferred_set:
            if doc_id in doc_scores:
                doc_scores[doc_id] += self.config.retrieval.preferred_doc_score_boost
        ranked = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)
        return [(self.documents[doc_id], doc_scores[doc_id]) for doc_id in ranked]

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
        query_vec: np.ndarray | None = None,
        user_id: str | int | None = None,
    ) -> str:
        minimal_context = self._use_minimal_answer_context(query_analysis)
        doc_aliases = self._build_document_aliases(top_docs, top_sections)
        doc_index_lines = [
            f"{alias}={self.documents[doc_id].title}"
            for doc_id, alias in doc_aliases.items()
            if doc_id in self.documents
        ]
        recent_message_limit = max(0, self.config.conversation.max_recent_messages)
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
                self.config.retrieval.answer_context_doc_summary_char_limit,
            )
            doc_context_lines = [
                f"[{doc_alias}]",
            ]
            if not minimal_context and self.config.output.include_document_metadata_in_answers:
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
                and self.config.output.include_document_orientation_in_answers
            ):
                doc_context_lines.append(f"Summary:\n{orientation_summary or '(empty)'}")
            if len(doc_context_lines) > 1:
                doc_context_parts.append("\n".join(doc_context_lines))

        # For aggregate queries (list all X, show all Y) use each document's
        # pre-generated FAQ as evidence instead of sections.  FAQs are already
        # high-level Q&A distillations — they give complete cross-document
        # coverage for listing questions.  Sections are better for detail queries.
        is_aggregate = QueryIntent.AGGREGATE in query_analysis.intents
        section_context_parts = []

        if is_aggregate:
            # Search FAQ questions across ALL documents independently of which
            # docs were retrieved — this gives cross-corpus coverage without
            # being limited by doc-level retrieval scoring.
            faq_hits = self.corpus.search_faq_questions(
                query_vec=query_vec if query_vec is not None else self._embed_one(query_analysis.original_question),
                top_k=self.config.retrieval.aggregate_faq_top_k,
                user_id=user_id,
            )
            if faq_hits:
                # Group Q&A pairs by document title for a clean prompt layout.
                by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for _, q, a, doc_title in faq_hits:
                    by_doc[doc_title].append((q, a))
                for doc_title, pairs in by_doc.items():
                    qa_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
                    section_context_parts.append(f"[{doc_title}]\n{qa_lines}")
            else:
                # Fallback: no FAQ embeddings available — use full FAQ text or summary
                for document, _ in top_docs:
                    faq_text = (document.faq or "").strip()
                    if not faq_text:
                        faq_text = self._truncate_context_text(
                            self.redactor.unmask_text(document.summary),
                            self.config.retrieval.answer_context_doc_summary_char_limit,
                        )
                    else:
                        faq_text = self.redactor.unmask_text(faq_text)
                    section_context_parts.append(f"[{document.title} FAQ]\n{faq_text}")
        else:
            prompt_sections = self._sections_for_answer_context(query_analysis, top_sections, top_docs)
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

                # Deduplicate evidence blocks across sections so the LLM doesn't
                # see the same raw text twice (common when two sections from the
                # same document are retrieved).
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
                    self.config.retrieval.answer_context_doc_summary_char_limit,
                )
                # Content-type hint helps the LLM interpret the evidence correctly
                _raw_text = section.raw_text or ""
                if "<table" in _raw_text.lower() or _raw_text.count("|") >= 6:
                    _ctype = "table"
                elif bool(re.search(r"\n\s{0,4}[-*\u2022]\s", _raw_text)):
                    _ctype = "list"
                else:
                    _ctype = "prose"
                section_context_lines = [
                    f"[{evidence_tag} {section.title}] [{_ctype}]",
                ]
                if not minimal_context and self.config.output.include_section_metadata_in_answers:
                    section_context_lines.append(
                        f"Section type: {section.section_type or 'general'} | Tags: {', '.join(section.metadata_tags) or 'none'}"
                    )
                if (
                    not minimal_context
                    and not query_analysis.exact_match_expected
                    and self.config.output.include_section_orientation_in_answers
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
        if doc_context_parts and (QueryIntent.DOCUMENT_LOOKUP in query_analysis.intents or self._uses_open_weight_prompt_profile()):
            prompt_parts.append(f"Doc notes:\n{chr(10).join(doc_context_parts)}")
        if self._supplementary_facts:
            prompt_parts.append(
                f"Background facts (use to supplement retrieved evidence; do not cite as a source):\n"
                f"{self._supplementary_facts}"
            )
        role_instruction = self._role_clarification_instruction(
            query_analysis, query_analysis.original_question
        )
        prompt_parts.append(
            "Answer only from the evidence. Prefer raw evidence over summaries. Never mention internal IDs. "
            "If the evidence states explicit exclusions, exceptions, or disqualifying conditions, include them in the answer — "
            "do not silently omit negative conditions."
            + (f"\n{role_instruction}" if role_instruction else "")
        )
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

        score_details: dict = getattr(self.corpus, "last_retrieval_score_details", {})

        lines.extend(["", "=== Top Sections (final selection) ==="])
        if top_sections:
            for index, (section, score) in enumerate(top_sections, start=1):
                document = self._get_doc_for_section(section)
                doc_title = document.title if document else self._get_doc_title_for_section(section)
                snippets = self.corpus.extract_evidence_snippets(section, query_analysis)
                sd = score_details.get(section.section_id, {})
                lines.extend(
                    [
                        f"{index}. {doc_title} :: {section.title} | final={score:.4f} | file={section.source_path}",
                        f"   type={section.section_type or 'general'} | tags={', '.join(section.metadata_tags) or 'none'}",
                        f"   summary={self.redactor.unmask_text(section.summary).strip() or '(empty)'}",
                        "   snippets:",
                    ]
                )
                if snippets:
                    lines.extend([f"   - {snippet}" for snippet in snippets])
                else:
                    lines.append("   - (none)")
                if sd:
                    llm_part = (
                        f"llm={sd['llm_score']:.4f}"
                        if sd.get("llm_score") is not None
                        else ("llm=skipped" if sd.get("llm_skipped") else "llm=n/a")
                    )
                    matched = ",".join(sd.get("matched_by") or []) or "?"
                    lines.append(
                        f"   os: matched_by={matched}"
                        f" | kw={sd.get('kw_score', 0.0):.3f}"
                        f" sim={sd.get('sim_score', 0.0):.3f}"
                        f" vec={sd.get('vec_score', 0.0):.3f}"
                        f" → hybrid={sd.get('os_hybrid_score','?'):.4f}"
                    )
                    lines.append(
                        f"   rerank: snippet={sd.get('snippet_overlap','?'):.3f}"
                        f" | focus={sd.get('focus_match','?'):.3f}"
                        f" | precise={sd.get('precise_match','?'):.3f}"
                        f" | sec_type={sd.get('section_type_boost','?'):.3f}"
                        f" | tag={sd.get('tag_boost','?'):.3f}"
                        f" | role={sd.get('role_alignment','?'):.3f}"
                        f" | title={sd.get('title_alignment','?'):.3f}"
                        f" → h={sd.get('h_score','?'):.4f}"
                        f" | {llm_part}"
                        f" → final={sd.get('final_score','?'):.4f}"
                    )
        else:
            lines.append("(none)")

        # All candidates that went through reranking (not just the final selection)
        lines.extend(["", "=== All Reranked Candidates ==="])
        if score_details:
            # Sort by final_score descending for easy scanning
            all_candidates = sorted(
                score_details.items(),
                key=lambda kv: kv[1].get("final_score", 0.0),
                reverse=True,
            )
            for sid, sd in all_candidates:
                sec = self.sections.get(sid)
                sec_label = f"{sec.title} ({sec.source_path})" if sec else sid
                doc = self.documents.get(sec.doc_id) if sec else None
                doc_label = doc.title if doc else "?"
                llm_part = (
                    f"llm={sd['llm_score']:.4f}"
                    if sd.get("llm_score") is not None
                    else ("llm=skipped" if sd.get("llm_skipped") else "llm=n/a")
                )
                matched = ",".join(sd.get("matched_by") or []) or "?"
                lines.append(
                    f"  [{doc_label}] {sec_label}"
                    f" | matched_by={matched}"
                    f" | kw={sd.get('kw_score', 0.0):.3f}"
                    f" sim={sd.get('sim_score', 0.0):.3f}"
                    f" vec={sd.get('vec_score', 0.0):.3f}"
                    f" os_hybrid={sd.get('os_hybrid_score','?'):.4f}"
                    f" | h={sd.get('h_score','?'):.4f}"
                    f" | {llm_part}"
                    f" | final={sd.get('final_score','?'):.4f}"
                )
        else:
            lines.append("(no score details available)")

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
                    f"- {self._get_doc_title_for_section(section)} :: {section.title} | score={score:.4f} | file={section.source_path}"
                    for section, score in top_sections
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
        if not self.config.storage.debug:
            return None
        raw_path = (self.config.storage.debug_log_dir or "").strip()
        if not raw_path:
            return None
        base_path = Path(self.config.storage.document_folder)
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
            provider=self.config.ai.ai_provider,
            model_name=self.config.ai.chat_model,
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
            f"Domain: {self.config.domain_context}\n"
            "Summarize this conversation for future retrieval and follow-up handling. "
            "Capture: specific contest/document names, section titles, current topic, "
            "roles and eligibility discussed, rewards or thresholds mentioned, "
            "key answered points, and any unresolved questions. "
            "Be concise and factual."
        )
        user_prompt = f"{recent_chat}\n\n{source_context}" if source_context else recent_chat
        summary = self._llm_text_with_debug_log(
            purpose="conversation_summary",
            system_prompt=self.redactor.mask_text(system_prompt),
            user_prompt=self.redactor.mask_text(user_prompt),
            max_output_tokens=self.config.output.conversation_summary_max_output_tokens,
        )
        return self.redactor.unmask_text(summary)
