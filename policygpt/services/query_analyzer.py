from dataclasses import dataclass, field

from policygpt.models import DocumentRecord
from policygpt.services.taxonomy import (
    DOMAIN_TOPIC_SYNONYMS,
    INTENT_PATTERNS,
    STOPWORDS,
    detect_matching_labels,
    humanize_term,
    is_informative_term,
    keywordize_text,
    normalize_text,
    tokenize_text,
    unique_preserving_order,
)


DETAIL_PHRASES: tuple[str, ...] = (
    "in detail",
    "detailed",
    "step by step",
    "elaborate",
    "full detail",
    "complete detail",
)


MULTI_DOC_COVERAGE_PHRASES: tuple[str, ...] = (
    "across policies",
    "across documents",
    "all policies",
    "all documents",
    "which policies",
    "which documents",
    "every policy",
    "every document",
    "list all",
    "compare",
    "comparison",
)


CONTEXT_REFERENCE_PHRASES: tuple[str, ...] = (
    "what about",
    "how about",
    "same policy",
    "same document",
    "same contest",
    "same doc",
    "same section",
    "this policy",
    "this document",
    "this contest",
    "this section",
    "that policy",
    "that document",
    "that contest",
    "that section",
    "this one",
    "that one",
    "from above",
    "from below",
    "why is it missing",
    "why it's missing",
    "why its missing",
)


CONTEXT_REFERENCE_TOKENS: set[str] = {
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "same",
    "here",
    "there",
    "above",
    "below",
}


DOCUMENT_LOOKUP_PHRASES: tuple[str, ...] = (
    "give me",
    "show me",
    "open",
    "share",
    "provide",
    "send me",
    "find",
    "locate",
)


DOCUMENT_LOOKUP_NOUNS: set[str] = {
    "policy",
    "policies",
    "procedure",
    "procedures",
    "process",
    "processes",
    "guideline",
    "guidelines",
    "manual",
    "document",
    "documents",
    "checklist",
    "checklists",
    "matrix",
    "faq",
    "form",
    "forms",
}


DOMAIN_TOPIC_PHRASE_TERMS: set[str] = {
    term
    for phrases in DOMAIN_TOPIC_SYNONYMS.values()
    for phrase in phrases
    for term in keywordize_text(phrase)
    if "_" in term
}


INTENT_TO_SECTION_TYPES: dict[str, tuple[str, ...]] = {
    "document_lookup": ("general", "scope", "process", "checklist"),
    "aggregate": ("scope", "eligibility", "definitions", "timeline", "exceptions"),
    "checklist": ("checklist", "process", "documents_required"),
    "process": ("process", "checklist", "timeline", "responsibilities"),
    "eligibility": ("eligibility", "scope", "exceptions"),
    "approval": ("approval", "matrix", "responsibilities"),
    "timeline": ("timeline", "process", "checklist"),
    "documents_required": ("documents_required", "checklist", "process"),
    "contact": ("contact", "responsibilities"),
    "comparison": ("eligibility", "approval", "process", "timeline", "scope", "exceptions"),
    "scope": ("scope", "eligibility", "definitions"),
    "exceptions": ("exceptions", "eligibility"),
}


@dataclass(frozen=True)
class QueryAnalysis:
    original_question: str
    normalized_question: str
    canonical_question: str
    detail_requested: bool
    multi_doc_expected: bool
    exact_match_expected: bool
    context_dependent: bool
    intents: list[str] = field(default_factory=list)
    topic_hints: list[str] = field(default_factory=list)
    focus_terms: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    expected_section_types: list[str] = field(default_factory=list)


class QueryAnalyzer:
    def analyze(
        self,
        user_question: str,
        active_document_titles: list[str] | None = None,
        candidate_documents: list[DocumentRecord] | None = None,
    ) -> QueryAnalysis:
        normalized_question = normalize_text(user_question)
        focus_terms = self._select_focus_terms(user_question)
        corpus_topics, corpus_expanded_terms, supporting_titles = self._infer_corpus_topics(
            user_question=user_question,
            normalized_question=normalized_question,
            focus_terms=focus_terms,
            candidate_documents=candidate_documents or [],
        )

        fallback_topics = detect_matching_labels(user_question, DOMAIN_TOPIC_SYNONYMS)
        topic_hints = corpus_topics or fallback_topics
        intents = detect_matching_labels(user_question, INTENT_PATTERNS)
        if self._looks_like_document_lookup(user_question, normalized_question):
            intents = ["document_lookup", *intents]
        if not intents:
            intents = ["general"]
        intents = unique_preserving_order(intents)

        context_dependent = self._needs_conversation_context(
            user_question=user_question,
            normalized_question=normalized_question,
            focus_terms=focus_terms,
        )

        expanded_terms = list(focus_terms)
        expanded_terms.extend(corpus_expanded_terms)
        if not corpus_expanded_terms:
            for topic in fallback_topics:
                expanded_terms.append(topic.replace(" ", "_"))
                expanded_terms.extend(keywordize_text(" ".join(DOMAIN_TOPIC_SYNONYMS.get(topic, ()))))

        if context_dependent and active_document_titles:
            for title in active_document_titles:
                expanded_terms.extend(self._filter_informative_terms(keywordize_text(title)))

        expected_section_types: list[str] = []
        for intent in intents:
            expected_section_types.extend(INTENT_TO_SECTION_TYPES.get(intent, ()))

        detail_requested = any(phrase in normalized_question for phrase in DETAIL_PHRASES)
        multi_doc_expected = self._expects_multi_doc_coverage(
            user_question=user_question,
            normalized_question=normalized_question,
            intents=intents,
        )
        exact_match_expected = self._prefers_precise_evidence(
            user_question=user_question,
            normalized_question=normalized_question,
            intents=intents,
            focus_terms=focus_terms,
            multi_doc_expected=multi_doc_expected,
        )
        canonical_lines = [f"User question: {user_question.strip()}"]
        if topic_hints:
            canonical_lines.append(f"Inferred policy topics: {', '.join(topic_hints)}")
        if intents:
            canonical_lines.append(f"Inferred answer intent: {', '.join(intents)}")
        if multi_doc_expected:
            canonical_lines.append("Expected evidence coverage: multiple documents")
        if exact_match_expected:
            canonical_lines.append("Evidence preference: exact raw policy wording")
        if context_dependent:
            canonical_lines.append("Conversation context: active follow-up")
        if supporting_titles:
            canonical_lines.append(f"Likely matching documents: {', '.join(supporting_titles)}")
        if expanded_terms:
            canonical_lines.append(f"Expanded search terms: {', '.join(unique_preserving_order(expanded_terms)[:24])}")

        return QueryAnalysis(
            original_question=user_question,
            normalized_question=normalized_question,
            canonical_question="\n".join(canonical_lines),
            detail_requested=detail_requested,
            multi_doc_expected=multi_doc_expected,
            exact_match_expected=exact_match_expected,
            context_dependent=context_dependent,
            intents=intents,
            topic_hints=topic_hints,
            focus_terms=focus_terms,
            expanded_terms=unique_preserving_order(expanded_terms),
            expected_section_types=unique_preserving_order(expected_section_types),
        )

    def _infer_corpus_topics(
        self,
        user_question: str,
        normalized_question: str,
        focus_terms: list[str],
        candidate_documents: list[DocumentRecord],
    ) -> tuple[list[str], list[str], list[str]]:
        if not candidate_documents or not focus_terms:
            return [], [], []

        query_terms = unique_preserving_order(focus_terms + self._filter_informative_terms(keywordize_text(user_question)))
        ranked_documents: list[tuple[DocumentRecord, float]] = []
        for document in candidate_documents:
            score = self._score_document_match(
                normalized_question=normalized_question,
                query_terms=query_terms,
                document=document,
            )
            if score > 0:
                ranked_documents.append((document, score))

        if not ranked_documents:
            return [], [], []

        ranked_documents.sort(key=lambda item: item[1], reverse=True)
        top_score = ranked_documents[0][1]
        selected_documents: list[tuple[DocumentRecord, float]] = []
        minimum_selected_score = top_score * 0.55 if top_score > 0 else 0.0
        for document, score in ranked_documents:
            if len(selected_documents) >= 3:
                break
            if selected_documents and score < minimum_selected_score:
                break
            selected_documents.append((document, score))

        if not selected_documents:
            selected_documents.append(ranked_documents[0])

        topic_weights: dict[str, float] = {}
        expansion_weights: dict[str, float] = {}
        supporting_titles: list[str] = []

        for document, score in selected_documents:
            supporting_titles.append(document.title)
            relative_weight = score / top_score if top_score > 0 else 1.0
            weight = max(relative_weight, 0.45)

            raw_topics = document.metadata_tags or self._fallback_document_topics(document)
            for topic in raw_topics:
                normalized_topic = normalize_text(topic)
                if not normalized_topic:
                    continue
                existing_score = topic_weights.get(normalized_topic, 0.0)
                topic_weights[normalized_topic] = max(existing_score, weight)
                for term in self._terms_from_labels([topic]):
                    expansion_weights[term] = expansion_weights.get(term, 0.0) + weight + 0.4

            expansion_terms = self._filter_informative_terms(document.title_terms + document.keywords)
            for term in expansion_terms[:18]:
                expansion_weights[term] = expansion_weights.get(term, 0.0) + weight

        ordered_topics = [
            topic
            for topic, _ in sorted(topic_weights.items(), key=lambda item: item[1], reverse=True)
        ]
        ordered_expansion_terms = [
            term
            for term, _ in sorted(
                expansion_weights.items(),
                key=lambda item: (item[1], item[0].count("_"), len(item[0])),
                reverse=True,
            )
            if term not in focus_terms
        ]
        return (
            unique_preserving_order(ordered_topics[:4]),
            unique_preserving_order(ordered_expansion_terms[:18]),
            unique_preserving_order(supporting_titles[:3]),
        )

    def _score_document_match(
        self,
        normalized_question: str,
        query_terms: list[str],
        document: DocumentRecord,
    ) -> float:
        if not query_terms:
            return 0.0

        query_set = set(query_terms)
        title_terms = set(self._filter_informative_terms(document.title_terms + keywordize_text(document.canonical_title)))
        keyword_terms = set(self._filter_informative_terms(document.keywords))
        tag_terms = set(self._terms_from_labels(document.metadata_tags))

        title_overlap = len(query_set.intersection(title_terms))
        keyword_overlap = len(query_set.intersection(keyword_terms))
        tag_overlap = len(query_set.intersection(tag_terms))
        direct_phrase_hits = sum(
            1
            for label in document.metadata_tags
            if normalize_text(label) and normalize_text(label) in normalized_question
        )

        return (2.8 * title_overlap) + (1.6 * keyword_overlap) + (2.2 * tag_overlap) + (1.8 * direct_phrase_hits)

    def _fallback_document_topics(self, document: DocumentRecord) -> list[str]:
        fallback_topics: list[str] = []
        for term in self._filter_informative_terms(document.title_terms + document.keywords):
            label = humanize_term(term)
            if label:
                fallback_topics.append(label)
            if len(fallback_topics) >= 4:
                break
        return unique_preserving_order(fallback_topics)

    @staticmethod
    def _terms_from_labels(labels: list[str]) -> list[str]:
        terms: list[str] = []
        for label in labels:
            if not label:
                continue
            normalized_label_term = label.replace(" ", "_")
            if is_informative_term(normalized_label_term):
                terms.append(normalized_label_term)
            terms.extend(keywordize_text(label))
        return unique_preserving_order([term for term in terms if is_informative_term(term)])

    @staticmethod
    def _filter_informative_terms(terms: list[str]) -> list[str]:
        return unique_preserving_order([term for term in terms if is_informative_term(term)])

    @staticmethod
    def _looks_like_document_lookup(user_question: str, normalized_question: str) -> bool:
        query_tokens = [token for token in tokenize_text(user_question) if token not in STOPWORDS]
        if not query_tokens:
            return False

        has_document_noun = any(token in DOCUMENT_LOOKUP_NOUNS for token in query_tokens)
        if not has_document_noun:
            return False

        if any(phrase in normalized_question for phrase in DOCUMENT_LOOKUP_PHRASES):
            return True

        if query_tokens[0] in {"give", "show", "open", "share", "provide", "send", "find", "locate"}:
            return True

        if len(query_tokens) <= 6 and query_tokens[-1] in DOCUMENT_LOOKUP_NOUNS:
            return True

        return False

    @staticmethod
    def _select_focus_terms(text: str) -> list[str]:
        raw_tokens = [
            token
            for token in tokenize_text(text)
            if (len(token) >= 2 or token.isdigit()) and token not in STOPWORDS
        ]
        unigram_terms = [token for token in raw_tokens if is_informative_term(token)]
        bigram_terms: list[str] = []
        for left, right in zip(raw_tokens, raw_tokens[1:]):
            term = f"{left}_{right}"
            if (
                (is_informative_term(left) and is_informative_term(right))
                or QueryAnalyzer._is_numeric_modifier_pair(left, right)
                or term in DOMAIN_TOPIC_PHRASE_TERMS
            ):
                bigram_terms.append(term)

        focus_terms = unigram_terms + bigram_terms
        if not focus_terms:
            raw_fallback_tokens = [token for token in tokenize_text(text) if (len(token) >= 2 or token.isdigit()) and token not in STOPWORDS]
            focus_terms = [
                token
                for token in raw_fallback_tokens
                if len(token) >= 2 or (token.isdigit() and token != "0")
            ]
        return unique_preserving_order(focus_terms[:20])

    @staticmethod
    def _is_numeric_modifier_pair(left: str, right: str) -> bool:
        return is_informative_term(left) and right.isdigit()

    @staticmethod
    def _expects_multi_doc_coverage(
        user_question: str,
        normalized_question: str,
        intents: list[str],
    ) -> bool:
        if "comparison" in intents:
            return True

        if any(phrase in normalized_question for phrase in MULTI_DOC_COVERAGE_PHRASES):
            return True

        query_tokens = set(tokenize_text(user_question))
        if query_tokens.intersection({"policies", "documents"}) and query_tokens.intersection(
            {"all", "every", "which", "across", "multiple", "different"}
        ):
            return True

        return False

    @staticmethod
    def _needs_conversation_context(
        user_question: str,
        normalized_question: str,
        focus_terms: list[str],
    ) -> bool:
        if any(phrase in normalized_question for phrase in CONTEXT_REFERENCE_PHRASES):
            return True

        raw_tokens = tokenize_text(user_question)
        informative_tokens = [token for token in raw_tokens if token not in STOPWORDS]
        if not informative_tokens:
            return True

        if any(token in CONTEXT_REFERENCE_TOKENS for token in raw_tokens):
            return True

        has_specific_phrase = any("_" in term for term in focus_terms)
        if len(informative_tokens) == 1:
            token = informative_tokens[0]
            if not has_specific_phrase and not token.isdigit():
                return True

        if len(informative_tokens) <= 2 and not has_specific_phrase:
            if all(token in {"missing", "more", "else", "also", "then"} or token.isdigit() for token in informative_tokens):
                return True

        return False

    @staticmethod
    def _prefers_precise_evidence(
        user_question: str,
        normalized_question: str,
        intents: list[str],
        focus_terms: list[str],
        multi_doc_expected: bool,
    ) -> bool:
        if multi_doc_expected or {"document_lookup", "aggregate", "comparison"}.intersection(intents):
            return False

        precision_phrases = (
            "what is",
            "what are",
            "who is",
            "when is",
            "define",
            "meaning of",
            "stands for",
            "what does",
            "does it mention",
            "is there",
            "how much",
            "how many",
        )
        if any(phrase in normalized_question for phrase in precision_phrases):
            return True

        informative_tokens = [
            token
            for token in tokenize_text(user_question)
            if token not in STOPWORDS
        ]
        if any(any(part.isdigit() for part in term.split("_")) for term in focus_terms):
            return True

        if len(informative_tokens) <= 6 and any("_" in term for term in focus_terms):
            return True

        return len(informative_tokens) <= 4 and "process" not in intents and "checklist" not in intents
