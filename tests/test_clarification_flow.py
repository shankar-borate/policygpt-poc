import unittest

import numpy as np

from policygpt.cache import CacheManager
from policygpt.config import Config
from policygpt.config.user_profiles import parse_user_profile_text, resolve_user_profile
from policygpt.core.bot import PolicyGPTBot
from policygpt.core.conversations import ConversationManager
from policygpt.core.retrieval.query_analyzer import QueryAnalysis
from policygpt.models import DocumentRecord


class _IdentityRedactor:
    @staticmethod
    def mask_text(text: str) -> str:
        return text

    @staticmethod
    def unmask_text(text: str) -> str:
        return text


class _StubQueryAnalyzer:
    def __init__(self, *analyses: QueryAnalysis) -> None:
        self._analyses = list(analyses)
        self.calls: list[dict] = []

    def analyze(
        self,
        user_question: str,
        active_document_titles=None,
        candidate_documents=None,
        entity_lookup=None,
        user_profile=None,
    ) -> QueryAnalysis:
        self.calls.append({
            "user_question": user_question,
            "user_profile": user_profile,
        })
        if not self._analyses:
            raise AssertionError("No queued QueryAnalysis available for test.")
        return self._analyses.pop(0)


class _StubCorpus:
    def __init__(self, document: DocumentRecord, faq_answer: str | None = None) -> None:
        self.documents = {document.doc_id: document}
        self.sections = {}
        self.entity_lookup = {}
        self._os_retriever = object()
        self._faq_answer = faq_answer

    def faq_fastpath_lookup(self, *_args, **_kwargs):
        return self._faq_answer


def _build_query_analysis(
    question: str,
    *,
    normalized_question: str | None = None,
    intents: list[str] | None = None,
    topic_hints: list[str] | None = None,
    focus_terms: list[str] | None = None,
    expanded_terms: list[str] | None = None,
    expected_section_types: list[str] | None = None,
    context_dependent: bool = False,
) -> QueryAnalysis:
    normalized = normalized_question or question.lower()
    focus = focus_terms or []
    return QueryAnalysis(
        original_question=question,
        normalized_question=normalized,
        canonical_question=f"User question: {question}",
        detail_requested=False,
        multi_doc_expected=False,
        exact_match_expected=False,
        context_dependent=context_dependent,
        intents=intents or ["general"],
        topic_hints=topic_hints or [],
        focus_terms=focus,
        expanded_terms=expanded_terms or list(focus),
        expected_section_types=expected_section_types or [],
    )


def _build_bot(query_analyzer: _StubQueryAnalyzer, *, faq_answer: str | None = None) -> PolicyGPTBot:
    document = DocumentRecord(
        doc_id="doc-1",
        title="Travel Policy",
        source_path="travel_policy.md",
        raw_text="",
        masked_text="",
        summary="",
        summary_embedding=np.zeros(1),
    )
    bot = PolicyGPTBot.__new__(PolicyGPTBot)
    bot.config = Config()
    bot.redactor = _IdentityRedactor()
    bot.cache = CacheManager()
    bot.corpus = _StubCorpus(document, faq_answer=faq_answer)
    bot.query_analyzer = query_analyzer
    bot.conversations = ConversationManager()
    bot._supplementary_facts = ""
    bot.ai = None
    bot._build_retrieval_query = lambda thread, analysis: analysis.original_question
    bot._embed_one = lambda text: np.zeros(1)
    return bot


class UserProfileUtilityTests(unittest.TestCase):
    def test_resolve_user_profile_returns_empty_for_unknown_user(self) -> None:
        self.assertTrue(resolve_user_profile("policy").is_empty())
        self.assertTrue(resolve_user_profile("policy", user_id="user-1").is_empty())

    def test_parse_user_profile_text_reads_labeled_fields(self) -> None:
        profile = parse_user_profile_text(
            "Role: Branch Manager; Department: Retail Banking; Grade: M3; Location: Mumbai"
        )

        self.assertEqual(profile.role, "Branch Manager")
        self.assertEqual(profile.department, "Retail Banking")
        self.assertEqual(profile.grade, "M3")
        self.assertEqual(profile.location, "Mumbai")

    def test_parse_user_profile_text_does_not_treat_question_as_role(self) -> None:
        profile = parse_user_profile_text("Am I eligible for travel allowance?")

        self.assertTrue(profile.is_empty())


class ClarificationFlowTests(unittest.TestCase):
    def test_personalized_question_requires_profile_before_answering(self) -> None:
        question = "Am I eligible for travel allowance?"
        query_analysis = _build_query_analysis(
            question,
            normalized_question="am i eligible for travel allowance",
            intents=["eligibility"],
            topic_hints=["travel allowance"],
            focus_terms=["eligible", "travel", "allowance"],
            expanded_terms=["eligible", "travel", "allowance"],
            expected_section_types=["eligibility"],
        )
        analyzer = _StubQueryAnalyzer(query_analysis)
        bot = _build_bot(analyzer)

        thread_id = bot.new_thread()
        result = bot.chat(thread_id, question)

        self.assertIn("what is your role, department, grade, and location?", result.answer.lower())
        self.assertEqual(result.thread.pending_clarification_kind, "profile")
        self.assertEqual(result.thread.pending_question, question)

    def test_profile_reply_resumes_original_question_with_stored_profile(self) -> None:
        question = "Am I eligible for travel allowance?"
        query_analysis = _build_query_analysis(
            question,
            normalized_question="am i eligible for travel allowance",
            intents=["eligibility"],
            topic_hints=["travel allowance"],
            focus_terms=["eligible", "travel", "allowance"],
            expanded_terms=["eligible", "travel", "allowance"],
            expected_section_types=["eligibility"],
        )
        analyzer = _StubQueryAnalyzer(query_analysis, query_analysis)
        bot = _build_bot(analyzer, faq_answer="Yes. Travel allowance is available for eligible employees.")

        thread_id = bot.new_thread()
        bot.chat(thread_id, question)
        result = bot.chat(
            thread_id,
            "Role: Branch Manager; Department: Retail Banking; Grade: M3; Location: Mumbai",
        )

        self.assertEqual(result.answer, "Yes. Travel allowance is available for eligible employees.")
        self.assertEqual(result.thread.pending_clarification_kind, "")
        self.assertEqual(result.thread.pending_question, "")
        self.assertEqual(result.thread.profile_role, "Branch Manager")
        self.assertEqual(result.thread.profile_department, "Retail Banking")
        self.assertEqual(analyzer.calls[-1]["user_question"], question)
        self.assertEqual(analyzer.calls[-1]["user_profile"].role, "Branch Manager")

    def test_pending_profile_reasks_when_user_sends_another_question(self) -> None:
        question = "Am I eligible for travel allowance?"
        query_analysis = _build_query_analysis(
            question,
            normalized_question="am i eligible for travel allowance",
            intents=["eligibility"],
            topic_hints=["travel allowance"],
            focus_terms=["eligible", "travel", "allowance"],
            expanded_terms=["eligible", "travel", "allowance"],
            expected_section_types=["eligibility"],
        )
        analyzer = _StubQueryAnalyzer(query_analysis)
        bot = _build_bot(analyzer)

        thread_id = bot.new_thread()
        bot.chat(thread_id, question)
        result = bot.chat(thread_id, "What is the policy?")

        self.assertIn("what is your role, department, grade, and location?", result.answer.lower())
        self.assertEqual(result.thread.pending_clarification_kind, "profile")
        self.assertEqual(result.thread.pending_question, question)
        self.assertEqual(result.thread.profile_role, "")

    def test_vague_question_triggers_general_clarification(self) -> None:
        query_analysis = _build_query_analysis(
            "Tell me more",
            normalized_question="tell me more",
            intents=["general"],
            topic_hints=[],
            focus_terms=["tell", "more"],
            expanded_terms=["tell", "more"],
        )
        analyzer = _StubQueryAnalyzer(query_analysis)
        bot = _build_bot(analyzer)

        thread_id = bot.new_thread()
        result = bot.chat(thread_id, "Tell me more")

        self.assertIn("bit more specific", result.answer.lower())
        self.assertEqual(result.thread.pending_clarification_kind, "question")
        self.assertEqual(result.thread.pending_question, "Tell me more")


if __name__ == "__main__":
    unittest.main()
