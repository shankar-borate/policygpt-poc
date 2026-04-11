import unittest
from unittest.mock import patch

import numpy as np

from policygpt.core.bot import PolicyGPTBot
from policygpt.config import RUNTIME_COST_PROFILE_PRESETS, Config
from policygpt.models import DocumentRecord, Message, SectionRecord, ThreadState
from policygpt.core.retrieval.query_analyzer import QueryAnalysis
from policygpt.observability.usage_metrics import estimate_text_tokens


class _IdentityRedactor:
    @staticmethod
    def unmask_text(text: str) -> str:
        return text


class _StubCorpus:
    def __init__(self, document: DocumentRecord) -> None:
        self.documents = {document.doc_id: document}

    @staticmethod
    def extract_answer_evidence_blocks(_section, _query_analysis):
        return [
            "Employees can submit a request after manager approval. "
            "Requests without approval are not eligible for processing."
        ]

    @staticmethod
    def extract_evidence_snippets(_section, _query_analysis, limit=1):
        return ["Employees can submit after manager approval."][:limit]


class RuntimeCostProfileConfigTests(unittest.TestCase):
    def test_runtime_cost_profile_layers_on_accuracy_profile(self) -> None:
        config = Config(
            ai_profile="bedrock-claude-sonnet-4-6",
            accuracy_profile="medium",
            runtime_cost_profile="aggressive",
        )

        self.assertEqual(config.top_docs, RUNTIME_COST_PROFILE_PRESETS["aggressive"]["top_docs"])
        self.assertEqual(config.max_sections_to_llm, RUNTIME_COST_PROFILE_PRESETS["aggressive"]["max_sections_to_llm"])
        self.assertEqual(config.chat_max_output_tokens, RUNTIME_COST_PROFILE_PRESETS["aggressive"]["chat_max_output_tokens"])
        self.assertFalse(config.include_document_orientation_in_answers)
        self.assertFalse(config.include_section_metadata_in_answers)

    def test_manual_runtime_knob_overrides_are_preserved(self) -> None:
        config = Config(
            accuracy_profile="medium",
            runtime_cost_profile="aggressive",
            top_docs=9,
            chat_max_output_tokens=333,
            include_document_orientation_in_answers=True,
        )

        self.assertEqual(config.top_docs, 9)
        self.assertEqual(config.chat_max_output_tokens, 333)
        self.assertTrue(config.include_document_orientation_in_answers)
        self.assertFalse(config.include_section_orientation_in_answers)

    def test_from_env_supports_runtime_cost_profile_override(self) -> None:
        with patch.dict("os.environ", {"POLICY_GPT_RUNTIME_COST_PROFILE": "aggressive"}, clear=False):
            config = Config.from_env()

        self.assertEqual(config.runtime_cost_profile, "aggressive")
        self.assertEqual(config.chat_max_output_tokens, RUNTIME_COST_PROFILE_PRESETS["aggressive"]["chat_max_output_tokens"])

    def test_from_env_supports_fine_grained_prompt_cost_overrides(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POLICY_GPT_RUNTIME_COST_PROFILE": "standard",
                "POLICY_GPT_ANSWER_CONTEXT_DOC_SUMMARY_CHAR_LIMIT": "90",
                "POLICY_GPT_RECENT_CHAT_MESSAGE_CHAR_LIMIT": "120",
                "POLICY_GPT_INCLUDE_DOCUMENT_METADATA_IN_ANSWERS": "0",
                "POLICY_GPT_INCLUDE_SECTION_METADATA_IN_ANSWERS": "false",
                "POLICY_GPT_INCLUDE_DOCUMENT_ORIENTATION_IN_ANSWERS": "no",
                "POLICY_GPT_INCLUDE_SECTION_ORIENTATION_IN_ANSWERS": "off",
            },
            clear=False,
        ):
            config = Config.from_env()

        self.assertEqual(config.runtime_cost_profile, "standard")
        self.assertEqual(config.answer_context_doc_summary_char_limit, 90)
        self.assertEqual(config.recent_chat_message_char_limit, 120)
        self.assertFalse(config.include_document_metadata_in_answers)
        self.assertFalse(config.include_section_metadata_in_answers)
        self.assertFalse(config.include_document_orientation_in_answers)
        self.assertFalse(config.include_section_orientation_in_answers)

    def test_unknown_runtime_cost_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Config(runtime_cost_profile="ultra-lean")


class RuntimeCostProfilePromptTests(unittest.TestCase):
    def test_aggressive_runtime_profile_removes_orientation_and_metadata_from_prompt(self) -> None:
        config = Config(runtime_cost_profile="aggressive")
        document, section = self._build_document_and_section()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(document)

        query_analysis = QueryAnalysis(
            original_question="Who can apply?",
            normalized_question="who can apply",
            canonical_question="User question: Who can apply?",
            detail_requested=False,
            multi_doc_expected=False,
            exact_match_expected=False,
            context_dependent=False,
            intents=["eligibility"],
            topic_hints=[],
            focus_terms=["apply"],
            expanded_terms=["apply"],
            expected_section_types=["eligibility"],
        )
        thread = ThreadState(
            thread_id="thread-1",
            recent_messages=[Message(role="user", content="Who can apply?")],
        )

        context = bot._build_answer_context(
            thread=thread,
            query_analysis=query_analysis,
            top_docs=[(document, 0.88)],
            top_sections=[(section, 0.91)],
        )

        self.assertNotIn("Score:", context)
        self.assertNotIn("File:", context)
        self.assertNotIn("Document orientation:", context)
        self.assertNotIn("Section orientation:", context)
        self.assertNotIn("Metadata: type=", context)
        self.assertNotIn("Section type:", context)
        self.assertNotIn("Retrieved document context:", context)
        self.assertNotIn("Question analysis:", context)
        self.assertNotIn("Evidence priority:", context)
        self.assertIn("Question:\nWho can apply?", context)
        self.assertIn("Evidence:\n[D1:S1 Eligibility]", context)

    def test_recent_chat_uses_compact_history_messages(self) -> None:
        config = Config(
            recent_chat_message_char_limit=60,
            include_document_orientation_in_answers=False,
            include_section_orientation_in_answers=False,
            include_document_metadata_in_answers=False,
            include_section_metadata_in_answers=False,
        )
        document, section = self._build_document_and_section()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(document)

        query_analysis = QueryAnalysis(
            original_question="Who can apply?",
            normalized_question="who can apply",
            canonical_question="User question: Who can apply?",
            detail_requested=False,
            multi_doc_expected=False,
            exact_match_expected=False,
            context_dependent=True,
            intents=["eligibility"],
            topic_hints=[],
            focus_terms=["apply"],
            expanded_terms=["apply"],
            expected_section_types=["eligibility"],
        )
        thread = ThreadState(
            thread_id="thread-1",
            recent_messages=[
                Message(
                    role="assistant",
                    content=(
                        "This is a prior answer that should keep only the useful part.\n\n"
                        "Reference: [Travel Policy](http://127.0.0.1:8010/api/documents/open?path=x)"
                    ),
                )
            ],
        )

        context = bot._build_answer_context(
            thread=thread,
            query_analysis=query_analysis,
            top_docs=[(document, 0.88)],
            top_sections=[(section, 0.91)],
        )

        self.assertIn("Recent:\nASSISTANT: This is a prior answer", context)
        self.assertNotIn("Reference:", context)
        self.assertNotIn("http://127.0.0.1:8010", context)

    def test_max_recent_messages_zero_removes_history_from_prompt(self) -> None:
        config = Config(
            max_recent_messages=0,
            include_document_orientation_in_answers=False,
            include_section_orientation_in_answers=False,
            include_document_metadata_in_answers=False,
            include_section_metadata_in_answers=False,
        )
        document, section = self._build_document_and_section()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(document)

        query_analysis = QueryAnalysis(
            original_question="Who can apply?",
            normalized_question="who can apply",
            canonical_question="User question: Who can apply?",
            detail_requested=False,
            multi_doc_expected=False,
            exact_match_expected=False,
            context_dependent=True,
            intents=["eligibility"],
            topic_hints=[],
            focus_terms=["apply"],
            expanded_terms=["apply"],
            expected_section_types=["eligibility"],
        )
        thread = ThreadState(
            thread_id="thread-1",
            recent_messages=[Message(role="assistant", content="Prior answer should be omitted.")],
        )

        context = bot._build_answer_context(
            thread=thread,
            query_analysis=query_analysis,
            top_docs=[(document, 0.88)],
            top_sections=[(section, 0.91)],
        )

        self.assertNotIn("Recent:\n", context)
        self.assertNotIn("Prior answer should be omitted.", context)

    def test_compact_history_message_strips_reference_suffix(self) -> None:
        config = Config()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config

        compact = bot._compact_history_message(
            "Answer body.\n\nReference: [Travel Policy](http://127.0.0.1:8010/api/documents/open?path=x)"
        )

        self.assertEqual(compact, "Answer body.")

    def test_system_prompt_is_compact_and_keeps_core_rules(self) -> None:
        bot = PolicyGPTBot.__new__(PolicyGPTBot)

        prompt = bot._system_prompt()

        self.assertLessEqual(estimate_text_tokens(prompt), 180)
        self.assertIn("provided evidence", prompt)
        self.assertIn("not clearly stated", prompt)
        self.assertIn("Prefer raw evidence blocks over summaries", prompt)
        self.assertIn("Use recent chat only", prompt)
        self.assertIn("Sources are shown separately", prompt)

    def test_aggregate_prompt_uses_minimal_context(self) -> None:
        config = Config(
            ai_profile="bedrock-claude-sonnet-4-6",
            include_document_orientation_in_answers=True,
            include_section_orientation_in_answers=True,
            include_document_metadata_in_answers=True,
            include_section_metadata_in_answers=True,
        )
        document, section = self._build_document_and_section()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(document)

        query_analysis = QueryAnalysis(
            original_question="Please list all contests",
            normalized_question="please list all contests",
            canonical_question="User question: Please list all contests",
            detail_requested=False,
            multi_doc_expected=True,
            exact_match_expected=False,
            context_dependent=False,
            intents=["aggregate"],
            topic_hints=["contest"],
            focus_terms=["contest"],
            expanded_terms=["contest", "list_all"],
            expected_section_types=["general"],
        )
        thread = ThreadState(thread_id="thread-1")

        context = bot._build_answer_context(
            thread=thread,
            query_analysis=query_analysis,
            top_docs=[(document, 0.88)],
            top_sections=[(section, 0.91)],
        )

        self.assertIn("Question:\nPlease list all contests", context)
        self.assertIn("Evidence:\n[D1:S1 Eligibility]", context)
        self.assertIn("Employees can submit after manager approval.", context)
        self.assertNotIn("Section type:", context)
        self.assertNotIn("Explains who can request travel", context)
        self.assertNotIn("This policy covers travel scope", context)
        self.assertNotIn("Doc notes:", context)
        self.assertNotIn("Requests without approval are not eligible", context)

    def test_open_weight_system_prompt_restores_full_rules(self) -> None:
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = Config(ai_profile="bedrock-120b")

        prompt = bot._system_prompt()

        self.assertIn("Rules:", prompt)
        self.assertIn("Do not hallucinate.", prompt)
        self.assertIn("Treat raw policy evidence blocks as the source of truth.", prompt)
        self.assertGreater(estimate_text_tokens(prompt), 180)

    def test_aggregate_answer_guidance_requests_short_description_per_item(self) -> None:
        bot = PolicyGPTBot.__new__(PolicyGPTBot)

        query_analysis = QueryAnalysis(
            original_question="Please list all contests",
            normalized_question="please list all contests",
            canonical_question="User question: Please list all contests",
            detail_requested=False,
            multi_doc_expected=True,
            exact_match_expected=False,
            context_dependent=False,
            intents=["aggregate"],
            topic_hints=["contest"],
            focus_terms=["contest"],
            expanded_terms=["contest"],
            expected_section_types=["general"],
        )

        guidance = bot._answer_format_guidance(query_analysis)

        self.assertIn("Name - short description", guidance)
        self.assertIn("Do not output bare names", guidance)

    def test_aggregate_section_filter_prefers_primary_contest_sections(self) -> None:
        bot = PolicyGPTBot.__new__(PolicyGPTBot)

        primary_section = SectionRecord(
            section_id="section-primary",
            title="Contest Name and Purpose OPEN CO-01",
            raw_text='The contest is named "Bali Bliss" and is issued by HDFC Life.',
            masked_text='The contest is named "Bali Bliss" and is issued by HDFC Life.',
            summary='Contest Name and Purpose: The contest is named "Bali Bliss".',
            summary_embedding=np.array([0.0], dtype=np.float32),
            source_path=r"D:\policy-mgmt\data\bali.html",
            doc_id="doc-bali",
            order_index=0,
            section_type="general",
        )
        comparison_section = SectionRecord(
            section_id="section-comparison",
            title="Higher-of Qualification Rule — Bali Bliss vs. Agency ACE EG-05",
            raw_text='An FC can qualify for the higher of "Bali Bliss" or "Agency ACE."',
            masked_text='An FC can qualify for the higher of "Bali Bliss" or "Agency ACE."',
            summary='Higher-of Qualification Rule for Bali Bliss vs Agency ACE.',
            summary_embedding=np.array([0.0], dtype=np.float32),
            source_path=r"D:\policy-mgmt\data\bali.html",
            doc_id="doc-bali",
            order_index=1,
            section_type="eligibility",
        )
        verification_section = SectionRecord(
            section_id="section-verification",
            title="Items Requiring Verification § S10",
            raw_text="Confirmation recommended with the issuing DPPM team.",
            masked_text="Confirmation recommended with the issuing DPPM team.",
            summary="Items Requiring Verification.",
            summary_embedding=np.array([0.0], dtype=np.float32),
            source_path=r"D:\policy-mgmt\data\other.html",
            doc_id="doc-other",
            order_index=0,
            section_type="general",
        )
        structure_section = SectionRecord(
            section_id="section-structure",
            title="Contest Structure Summary OV-02",
            raw_text='The contest has two components: "Early Bird Dhamaka" and "Main Contest."',
            masked_text='The contest has two components: "Early Bird Dhamaka" and "Main Contest."',
            summary='Contest Structure Summary with two components.',
            summary_embedding=np.array([0.0], dtype=np.float32),
            source_path=r"D:\policy-mgmt\data\dhurandhar.html",
            doc_id="doc-dhurandhar",
            order_index=0,
            section_type="general",
        )

        query_analysis = QueryAnalysis(
            original_question="Please list all contests",
            normalized_question="please list all contests",
            canonical_question="User question: Please list all contests",
            detail_requested=False,
            multi_doc_expected=True,
            exact_match_expected=False,
            context_dependent=False,
            intents=["aggregate"],
            topic_hints=["contest"],
            focus_terms=["contest"],
            expanded_terms=["contest"],
            expected_section_types=["general"],
        )

        filtered = bot._sections_for_answer_context(
            query_analysis,
            [
                (comparison_section, 0.95),
                (verification_section, 0.94),
                (primary_section, 0.88),
                (structure_section, 0.87),
            ],
        )

        filtered_titles = [section.title for section, _ in filtered]
        self.assertIn(primary_section.title, filtered_titles)
        self.assertIn(structure_section.title, filtered_titles)
        self.assertNotIn(comparison_section.title, filtered_titles)
        self.assertNotIn(verification_section.title, filtered_titles)

    def test_open_weight_aggregate_context_keeps_one_section_per_top_document(self) -> None:
        config = Config(
            ai_profile="bedrock-120b",
            include_document_orientation_in_answers=True,
            include_section_orientation_in_answers=True,
            include_document_metadata_in_answers=True,
            include_section_metadata_in_answers=True,
        )
        embedding = np.array([0.0], dtype=np.float32)
        doc_primary = DocumentRecord(
            doc_id="doc-primary",
            title="Primary Contest",
            source_path=r"D:\policy-mgmt\data\primary.html",
            raw_text="Primary contest text.",
            masked_text="Primary contest text.",
            summary="Primary contest summary.",
            summary_embedding=embedding,
            sections=[],
        )
        doc_secondary = DocumentRecord(
            doc_id="doc-secondary",
            title="Secondary Contest",
            source_path=r"D:\policy-mgmt\data\secondary.html",
            raw_text="Secondary contest text.",
            masked_text="Secondary contest text.",
            summary="Secondary contest summary.",
            summary_embedding=embedding,
            sections=[],
        )
        primary_section = SectionRecord(
            section_id="section-primary",
            title="Contest Name and Purpose",
            raw_text='The contest is named "Primary Contest".',
            masked_text='The contest is named "Primary Contest".',
            summary='Contest Name and Purpose: "Primary Contest".',
            summary_embedding=embedding,
            source_path=doc_primary.source_path,
            doc_id=doc_primary.doc_id,
            order_index=0,
            section_type="general",
        )
        secondary_negative_section = SectionRecord(
            section_id="section-secondary",
            title="Primary Eligible Audience EG-01",
            raw_text="The contest is applicable to all employees.",
            masked_text="The contest is applicable to all employees.",
            summary="Primary Eligible Audience EG-01.",
            summary_embedding=embedding,
            source_path=doc_secondary.source_path,
            doc_id=doc_secondary.doc_id,
            order_index=0,
            section_type="eligibility",
        )
        doc_primary.sections.append(primary_section)
        doc_secondary.sections.append(secondary_negative_section)

        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(doc_primary)
        bot.corpus.documents[doc_secondary.doc_id] = doc_secondary

        query_analysis = QueryAnalysis(
            original_question="Please list all contests",
            normalized_question="please list all contests",
            canonical_question="User question: Please list all contests",
            detail_requested=False,
            multi_doc_expected=True,
            exact_match_expected=False,
            context_dependent=False,
            intents=["aggregate"],
            topic_hints=["contest"],
            focus_terms=["contest"],
            expanded_terms=["contest"],
            expected_section_types=["general"],
        )
        thread = ThreadState(thread_id="thread-1")

        context = bot._build_answer_context(
            thread=thread,
            query_analysis=query_analysis,
            top_docs=[(doc_primary, 0.95), (doc_secondary, 0.90)],
            top_sections=[(primary_section, 0.95), (secondary_negative_section, 0.90)],
        )

        self.assertIn("Doc notes:", context)
        self.assertIn("Secondary Contest", context)
        self.assertIn("[D2:S1 Primary Eligible Audience EG-01]", context)

    def test_exact_context_filters_low_confidence_tail_sections(self) -> None:
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = Config(ai_profile="bedrock-120b")
        embedding = np.array([0.0], dtype=np.float32)
        strong_section = SectionRecord(
            section_id="section-strong",
            title="Reward Criteria",
            raw_text="Top 25 qualify with 100 percent achievement.",
            masked_text="Top 25 qualify with 100 percent achievement.",
            summary="Reward Criteria.",
            summary_embedding=embedding,
            source_path=r"D:\policy-mgmt\data\contest.html",
            doc_id="doc-1",
            order_index=0,
            section_type="general",
        )
        weak_section = SectionRecord(
            section_id="section-weak",
            title="Qualifier Definition RC-06",
            raw_text="Low-confidence tail section.",
            masked_text="Low-confidence tail section.",
            summary="Qualifier Definition.",
            summary_embedding=embedding,
            source_path=r"D:\policy-mgmt\data\contest.html",
            doc_id="doc-1",
            order_index=1,
            section_type="eligibility",
        )

        query_analysis = QueryAnalysis(
            original_question="What do I need to do?",
            normalized_question="what do i need to do",
            canonical_question="User question: What do I need to do?",
            detail_requested=False,
            multi_doc_expected=False,
            exact_match_expected=True,
            context_dependent=False,
            intents=["checklist"],
            topic_hints=[],
            focus_terms=["need", "do"],
            expanded_terms=["need", "do"],
            expected_section_types=["general"],
        )

        filtered = bot._sections_for_answer_context(
            query_analysis,
            [(strong_section, 1.20), (weak_section, 0.48)],
            None,
        )

        filtered_titles = [section.title for section, _ in filtered]
        self.assertIn(strong_section.title, filtered_titles)
        self.assertNotIn(weak_section.title, filtered_titles)

    def test_date_sensitive_prompt_includes_current_date_context(self) -> None:
        config = Config(
            include_document_orientation_in_answers=False,
            include_section_orientation_in_answers=False,
            include_document_metadata_in_answers=False,
            include_section_metadata_in_answers=False,
        )
        document, section = self._build_document_and_section()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(document)

        query_analysis = QueryAnalysis(
            original_question="Am I eligible today?",
            normalized_question="am i eligible today",
            canonical_question="User question: Am I eligible today?",
            detail_requested=False,
            multi_doc_expected=False,
            exact_match_expected=False,
            context_dependent=False,
            intents=["eligibility"],
            topic_hints=[],
            focus_terms=["eligible", "today"],
            expanded_terms=["eligible", "today"],
            expected_section_types=["eligibility"],
        )
        thread = ThreadState(thread_id="thread-1")

        with patch("policygpt.bot.datetime") as mock_datetime:
            mock_datetime.now.return_value.astimezone.return_value.date.return_value.isoformat.return_value = "2026-04-07"
            context = bot._build_answer_context(
                thread=thread,
                query_analysis=query_analysis,
                top_docs=[(document, 0.88)],
                top_sections=[(section, 0.91)],
            )

        self.assertIn("Current date: 2026-04-07", context)
        self.assertIn("Interpret relative dates such as today, current, and now against this date.", context)

    def test_non_date_sensitive_prompt_omits_current_date_context(self) -> None:
        config = Config(
            include_document_orientation_in_answers=False,
            include_section_orientation_in_answers=False,
            include_document_metadata_in_answers=False,
            include_section_metadata_in_answers=False,
        )
        document, section = self._build_document_and_section()
        bot = PolicyGPTBot.__new__(PolicyGPTBot)
        bot.config = config
        bot.redactor = _IdentityRedactor()
        bot.corpus = _StubCorpus(document)

        query_analysis = QueryAnalysis(
            original_question="Who can apply?",
            normalized_question="who can apply",
            canonical_question="User question: Who can apply?",
            detail_requested=False,
            multi_doc_expected=False,
            exact_match_expected=False,
            context_dependent=False,
            intents=["general"],
            topic_hints=[],
            focus_terms=["apply"],
            expanded_terms=["apply"],
            expected_section_types=["eligibility"],
        )
        thread = ThreadState(thread_id="thread-1")

        context = bot._build_answer_context(
            thread=thread,
            query_analysis=query_analysis,
            top_docs=[(document, 0.88)],
            top_sections=[(section, 0.91)],
        )

        self.assertNotIn("Current date:", context)

    def test_sanitize_answer_removes_internal_aliases(self) -> None:
        cleaned = PolicyGPTBot._sanitize_answer_for_user(
            "## Contests\n- Bali Bliss *(D2)*\n- **D3** — Dhurandhar\n[D1:S1] Details"
        )

        self.assertNotIn("D2", cleaned)
        self.assertNotIn("D3", cleaned)
        self.assertNotIn("D1:S1", cleaned)
        self.assertIn("Bali Bliss", cleaned)
        self.assertIn("Dhurandhar", cleaned)
        self.assertIn("Details", cleaned)

    def test_sanitize_answer_removes_reasoning_and_evidence_lines(self) -> None:
        cleaned = PolicyGPTBot._sanitize_answer_for_user(
            "<reasoning>hidden thoughts</reasoning>\n"
            "Contests mentioned\n"
            "Evidence: D1:S1.E1 - internal citation\n"
            "Source: File A\n"
            "- Bali Bliss\n"
            "Reference: File B"
        )

        self.assertEqual(cleaned, "Contests mentioned\n- Bali Bliss")

    def test_sanitize_answer_removes_bare_section_evidence_labels(self) -> None:
        cleaned = PolicyGPTBot._sanitize_answer_for_user(
            "S2.E1: the contest rewards a Sit Down Lunch with cricketer Yuvraj Singh."
        )

        self.assertEqual(cleaned, "the contest rewards a Sit Down Lunch with cricketer Yuvraj Singh.")

    def test_sanitize_answer_removes_empty_citation_brackets(self) -> None:
        cleaned = PolicyGPTBot._sanitize_answer_for_user(
            "Dhurandhar - The End Game【】\nBali Bliss []"
        )

        self.assertEqual(cleaned, "Dhurandhar - The End Game\nBali Bliss")

    @staticmethod
    def _build_document_and_section() -> tuple[DocumentRecord, SectionRecord]:
        embedding = np.array([0.0], dtype=np.float32)
        document = DocumentRecord(
            doc_id="doc-1",
            title="Travel Policy",
            source_path=r"D:\policy-mgmt\data\travel_policy.txt",
            raw_text="Employees can travel after approval.",
            masked_text="Employees can travel after approval.",
            summary="This policy covers travel scope, approvals, and reimbursement rules.",
            summary_embedding=embedding,
            sections=[],
            document_type="policy",
            version="1.0",
            effective_date="2026-01-01",
            metadata_tags=["travel", "reimbursement"],
            audiences=["employees"],
            keywords=["travel", "approval"],
            title_terms=["travel", "policy"],
        )
        section = SectionRecord(
            section_id="section-1",
            title="Eligibility",
            raw_text="Employees can submit a request after manager approval.",
            masked_text="Employees can submit a request after manager approval.",
            summary="Explains who can request travel and the approval precondition.",
            summary_embedding=embedding,
            source_path=document.source_path,
            doc_id=document.doc_id,
            order_index=0,
            section_type="eligibility",
            metadata_tags=["travel", "eligibility"],
            keywords=["request", "approval"],
            title_terms=["eligibility"],
        )
        document.sections.append(section)
        return document, section


if __name__ == "__main__":
    unittest.main()
