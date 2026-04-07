import unittest
from unittest.mock import patch

from policygpt.config import ACCURACY_PROFILE_PRESETS, AI_PROFILE_PRESETS, Config


class AccuracyProfileTests(unittest.TestCase):
    def test_accuracy_profiles_apply_for_all_ai_profiles(self) -> None:
        for ai_profile, ai_preset in AI_PROFILE_PRESETS.items():
            for accuracy_profile in ("high", "medium", "low"):
                with self.subTest(ai_profile=ai_profile, accuracy_profile=accuracy_profile):
                    config = Config(ai_profile=ai_profile, accuracy_profile=accuracy_profile)
                    expected_accuracy = ACCURACY_PROFILE_PRESETS[accuracy_profile]

                    self.assertEqual(config.chat_model, ai_preset["chat_model"])
                    self.assertEqual(config.embedding_model, ai_preset["embedding_model"])
                    self.assertEqual(config.chat_max_output_tokens, expected_accuracy["chat_max_output_tokens"])
                    self.assertEqual(config.top_docs, expected_accuracy["top_docs"])
                    self.assertEqual(
                        config.conversation_summary_max_output_tokens,
                        expected_accuracy["conversation_summary_max_output_tokens"],
                    )

    def test_manual_knob_overrides_are_preserved(self) -> None:
        config = Config(
            ai_profile="bedrock-claude-sonnet-4-6",
            accuracy_profile="medium",
            chat_max_output_tokens=777,
            top_docs=9,
        )

        self.assertEqual(config.chat_max_output_tokens, 777)
        self.assertEqual(config.top_docs, 9)
        self.assertEqual(config.max_sections_to_llm, ACCURACY_PROFILE_PRESETS["medium"]["max_sections_to_llm"])

    def test_manual_knob_overrides_are_preserved_for_high_accuracy_profile(self) -> None:
        config = Config(
            accuracy_profile="high",
            doc_summary_input_token_budget=1800,
            doc_summary_max_output_tokens=220,
        )

        self.assertEqual(config.doc_summary_input_token_budget, 1800)
        self.assertEqual(config.doc_summary_max_output_tokens, 220)

    def test_skip_section_summary_defaults_true(self) -> None:
        config = Config()

        self.assertTrue(config.skip_section_summary)

    def test_from_env_supports_accuracy_profile_override(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POLICY_GPT_ACCURACY_PROFILE": "low",
            },
            clear=False,
        ):
            config = Config.from_env()

        self.assertEqual(config.accuracy_profile, "low")
        self.assertEqual(config.chat_max_output_tokens, ACCURACY_PROFILE_PRESETS["low"]["chat_max_output_tokens"])

    def test_from_env_supports_summary_budget_overrides(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POLICY_GPT_DOC_SUMMARY_INPUT_TOKEN_BUDGET": "1800",
                "POLICY_GPT_DOC_SUMMARY_COMBINE_TOKEN_BUDGET": "1200",
                "POLICY_GPT_SECTION_SUMMARY_INPUT_TOKEN_BUDGET": "900",
                "POLICY_GPT_MIN_RECURSIVE_SUMMARY_TOKEN_BUDGET": "150",
                "POLICY_GPT_DOC_SUMMARY_MAX_OUTPUT_TOKENS": "220",
                "POLICY_GPT_DOC_SUMMARY_CHUNK_MAX_OUTPUT_TOKENS": "110",
                "POLICY_GPT_SECTION_SUMMARY_MAX_OUTPUT_TOKENS": "105",
            },
            clear=False,
        ):
            config = Config.from_env()

        self.assertEqual(config.doc_summary_input_token_budget, 1800)
        self.assertEqual(config.doc_summary_combine_token_budget, 1200)
        self.assertEqual(config.section_summary_input_token_budget, 900)
        self.assertEqual(config.min_recursive_summary_token_budget, 150)
        self.assertEqual(config.doc_summary_max_output_tokens, 220)
        self.assertEqual(config.doc_summary_chunk_max_output_tokens, 110)
        self.assertEqual(config.section_summary_max_output_tokens, 105)

    def test_from_env_supports_skip_section_summary_override(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POLICY_GPT_SKIP_SECTION_SUMMARY": "0",
            },
            clear=False,
        ):
            config = Config.from_env()

        self.assertFalse(config.skip_section_summary)

    def test_unknown_accuracy_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Config(accuracy_profile="ultra")


if __name__ == "__main__":
    unittest.main()
