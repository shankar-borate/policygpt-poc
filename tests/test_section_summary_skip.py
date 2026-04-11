import unittest

import numpy as np

from policygpt.config import Config
from policygpt.core.corpus import DocumentCorpus
from policygpt.extraction.redaction import Redactor


class _StubExtractor:
    @staticmethod
    def extract(_path: str) -> tuple[str, list[tuple[str, str]]]:
        return (
            "Travel Policy",
            [
                (
                    "Eligibility",
                    "Employees can submit a travel request after manager approval and must attach receipts.",
                )
            ],
        )


class _StubAI:
    @staticmethod
    def embed_texts(texts: list[str]) -> list[np.ndarray]:
        return [
            np.array([1.0, float(len(text) or 1)], dtype=np.float32)
            for text in texts
        ]

    @staticmethod
    def llm_text(system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        raise AssertionError(
            f"Unexpected LLM call: {system_prompt[:40]!r} / {user_prompt[:40]!r} / {max_output_tokens}"
        )


class SectionSummarySkipTests(unittest.TestCase):
    def test_ingest_file_uses_fallback_summary_when_skip_is_enabled(self) -> None:
        corpus = self._build_corpus(Config(skip_section_summary=True, debug=False))
        corpus._create_document_summary = lambda **_kwargs: "Travel approval policy summary."
        corpus._write_ingestion_log = lambda _document: None

        def _unexpected_section_summary(**_kwargs):
            raise AssertionError("Section summary LLM path should not be called when skip_section_summary=True.")

        corpus._create_section_summary = _unexpected_section_summary

        status, detail = corpus.ingest_file(r"D:\policy-mgmt\data\travel_policy.html")

        self.assertEqual(status, "ingested")
        self.assertIn("1 sections", detail)
        section = next(iter(corpus.sections.values()))
        self.assertIn("Eligibility", section.summary)
        self.assertIn("manager approval", corpus.redactor.unmask_text(section.summary))

    def test_ingest_file_uses_llm_summary_when_skip_is_disabled(self) -> None:
        corpus = self._build_corpus(Config(skip_section_summary=False, debug=False))
        corpus._create_document_summary = lambda **_kwargs: "Travel approval policy summary."
        corpus._write_ingestion_log = lambda _document: None
        corpus._create_section_summary = lambda **_kwargs: "Generated retrieval summary."

        status, detail = corpus.ingest_file(r"D:\policy-mgmt\data\travel_policy.html")

        self.assertEqual(status, "ingested")
        self.assertIn("1 sections", detail)
        section = next(iter(corpus.sections.values()))
        self.assertEqual(section.summary, "Generated retrieval summary.")

    @staticmethod
    def _build_corpus(config: Config) -> DocumentCorpus:
        return DocumentCorpus(
            config=config,
            extractor=_StubExtractor(),
            ai=_StubAI(),
            redactor=Redactor({}),
        )


if __name__ == "__main__":
    unittest.main()
