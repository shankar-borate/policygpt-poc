"""LLM output token budgets and answer context flags.

Defaults match the 'high' accuracy baseline.
Config.__post_init__ overwrites these based on accuracy_profile and
runtime_cost_profile.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputConfig:
    doc_summary_max_output_tokens: int = 1200
    doc_summary_max_output_tokens_cap: int = 2400
    doc_summary_chunk_max_output_tokens: int = 660
    section_summary_max_output_tokens: int = 660
    chat_max_output_tokens: int = 2700
    conversation_summary_max_output_tokens: int = 750

    # None means "use the runtime_cost_profile default" (resolved in __post_init__)
    include_document_metadata_in_answers: bool | None = None
    include_section_metadata_in_answers: bool | None = None
    include_document_orientation_in_answers: bool | None = None
    include_section_orientation_in_answers: bool | None = None
