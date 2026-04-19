"""File system and observability configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from policygpt.constants import FileExtension

_SUPPORTED_FILE_PATTERNS: tuple[str, ...] = tuple(f"*{ext.value}" for ext in FileExtension)


@dataclass
class StorageConfig:
    # Root folder where policy documents are read from.
    # Examples: r"D:\policy-mgmt\data\contest"  |  "/mnt/policies"
    document_folder: str = r"D:\policy-mgmt\data\contest"

    # Directory for debug artefacts (converted HTML, logs). None = disabled.
    debug_log_dir: str | None = None

    # Path to a supplementary facts file injected into every prompt. None = disabled.
    supplementary_facts_file: str | None = None

    supported_file_patterns: tuple[str, ...] = field(default_factory=lambda: _SUPPORTED_FILE_PATTERNS)
    excluded_file_name_parts: tuple[str, ...] = field(default_factory=lambda: ("_summary",))
    rewrite_save_to_disk: bool = True
    debug: bool = True

    # Base URL used when building public document links.
    # Example: "http://127.0.0.1:8012"  |  "https://policygpt.example.com"
    public_base_url: str = "http://127.0.0.1:8012"

    usd_to_inr_exchange_rate: float = 93.0
