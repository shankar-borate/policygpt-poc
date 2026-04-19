"""Hybrid search and OpenSearch configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    hybrid_search_enabled: bool = True
    hybrid_search_provider: str = "opensearch"

    # OpenSearch connection (used when hybrid_search_provider = "opensearch")
    opensearch_host: str = ""
    opensearch_port: int = 9200
    opensearch_username: str = ""
    opensearch_password: str = ""
    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = False
    opensearch_index_prefix: str = "product6"

    # Blend weights — normalised at query time, do not need to sum to 1.0
    hybrid_keyword_weight: float = 0.30
    hybrid_similarity_weight: float = 0.20
    hybrid_vector_weight: float = 0.50
