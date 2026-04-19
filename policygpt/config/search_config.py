"""Hybrid search and OpenSearch configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    hybrid_search_enabled: bool = True

    # "opensearch" — only supported provider currently.
    hybrid_search_provider: str = "opensearch"

    # OpenSearch connection — used when hybrid_search_provider = "opensearch".
    # None = opensearch disabled (hybrid search falls back to vector-only).
    opensearch_host: str | None = None
    opensearch_port: int = 9200

    # OpenSearch credentials. None = no authentication (anonymous access).
    opensearch_username: str | None = None
    opensearch_password: str | None = None

    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = False
    # Prefix for all OpenSearch indices created by this app.
    # Five indices are created: {prefix}_sections, {prefix}_documents,
    # {prefix}_faqs, {prefix}_recipient, {prefix}_threads.
    # Change this to isolate different deployments on the same OpenSearch cluster.
    # Examples: "policygpt_prod" | "policygpt_staging" | "contest_v2"
    opensearch_index_prefix: str = "product7"

    # Blend weights — normalised at query time, do not need to sum to 1.0
    hybrid_keyword_weight: float = 0.30
    hybrid_similarity_weight: float = 0.20
    hybrid_vector_weight: float = 0.50
