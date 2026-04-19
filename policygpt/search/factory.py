"""VectorStore factory.

Returns the correct VectorStore implementation based on Config.
Adding a new provider requires only:
  1. Implement VectorStore in providers/<name>/store.py
  2. Add an elif branch here

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import logging
from typing import Optional

from policygpt.config import Config
from policygpt.search.base import VectorStore

logger = logging.getLogger(__name__)

# Registry maps provider key → import path of the store class.
# Add new providers here without touching any other file.
_PROVIDER_REGISTRY: dict[str, str] = {
    "opensearch": "policygpt.search.providers.opensearch.store.OpenSearchVectorStore",
    # Future providers — uncomment and implement to activate:
    # "pinecone":   "policygpt.search.providers.pinecone.store.PineconeVectorStore",
    # "weaviate":   "policygpt.search.providers.weaviate.store.WeaviateVectorStore",
    # "pgvector":   "policygpt.search.providers.pgvector.store.PgVectorStore",
}


def create_vector_store(config: Config) -> Optional[VectorStore]:
    """Instantiate and return the configured VectorStore, or None if disabled.

    Returns None when opensearch_enabled=False so corpus.py can fall back to
    the existing in-memory retrieval path without any code change.
    """
    if not config.search.hybrid_search_enabled:
        return None

    provider = config.search.hybrid_search_provider
    import_path = _PROVIDER_REGISTRY.get(provider)

    if import_path is None:
        raise ValueError(
            f"Unknown vector store provider: {provider!r}. "
            f"Available: {list(_PROVIDER_REGISTRY)}"
        )

    # Validate OpenSearch credentials are present before attempting connection
    if provider == "opensearch":
        if not config.search.opensearch_host:
            raise RuntimeError(
                "OS_HOST is not set. "
                "Copy opensearch.env.example to opensearch.env and fill in credentials."
            )
        if not config.search.opensearch_username or not config.search.opensearch_password:
            raise RuntimeError(
                "OS_USERNAME or OS_PASSWORD is not set. "
                "Copy opensearch.env.example to opensearch.env and fill in credentials."
            )

    store_class = _import_class(import_path)
    store: VectorStore = store_class(config)

    if not store.health_check():
        raise RuntimeError(
            f"Hybrid search provider '{provider}' is not reachable. "
            f"Check connection settings and credentials."
        )

    # Determine embedding dimension from the configured model
    embedding_dim = _resolve_embedding_dim(config)
    store.ensure_index(embedding_dim)

    logger.info("Vector store ready: provider=%s  index_prefix=%s", provider, config.search.opensearch_index_prefix)
    return store


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve_embedding_dim(config: Config) -> int:
    """Return the embedding vector dimension for the configured model."""
    model = (config.ai.embedding_model or "").lower()
    # Titan text embedding v2 → 1024; v1 → 1536; OpenAI ada-002 → 1536
    if "v2" in model and "titan" in model:
        return 1024
    return 1536
