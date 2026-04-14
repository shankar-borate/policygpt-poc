"""Search layer — provider-agnostic hybrid retrieval.

Public surface area:
  VectorStore          — abstract interface (base.py)
  SearchQuery          — input DTO  (models.py)
  SearchResult         — output DTO (models.py)
  FaqResult            — FAQ hit DTO (models.py)
  SearchType           — enum       (models.py)
  HybridSearcher       — orchestrator (hybrid.py)
  OpenSearchRetriever  — corpus.py integration bridge (retriever.py)
  create_vector_store  — factory (factory.py)
"""

from policygpt.search.base import VectorStore
from policygpt.search.factory import create_vector_store
from policygpt.search.hybrid import HybridSearcher
from policygpt.search.models import FaqResult, SearchQuery, SearchResult, SearchType
from policygpt.search.retriever import OpenSearchRetriever

__all__ = [
    "VectorStore",
    "create_vector_store",
    "HybridSearcher",
    "OpenSearchRetriever",
    "FaqResult",
    "SearchQuery",
    "SearchResult",
    "SearchType",
]
