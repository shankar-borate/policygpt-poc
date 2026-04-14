"""policygpt.retrieval — query analysis, section retrieval, and reranking.

Layers
------
query_analyzer.py   Re-exports QueryAnalysis and QueryAnalyzer.
reranker.py         Wraps corpus section-reranking and diversity selection.
pipeline.py         RetrievalPipeline — orchestrates the full retrieval flow.

Usage
-----
from policygpt.retrieval import RetrievalPipeline

pipeline = RetrievalPipeline.from_corpus(corpus)
sections, analysis = pipeline.retrieve(question, user_id=user_id)
faq_answer = pipeline.faq_fastpath(question, user_id=user_id)
"""

from policygpt.retrieval.query_analyzer import QueryAnalysis, QueryAnalyzer, detect_conversational_intent
from policygpt.retrieval.reranker import Reranker
from policygpt.retrieval.pipeline import RetrievalPipeline

__all__ = [
    "QueryAnalysis",
    "QueryAnalyzer",
    "detect_conversational_intent",
    "Reranker",
    "RetrievalPipeline",
]
