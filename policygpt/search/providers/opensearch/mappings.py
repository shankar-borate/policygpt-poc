"""OpenSearch index mappings.

Two indices:
  {prefix}_sections   — one document per section (primary retrieval unit)
  {prefix}_documents  — one document per policy file (for document-level lookup)

The sections index uses:
  - A custom policy_analyzer (English stopwords + snowball stemmer) on all text
    fields so keyword and similarity searches handle plurals, verb forms, etc.
  - A knn_vector field on "embedding" for approximate nearest-neighbour search.
  - keyword sub-fields on title columns for exact aggregation / filtering.

Design note: knn=True must be set at index creation time in OpenSearch.  The
HNSW method (nmslib engine) is used for cosine similarity — it is the most
widely supported combination across OpenSearch versions.
"""

from __future__ import annotations


def sections_mapping(embedding_dim: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "analysis": {
                "analyzer": {
                    "policy_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                # ── Identity ──────────────────────────────────────────────────
                "section_id":  {"type": "keyword"},
                "doc_id":      {"type": "keyword"},

                # ── Text fields (keyword + similarity search) ─────────────────
                "document_title": {
                    "type": "text",
                    "analyzer": "policy_analyzer",
                    # keyword sub-field for exact filtering / aggregations
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "section_title": {
                    "type": "text",
                    "analyzer": "policy_analyzer",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "raw_text": {
                    "type": "text",
                    "analyzer": "policy_analyzer",
                },
                "summary": {
                    "type": "text",
                    "analyzer": "policy_analyzer",
                },

                # ── Facets / filters ──────────────────────────────────────────
                "section_type":   {"type": "keyword"},
                "metadata_tags":  {"type": "keyword"},
                "keywords":       {"type": "keyword"},
                "audiences":      {"type": "keyword"},
                "source_path":    {"type": "keyword"},
                "order_index":    {"type": "integer"},
                "domain":         {"type": "keyword"},

                # ── Images ────────────────────────────────────────────────────
                # Base64 data URIs for images found in this section.
                # Stored verbatim for retrieval but never indexed/searched.
                "images":         {"type": "object", "enabled": False},

                # ── Dense vector (vector search) ──────────────────────────────
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16,
                        },
                    },
                },
            }
        },
    }


def faqs_mapping(embedding_dim: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "analysis": {
                "analyzer": {
                    "policy_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "faq_id":          {"type": "keyword"},
                "doc_id":          {"type": "keyword"},
                "document_title": {
                    "type": "text",
                    "analyzer": "policy_analyzer",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "question": {
                    "type": "text",
                    "analyzer": "policy_analyzer",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}},
                },
                "answer":          {"type": "text", "analyzer": "policy_analyzer"},
                "source_path":     {"type": "keyword"},
                # Dense vector for question similarity search
                "question_embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                "domain":          {"type": "keyword"},
            }
        },
    }


def recipient_mapping() -> dict:
    """Mapping for the recipient index — one record per (user_id, doc_id) pair.

    Kept deliberately thin: no text analysis, no vectors — only keyword lookups
    and aggregations.  Shard count is higher than other indexes because with
    200k users × N documents the record count can be large.
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": 2,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "user_id":    {"type": "keyword"},
                "doc_id":     {"type": "keyword"},
                "granted_at": {"type": "date"},
            }
        },
    }


def documents_mapping() -> dict:
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "doc_id":         {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "document_type":  {"type": "keyword"},
                "version":        {"type": "keyword"},
                "effective_date": {"type": "keyword"},
                "metadata_tags":  {"type": "keyword"},
                "audiences":      {"type": "keyword"},
                "keywords":       {"type": "keyword"},
                "source_path":    {"type": "keyword"},
                "original_source_path": {"type": "keyword"},
                "summary":        {"type": "text", "analyzer": "standard"},
                "domain":         {"type": "keyword"},
            }
        },
    }
