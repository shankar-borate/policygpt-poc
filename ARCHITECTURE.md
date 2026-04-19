# PolicyGPT — Architecture Document

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Ingestion Pipeline](#4-ingestion-pipeline)
5. [Retrieval & Chat Pipeline](#5-retrieval--chat-pipeline)
6. [Core Components](#6-core-components)
7. [Data Models](#7-data-models)
8. [Vector Store & Hybrid Search](#8-vector-store--hybrid-search)
9. [User Profile & Domain Profiles](#9-user-profile--domain-profiles)
10. [API Layer](#10-api-layer)
11. [Configuration System](#11-configuration-system)
12. [Logging & Observability](#12-logging--observability)
13. [Key Design Decisions](#13-key-design-decisions)

---

## 1. System Overview

PolicyGPT is a **Retrieval-Augmented Generation (RAG)** system that lets users ask natural-language questions against a corpus of enterprise documents — HR policies, IT procedures, finance rules, insurance contest guidelines, and technical manuals.

**What it does:**

| Capability | Description |
|---|---|
| Multi-format ingestion | PDF, DOCX, PPTX, XLSX, HTML, plain text, images |
| Hybrid search | BM25 keyword + dense vector semantic search via OpenSearch |
| LLM answer generation | Evidence-grounded answers using Claude (Bedrock) or OpenAI |
| Conversation threads | Multi-turn chat with history, context carry-over, and persistence |
| FAQ fast-path | Pre-generated Q&A bypasses full retrieval for common questions |
| Domain profiles | Separate configurations for HR policies, insurance contests, technical docs |
| User profiles | Role-aware retrieval and answer generation per user |
| Access control | Per-user document access lists enforced at index time |

**Core principle:** Every factual claim in an answer must be traceable to retrieved document evidence. The system never fabricates numbers, dates, names, or policy rules.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client (Browser)                           │
│               Web UI  ·  REST API  ·  Document Viewer              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP
┌────────────────────────────────▼────────────────────────────────────┐
│                     FastAPI  (PolicyApiServer)                      │
│           routes/chat.py  ·  runtime.py  ·  WebUIRenderer          │
└────┬──────────────────────────────────────────────────────┬─────────┘
     │ chat / thread operations                             │ search
     │                                                      │
┌────▼──────────────────────────────────┐     ┌────────────▼─────────┐
│           PolicyGPTBot                │     │  OpenSearch          │
│  ─────────────────────────────────    │     │  (kNN + BM25 + MLT)  │
│  QueryAnalyzer                        │     └──────────────────────┘
│  DocumentCorpus                       │
│  ConversationManager                  │
│  AIService  (Claude / OpenAI)         │
└──────────────────┬────────────────────┘
                   │
      ┌────────────┴─────────────┐
      │                          │
┌─────▼──────┐        ┌──────────▼──────────┐
│ In-Memory  │        │  OpenSearch          │
│ Corpus     │        │  Vector Store        │
│ (docs +    │        │  (sections + FAQ +   │
│  sections) │        │   ACL per user)      │
└────────────┘        └─────────────────────┘
```

**Startup sequence:**

1. `ServerRuntime` creates a `PolicyGPTBot` with an empty corpus — API is immediately available.
2. A background thread runs `IngestionPipeline` — documents are processed and indexed progressively.
3. `GET /api/health` exposes ingestion progress so the UI can show a loading state.

---

## 3. Folder Structure

```
policygpt/
│
├── api/                          # HTTP layer
│   ├── routes/chat.py            # All FastAPI route handlers
│   ├── runtime.py                # Two-phase startup, thread-safe bot access
│   └── renderers/                # HTML template rendering (UI + document viewer)
│
├── common/models/                # Canonical data models
│   ├── documents.py              # DocumentRecord, SectionRecord
│   ├── conversation.py           # Message, ThreadState
│   └── retrieval.py              # SourceReference, ChatResult, QueryAnalysis
│
├── config/
│   ├── settings.py               # Config (frozen dataclass, ~100 fields)
│   ├── user_profiles.py          # UserProfile, domain defaults, resolve_user_profile()
│   └── opensearch.env            # Local env file for secrets
│
├── constants/                    # Enums: ConversationalIntent, QueryIntent, FileExtension
│
├── core/
│   ├── bot.py                    # PolicyGPTBot — main orchestrator
│   ├── corpus.py                 # DocumentCorpus — document store + retrieval
│   ├── conversations.py          # ConversationManager — thread lifecycle
│   ├── ai/                       # AIService interface + OpenAI / Bedrock providers
│   ├── domain/                   # DomainProfile per domain (policy, contest, product_technical)
│   └── retrieval/
│       └── query_analyzer.py     # QueryAnalyzer — intent detection, query expansion
│
├── ingestion/
│   ├── pipeline.py               # IngestionPipeline — orchestration
│   ├── readers/                  # FolderReader, ApiReader, SqsReader
│   ├── converters/               # PDF / DOCX / PPTX / Excel / image → HTML
│   ├── pipeline_extractors/      # HTML / PDF / text / PPT / image → ExtractedDocument
│   ├── enrichment/               # Summarizer, FaqGenerator, EntityEnricher
│   ├── extraction/               # MetadataExtractor, Redactor, OCR
│   └── rewriter/                 # PolicyRewriter — optional HTML enhancement
│
├── search/
│   ├── base.py                   # VectorStore ABC (provider-agnostic interface)
│   ├── hybrid.py                 # HybridSearcher (blends BM25 + MLT + vector)
│   ├── retriever.py              # OpenSearchRetriever — bridge to corpus
│   └── providers/opensearch/     # OpenSearch implementation + index mappings
│
├── storage/                      # Thread persistence (OpenSearch-backed)
│
├── observability/
│   ├── usage_metrics.py          # LLMUsageTracker — tokens, cost (USD / INR)
│   └── debug_logging.py          # LLM request / response debug logs
│
├── factory.py                    # Bot + pipeline factories
└── cli.py                        # CLI entry point
```

---

## 4. Ingestion Pipeline

The ingestion pipeline transforms raw files on disk into enriched, indexed, searchable records. It runs once at startup (or on demand) in a background thread.

### 4.1 End-to-End Flow

```
FolderReader
  │
  │  yields IngestMessage per file
  │    { content: bytes, content_type, source_path, user_ids, domain }
  ▼
HtmlConverterRegistry  (optional, controlled per format in Config)
  │
  │  PDF  → HTML  (preserves tables and headings)
  │  DOCX → HTML
  │  PPTX → HTML  (one slide per section)
  │  XLSX → HTML tables
  │  IMG  → HTML via OCR  (AWS Textract)
  │
  │  Converted files cached on disk — re-ingest skips conversion
  ▼
ExtractorRegistry
  │
  │  HtmlExtractor  — splits on <h1>…<h6> tag boundaries
  │  PdfExtractor   — splits on page breaks + heading detection
  │  TextExtractor  — splits on blank lines / heuristics
  │  PptExtractor   — one section per slide
  │  ImageExtractor — OCR text as single section
  │
  │  yields ExtractedDocument { title, sections[] }
  ▼
PolicyRewriter  (optional: rewrite_policies_enabled)
  │
  │  Adds metadata block, table of contents, overview, roles table
  │  Does NOT modify original legal / policy language
  │  Output saved to {debug_log_dir}/improved/
  ▼
MetadataExtractor
  │  version, effective_date, document_type, audiences, keywords
  ▼
Redactor
  │  Masks sensitive terms before embedding  (config.redaction_rules)
  │  e.g. "Kotak" → "KKK" so brand names don't leak into semantic vectors
  ▼
Summarizer  [LLM call]
  │
  │  Document-level hierarchical summary
  │    Large doc → chunk summaries → combine → finalize (token-budgeted)
  │
  │  Per-section summaries
  │    "Capture EXACT amounts, thresholds, dates verbatim."
  ▼
FaqGenerator  [LLM call]
  │
  │  Generates up to 30 Q&A pairs per document
  │  Short concise answers + 10 paragraph-style extended answers
  │  Embeds each question separately (for cosine fast-path at query time)
  │  faq_qa_pairs: list[ (question, answer) ]
  ▼
EntityEnricher  [LLM call]
  │
  │  Extracts: roles, locations, time_periods, thresholds, policies,
  │            benefits, actions, processes, abbreviations
  │  Categories are domain-specific (from DomainProfile.entity_categories)
  │  Builds entity_lookup: { normalized_name → ExtractedEntity }
  │    used at query time to expand user terms with contextual synonyms
  ▼
DocumentCorpus.index_enriched_document()
  │
  ├─ Store DocumentRecord in  corpus.documents  { doc_id → DocumentRecord }
  ├─ Store SectionRecord[] in corpus.sections   { section_id → SectionRecord }
  ├─ OpenSearchVectorStore.index_document()    [sections index in OS]
  └─ OpenSearchVectorStore.index_faq_pairs()   [FAQ question embeddings in OS]
```

### 4.2 What Gets Stored Per Document

| Data | Storage | Used For |
|---|---|---|
| Raw text + sections | In-memory corpus | Evidence extraction, snippet building |
| Section summary embeddings | OpenSearch kNN index | Semantic retrieval |
| Document summary | In-memory + OS | Document-level matching |
| FAQ Q&A pairs | In-memory + OS | Fast-path lookup, aggregate context |
| FAQ question embeddings | OS kNN index | Cosine fast-path |
| Entity map | In-memory | Query-time term expansion |
| Metadata (audiences, type, version) | In-memory + OS | Scoring boosts, filters |

### 4.3 Caching and Incremental Re-ingestion

- **Converted HTML** cached in `{debug_log_dir}/html/` — unchanged files skip the converter.
- **OS cache check** — `document_indexed_for_path(source_path)` returns True if the document is already in OpenSearch; LLM enrichment is skipped for unchanged files.
- **Policy rewrites** saved to `{debug_log_dir}/improved/` for inspection.

### 4.4 Key Ingestion Config

| Field | Default | Purpose |
|---|---|---|
| `to_html_enabled` | true | Master switch for format conversion |
| `pdf_to_html_enabled` | true | PDF → HTML conversion |
| `rewrite_policies_enabled` | false | Optional policy HTML enhancement |
| `generate_faq` | true | FAQ Q&A generation |
| `faq_max_questions` | 30 | Q&A pairs per document |
| `faq_max_output_tokens` | 6000 | Token budget for FAQ generation |
| `generate_entity_map` | true | Named entity extraction |
| `ocr_enabled` | false | OCR for image files |
| `ocr_provider` | textract | AWS Textract or local OCR |

---

## 5. Retrieval & Chat Pipeline

Every user message passes through a sequential pipeline from intent detection to final answer delivery.

### 5.1 Full Chat Flow

```
User Question
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 1 — Conversational Intent Detection  (pattern-based, zero cost)
  │
  │  detect_conversational_intent()
  │    ├─ greeting / farewell / thanks    → canned reply, done
  │    ├─ identity ("who are you?")       → domain identity reply, done
  │    ├─ chitchat                        → polite deflect, done
  │    └─ self_referential               → _self_referential_reply()
  │         "why didn't you mention X?"    reads thread history, LLM explains gap
  │
  │  ~40 substring patterns cover common self-referential phrasings.
  │  This avoids running RAG for conversational follow-ups.
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 2 — Query Analysis
  │
  │  QueryAnalyzer.analyze(
  │      user_question, active_document_titles,
  │      candidate_documents, entity_lookup, user_profile
  │  ) → QueryAnalysis
  │
  │  Outputs:
  │    • intents: EXACT_MATCH | AGGREGATE | TIMELINE | MULTI_DOC |
  │               DOCUMENT_LOOKUP | NATURAL
  │    • context_dependent  (follow-up referencing previous answer?)
  │    • focus_terms         (key terms extracted from question)
  │    • expanded_terms      (focus_terms + entity synonyms + user_profile.tags)
  │    • topic_hints         (inferred document topics)
  │    • multi_doc_expected  (question spans multiple documents)
  │    • canonical_question  (normalized + user context line appended)
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 3 — Embed Question
  │
  │  _embed_one(user_question) → query_vec  (L2-normalized numpy array)
  │
  │  Raw question only — NOT the expanded canonical_question.
  │  Expansion is used for BM25/term matching, not for embedding.
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 4 — FAQ Fast-Path
  │
  │  faq_fastpath_lookup(query_vec, min_score=0.92)
  │
  │  Cosine similarity against all pre-embedded FAQ questions.
  │  If score ≥ 0.92 → return stored FAQ answer directly.
  │  Skips full retrieval + LLM generation. (milliseconds vs seconds)
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 5 — Answer Cache Lookup
  │
  │  Key: (normalized_question, frozenset(active_doc_ids))
  │  Context-dependent queries bypass cache (answer depends on prior turn).
  │  Cache hit → return stored answer immediately.
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 6 — Compound Question Detection
  │
  │  _split_compound_question(user_question)
  │
  │  "Who qualifies AND what is the reward?" → two sub-questions.
  │  Each sub-question: separate retrieval → merged context → one LLM call.
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 7 — Section Retrieval
  │
  │  corpus.retrieve_top_sections(
  │      query_vec, query_analysis,
  │      preferred_section_ids,   ← pinned for context-dependent follow-ups
  │      user_id, user_profile
  │  )
  │
  │  ┌──────────────────────────────────────────────────────────┐
  │  │  Hybrid Search Path  (when hybrid_search_enabled=True)   │
  │  │                                                          │
  │  │  OpenSearchRetriever.retrieve()                          │
  │  │    └─ HybridSearcher.search()                            │
  │  │         BM25 keyword match      weight 0.30              │
  │  │       + More-Like-This vocab    weight 0.20              │
  │  │       + Dense kNN vector        weight 0.50              │
  │  │         ACL filter: user must be in doc.user_id_access   │
  │  └──────────────────────────────────────────────────────────┘
  │  OR
  │  ┌─────────────────────────────────────────────────────────┐
  │  │  In-Memory Fallback  (when hybrid_search_enabled=False) │
  │  │  Cosine similarity against stored section embeddings     │
  │  └─────────────────────────────────────────────────────────┘
  │
  │  → candidate_sections  (up to rerank_section_candidates, default 12)
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 8 — Re-ranking
  │
  │  _rerank_sections(query_analysis, candidates, user_profile)
  │
  │  Heuristic scoring  (fast, no LLM):
  │
  │    h_score =
  │      0.72 × base_score          (OS hybrid score)
  │    + 0.14 × snippet_overlap     (query terms found in evidence snippets)
  │    + 0.08 × section_type_boost  (section type matches expected type)
  │    + 0.06 × focus_match         (focus terms appear in section text)
  │    + 0.08 × precise_match       (exact multi-word focus term matches)
  │    + 0.04 × tag_boost           (topic hints match section metadata_tags)
  │    + 0.04 × role_alignment      (user_profile.tags match section signals)
  │    + 0.02 × title_alignment     (focus terms appear in section / doc title)
  │
  │  If top h_score ≥ 0.82 → skip LLM rerank (confident enough already)
  │
  │  LLM rerank  (one call, up to 12 candidates):
  │    LLM scores each section 0.0–1.0 for relevance
  │    final_score = 0.70 × h_score + 0.30 × llm_score
  │
  │  score_details saved to corpus.last_retrieval_score_details
  │  (every component logged to retrieval debug file)
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 9 — Answerability Check
  │
  │  _is_answerable(query_analysis, top_docs, top_sections)
  │    Gate: best_score ≥ answerability_min_section_score (0.24)
  │          AND minimum support matches present
  │
  │  If NOT answerable:
  │    → _keyword_title_fallback_sections()    [second-pass retrieval]
  │        Scores sections by keyword hits in title / keywords / metadata
  │        Passes if: hits / len(query_terms) ≥ 0.50
  │
  │  If still not answerable:
  │    → _build_unanswerable_response()
  │        Shows topically related sections as hints
  │        Suggests FAQ questions
  │        Generates a clarifying question
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 10 — Dual Answer Check
  │
  │  _is_dual_answer_candidate(top_sections)
  │    If two top docs have close scores (score_b / score_a ≥ 0.88)
  │    AND query is not context-dependent AND not an exact-match query:
  │      → _build_dual_answer()   "Option A — DocX / Option B — DocY"
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 11 — Evidence Assembly
  │
  │  _build_prompt_payload(top_sections, query_analysis)
  │
  │    Per section:
  │      • Section title + document title  (orientation)
  │      • Section summary                 (context)
  │      • Raw text snippets               (up to 3 × 320 chars)
  │    Plus:
  │      • Supplementary facts from config file  (business rules)
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 12 — LLM Answer Generation
  │
  │  _generate_answer()
  │
  │  System prompt contains:
  │    • Domain context  (from DomainProfile)
  │    • User role context  (from UserProfile.context_line())
  │    • Answer format guidance  (intent-specific: bullets, table, prose)
  │    • Anti-hallucination rules
  │    • "No labels like Direct answer: or Summary:"
  │
  │  User prompt contains:
  │    • Recent conversation history  (max_recent_messages turns)
  │    • Evidence payload
  │    • User question
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 13 — Grounding Guard  (optional second LLM call)
  │
  │  grounding_guard_enabled = True  (config)
  │  Verifies every factual claim in the answer is supported by evidence.
  │  Flags or rewrites unsupported claims.
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 14 — Confidence Classification + Follow-up
  │
  │  HIGH    if best_score ≥ 0.55   (silent)
  │  MEDIUM  if best_score ≥ 0.38
  │  LOW     → _generate_low_confidence_followup()
  │              60-token nudge asking user to refine the question
  │
  ▼  ─────────────────────────────────────────────────────────────────────
  │  STEP 15 — Source Filtering + Thread Update
  │
  │  Drop sources scoring < best_score × 0.45
  │  Keep at least one source always.
  │
  │  Thread update:
  │    display_messages += [user_message, assistant_message]
  │    recent_messages  += [compacted history entries]
  │    active_doc_ids, active_section_ids, last_answer_sources updated
  │    title derived from first user message
  │
  │  ConversationManager.save_thread()
  │    Persist to OpenSearch (if storage configured)
  │    Clear display_messages from memory (loaded from OS on next read)
  │
  ▼  ─────────────────────────────────────────────────────────────────────

Return  ChatResult(thread_id, answer, sources)
```

### 5.2 Retrieval Sizing — Accuracy Profiles

| Profile | top_docs | top_sections/doc | max_sections_to_LLM | rerank_candidates |
|---|---|---|---|---|
| `vhigh` | 5 | 4 | 6 | 20 |
| `high` | 4 | 3 | 4 | 12 |
| `medium` | 3 | 2 | 3 | 8 |
| `low` | 2 | 2 | 2 | 6 |

Query-type overrides:
- **Exact match** — wider per-doc section window
- **Aggregate / multi-doc** — wider doc breadth (`broad_top_docs`)
- **Context-dependent** — pinned to previously active sections (preferred_section_ids)

### 5.3 LLM Call Budget Per Query

| Step | Calls | Approx tokens out |
|---|---|---|
| Conversational / self-referential reply | 0–1 | ~100 |
| LLM rerank (if heuristic score < 0.82) | 0–1 | ~150 |
| Main answer generation | 1 | up to 6000 |
| Grounding guard | 0–1 | ~20 |
| Clarifying question (unanswerable only) | 0–1 | ~60 |
| Low-confidence follow-up nudge | 0–1 | ~60 |
| **Typical per query** | **1–2** | |

---

## 6. Core Components

### 6.1 PolicyGPTBot (`core/bot.py`)

The central orchestrator. Owns the full chat loop from intent detection to response delivery.

**Key responsibilities:**
- Conversational intent bypass (greetings, farewells, self-referential queries)
- Compound question splitting and parallel retrieval
- Answer caching (in-memory, keyed by normalized question + active docs)
- FAQ fast-path lookup
- Dual-answer generation when two docs have close confidence scores
- Clarifying question generation on unanswerable queries
- Low-confidence follow-up nudge
- Retrieval debug logging

**Selected key methods:**

| Method | Purpose |
|---|---|
| `chat(thread_id, user_question, user_id, user_profile)` | Main entry point → ChatResult |
| `_embed_one(text)` | Embed question → L2-normalized vector |
| `_self_referential_reply(thread, question)` | Explain what the previous answer missed |
| `_keyword_title_fallback_sections(...)` | Second-pass retrieval on title keyword match |
| `_build_unanswerable_response(...)` | Suggest related sections + clarifying question |
| `_is_dual_answer_candidate(...)` | Check score ratio between top two docs |
| `_build_dual_answer(...)` | Generate Option A / Option B answer |
| `_system_prompt(user_profile)` | Build system prompt with role context injected |
| `_write_retrieval_log(...)` | Full score breakdown to debug file |

### 6.2 DocumentCorpus (`core/corpus.py`)

In-memory document store. Acts as both the ingestion sink and the retrieval source.

**Key storage:**

| Attribute | Type | Contents |
|---|---|---|
| `documents` | `dict[str, DocumentRecord]` | All documents keyed by doc_id |
| `sections` | `dict[str, SectionRecord]` | All sections keyed by section_id |
| `entity_lookup` | `dict[str, ExtractedEntity]` | Cross-doc entity expansion table |
| `last_retrieval_score_details` | `dict` | Score breakdown from last retrieval |

**Key methods:**

| Method | Purpose |
|---|---|
| `ingest_folder(path, ...)` | Iterate files, call `ingest_file()` per file |
| `ingest_file(message, ...)` | Extract → enrich → index a single document |
| `retrieve_top_sections(...)` | Delegate to hybrid or in-memory retrieval |
| `_rerank_sections(...)` | Heuristic + LLM reranking with score_details |
| `faq_fastpath_lookup(query_vec, ...)` | Cosine match against FAQ question embeddings |
| `rebuild_indexes()` | Post-ingestion: rebuild entity lookup, reset retriever |

### 6.3 QueryAnalyzer (`core/retrieval/query_analyzer.py`)

Converts a raw user question into a structured `QueryAnalysis` used throughout the retrieval layer.

**Steps:**
1. Check `_SELF_REFERENTIAL_SUBSTRINGS` (~40 patterns) → return early if self-referential
2. Normalize and tokenize the question
3. Detect query intent (EXACT_MATCH, AGGREGATE, MULTI_DOC, TIMELINE, DOCUMENT_LOOKUP)
4. Detect context-dependence (follow-up pronouns, "that document", "what about", etc.)
5. Extract focus terms and topic hints
6. Expand terms using `entity_lookup` (synonym enrichment from ingest-time entities)
7. Inject `user_profile.tags` into `expanded_terms`
8. Build `canonical_question` with user context line appended

**Caching:** Results cached per normalized question. Cache bypassed when `user_profile` is non-empty.

### 6.4 ConversationManager (`core/conversations.py`)

Manages thread lifecycle. Backed by OpenSearch when storage is configured, otherwise pure in-memory.

**Thread state carry-over for context-dependent queries:**
- `active_doc_ids` — which documents the last answer was from
- `active_section_ids` — pinned as preferred sections for retrieval
- `recent_messages` — rolling window of last N turns passed to LLM

**Auto-summarization:** When `summarize_after_turns` is reached, older turns are compressed into `conversation_summary` to keep the LLM context window bounded.

### 6.5 ServerRuntime (`api/runtime.py`)

Thread-safe wrapper around the bot. Manages two-phase startup.

**State machine:**

```
"starting"
    ↓  (bot created)
"ready"  ←────────────────────────────────────┐
    ↓  (background ingestion starts)          │
"ingesting"  (bot still usable, docs loading) │
    ↓  (ingestion completes)                  │
"ready"  ──────────────────────────────────────┘
    OR
"failed"  (unrecoverable ingestion error)
```

All HTTP routes call `require_bot()` which returns HTTP 503 while status is `"starting"`.

---

## 7. Data Models

### 7.1 SectionRecord — fundamental unit of retrieval

```
SectionRecord
  section_id          str       Stable unique ID
  title               str       Section heading
  raw_text            str       Verbatim content
  masked_text         str       Redacted copy (used before LLM calls)
  summary             str       LLM-generated summary
  summary_embedding   ndarray   L2-normalized vector (used for retrieval)
  source_path         str       File path (stable identifier)
  doc_id              str       Parent document ID
  order_index         int       Position within document
  section_type        str       "general" | "metadata" | "faq" | "table" | ...
  metadata_tags       list[str] Domain / content tags
  keywords            list[str] Extracted from title + content
  title_terms         list[str] Tokenized title words
  token_length        int       Token count for budget management
```

### 7.2 DocumentRecord

```
DocumentRecord
  doc_id              str
  title               str
  source_path         str
  summary             str       Hierarchical document summary
  summary_embedding   ndarray
  sections            list[SectionRecord]
  document_type       str       "policy" | "manual" | "procedure" | ...
  version             str
  effective_date      str
  metadata_tags       list[str]
  audiences           list[str] e.g. ["Branch Manager", "Employee"]
  keywords            list[str]
  faq                 str       Raw FAQ text
  faq_qa_pairs        list[tuple[str, str]]
  faq_q_embeddings    list[ndarray]
  entity_map          DocumentEntityMap
  original_source_path str      Pre-conversion path (PDF before HTML conversion)
```

### 7.3 ThreadState

```
ThreadState
  thread_id           str
  user_id             str
  recent_messages     list[Message]      LLM context window (rolling)
  display_messages    list[Message]      Full display history
  conversation_summary str               Compressed older history
  active_doc_ids      list[str]          Last referenced documents
  active_section_ids  list[str]          Last referenced sections
  title               str                Derived from first user message
  created_at          str                ISO 8601
  updated_at          str                ISO 8601
  last_answer_sources list[SourceReference]
```

### 7.4 QueryAnalysis

```
QueryAnalysis
  original_question    str
  normalized_question  str
  context_dependent    bool
  intents              set[QueryIntent]
  focus_terms          list[str]    Key terms from question
  expanded_terms       list[str]    Synonyms + entity expansions + profile tags
  topic_hints          list[str]    Inferred document topics
  expected_section_types list[str]
  multi_doc_expected   bool
  canonical_question   str          With user context line appended
```

---

## 8. Vector Store & Hybrid Search

### 8.1 Architecture

```
DocumentCorpus
  └─ OpenSearchRetriever
       └─ HybridSearcher
            └─ OpenSearchVectorStore
                 └─ OpenSearch cluster
                      ├─ {prefix}_sections  index
                      └─ {prefix}_documents index
```

### 8.2 Hybrid Search — How Scores Are Blended

Each section is scored by three independent signals, then combined:

| Signal | Mechanism | Default Weight |
|---|---|---|
| **Keyword (BM25)** | Exact + fuzzy term frequency matching | 0.30 |
| **Vocabulary (MLT)** | More-Like-This — document vocabulary overlap | 0.20 |
| **Semantic (kNN)** | Dense cosine similarity in embedding space | 0.50 |

```
hybrid_score = 0.30 × BM25  +  0.20 × MLT  +  0.50 × cosine
```

This catches both exact term matches (policy codes, clause numbers) AND paraphrased queries that share no literal vocabulary with the document.

### 8.3 OpenSearch Index — Section Fields

| Field | Type | Purpose |
|---|---|---|
| `section_id` | keyword | Unique ID |
| `title` | text | BM25 term matching |
| `raw_text` | text | BM25 + MLT |
| `summary_embedding` | knn_vector (1024 or 1536 dim) | Dense vector search |
| `metadata_tags` | keyword[] | Tag-based boosts and filters |
| `keywords` | text | BM25 matching |
| `user_id_access` | keyword[] | ACL filter — enforced at query time |
| `doc_id` | keyword | Join to parent document |
| `source_path` | keyword | Dedup, cache invalidation |

### 8.4 Access Control Model

- Every document has `user_id_access: list[str]` in the index.
- Admin users have `is_admin: true` — their queries skip ACL filtering.
- `ingestion_user_ids` in config sets default access for all ingested documents.
- ACL is a hard filter applied at the OpenSearch query level — unauthorized sections are never returned, not just ranked lower.

### 8.5 Provider Abstraction

`VectorStore` (`search/base.py`) is a pure interface. The OpenSearch implementation is one registered provider:

```python
_PROVIDER_REGISTRY = {
    "opensearch": "policygpt.search.providers.opensearch.store.OpenSearchVectorStore",
    # Future: "pinecone", "weaviate", "pgvector"
}
```

When `hybrid_search_enabled=False`, the factory returns `None` and the corpus falls back to in-memory cosine similarity — no OpenSearch dependency required.

---

## 9. User Profile & Domain Profiles

### 9.1 UserProfile (`config/user_profiles.py`)

Describes the user making the request. Biases retrieval and tailors the answer tone.

```python
@dataclass
class UserProfile:
    role: str         # "Branch Manager", "Sales Agent", "Software Engineer"
    grade: str        # "M3", "E4", "Senior"
    department: str   # "Retail Banking", "Agency Sales"
    location: str     # "Mumbai", "Remote"
    tags: tuple[str]  # Auto-built token set from all fields above
```

**Three injection points:**

| Where | What happens |
|---|---|
| `QueryAnalyzer.analyze()` | `user_profile.tags` appended to `expanded_terms`; context line added to `canonical_question` |
| `_rerank_sections()` | `role_alignment` boost — profile tags matched against section `metadata_tags`, `keywords`, `doc.audiences` |
| `_system_prompt(user_profile)` | `user_profile.context_line()` ("Branch Manager \| Retail Banking") injected into LLM system prompt |

### 9.2 Domain Default Profiles

Used until real per-user login is wired in. Represents the typical user population for each domain.

| Domain | Default Role | Department |
|---|---|---|
| `policy` | Employee | All Departments |
| `contest` | Financial Consultant | Agency Sales |
| `product_technical` | Software Engineer | Platform Engineering |

**Resolved at request time** in `api/routes/chat.py`:

```python
user_profile = resolve_user_profile(
    domain_type=self.config.domain_type,
    user_id=user_id,   # TODO: JWT profile lookup when login is wired
)
```

The `resolve_user_profile()` function is the single place to change when real login is added — no other code needs to change.

### 9.3 DomainProfile (`core/domain/`)

Drives domain-specific prompt language, entity categories, and UI copy. Separate from `UserProfile`.

| Field | Example (policy domain) |
|---|---|
| `domain_context` | "Documents are enterprise employee policy documents covering HR, IT, finance…" |
| `persona_description` | "an enterprise company's employees" |
| `doc_type_label` | "policy documents" |
| `entity_categories` | role, benefit, threshold, process, policy, abbreviation |
| `aggregate_response_hint` | How to list policies without duplication |
| `ui_prompt_chips` | "Leave entitlements", "Travel policy", "Approval matrix", … |

Built-in domains: `policy`, `contest`, `product_technical`. New domains register via `register("name", PROFILE)`.

---

## 10. API Layer

### 10.1 Routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve web UI |
| GET | `/api/health` | Status, ingestion progress, document/section counts |
| GET | `/api/usage` | LLM token usage and cost (USD + INR) |
| GET | `/api/domain` | Domain UI labels and prompt chips |
| GET | `/api/threads` | List threads for user |
| POST | `/api/threads` | Create new thread |
| GET | `/api/threads/{id}` | Get thread with full message history |
| POST | `/api/threads/{id}/reset` | Reset thread to blank state |
| POST | `/api/chat` | Send message, get answer + sources |
| GET | `/api/search` | Direct hybrid document search (paginated) |
| GET | `/api/documents/open` | Serve raw document file inline |
| GET | `/api/documents/view` | Render document viewer (iframe + section highlight) |

### 10.2 Chat Response Shape

```json
{
  "thread": {
    "thread_id": "abc123",
    "title": "Leave policy query",
    "messages": [
      { "role": "user",      "content": "What is the annual leave entitlement?" },
      { "role": "assistant", "content": "Employees receive 18 days of annual leave..." }
    ],
    "sources": [
      {
        "document_title": "HR Leave Policy",
        "source_path":    "D:/docs/HR_Leave_Policy.pdf",
        "file_name":      "HR_Leave_Policy.pdf",
        "document_url":   "/api/documents/open?path=...",
        "images":         []
      }
    ],
    "conversation_summary": ""
  },
  "answer": "Employees receive 18 days of annual leave per year..."
}
```

### 10.3 User Identity Resolution

User ID resolved in priority order:
1. `user_id` query parameter
2. `user_id` cookie
3. Empty string (anonymous)

When `hybrid_search_enabled=True`, a user_id is **required** — returns HTTP 401 otherwise, because OpenSearch ACL enforces per-user document access.

---

## 11. Configuration System

`Config` is a **frozen dataclass** — immutable after creation. All values are resolved at startup.

**Loading order:**

```
Environment variables
  → defaults in Config fields
  → accuracy_profile preset  (vhigh / high / medium / low)
  → runtime_cost_profile preset  (standard / aggressive)
  → domain-specific overrides
  → derived paths  (debug_log_dir, supplementary_facts_file)
```

**Key configuration groups:**

| Group | Key Fields |
|---|---|
| AI model | `ai_profile`, `chat_model`, `embedding_model`, `bedrock_region` |
| Ingestion | `document_folder`, `generate_faq`, `to_html_enabled`, `ocr_enabled` |
| Retrieval sizing | `top_docs`, `top_sections_per_doc`, `max_sections_to_llm` |
| Reranking weights | `section_semantic_weight` (0.36), `hybrid_keyword_weight` (0.30), etc. |
| Hybrid search | `hybrid_search_enabled`, `opensearch_host`, `opensearch_port` |
| Answerability | `answerability_min_section_score` (0.24), `confidence_high_score` (0.55) |
| Conversation | `max_recent_messages`, `summarize_after_turns` |
| FAQ fast-path | `faq_fastpath_enabled`, `faq_fastpath_min_score` (0.92) |
| Access control | `ingestion_user_ids` |
| Dual answer | `dual_answer_enabled`, `dual_answer_score_ratio` (0.88) |
| Follow-up | `followup_on_low_confidence` |
| Debug | `debug_log_dir`, `debug` |

**Factory:** `Config.from_env()` loads all values from environment variables with sensible defaults.

---

## 12. Logging & Observability

### 12.1 Retrieval Log

Written to `{debug_log_dir}/retrieval/{timestamp}_{thread_id}.txt` after every chat turn.

Contents:

```
Thread ID + User question

=== Query Analysis ===
  canonical_question (with user context line)
  intents, focus_terms, expanded_terms

=== Top Documents ===
  title | score | tags | audiences | summary (per doc)

=== Top Sections (final selection) ===
  For each section:
    doc_title :: section_title | final_score | file
    type | tags | summary
    snippets (up to 3)
    scores: os_hybrid | snippet_overlap | focus_match | precise_match |
            section_type_boost | tag_boost | role_alignment | title_alignment |
            h_score | llm_score | final_score

=== All Reranked Candidates ===
  All sections that went through reranking, sorted by final_score
  (not just the ones selected — useful for diagnosing misses)

=== Decision ===
  Answerable: yes / no

=== Prompt Payload ===
  Exact evidence text sent to LLM

=== Final Answer ===
=== Sources ===
```

### 12.2 LLM Debug Log

Written to `{debug_log_dir}/debug_*.jsonl`. Captures:
- System prompt + user prompt (exact text sent)
- Raw LLM response
- Token counts and latency

### 12.3 Usage Metrics

`LLMUsageTracker` aggregates per-session:
- Input / output tokens per model
- Cost in USD and INR (configurable `usd_to_inr_exchange_rate`)
- Request count and average latency

Exposed at `GET /api/usage`.

---

## 13. Key Design Decisions

### Two-Phase Startup
Bot is created and accepts requests immediately. Ingestion runs in the background. Users can start asking questions as soon as the first document is indexed — no cold-start wait. `/api/health` exposes fine-grained progress for the UI loading state.

### Hybrid Search over Pure Vector Search
Pure semantic search misses exact term queries (policy codes, clause numbers, thresholds like "₹3,49,000"). BM25 catches those; dense vector catches paraphrases. Combining at score level — not re-ranking stage — gives the best of both.

### Heuristic Rerank First, LLM Rerank Only When Needed
A second LLM call for reranking costs latency and money. The heuristic scorer (snippet overlap, focus term match, section type boost, role alignment) is fast and accurate enough to skip the LLM call when the top heuristic score is ≥ 0.82. This fires on the majority of queries.

### FAQ Fast-Path
For common, predictable questions (leave balance, expense limits, etc.) the full pipeline is unnecessary. Pre-generated FAQ answers with embedded questions allow a direct cosine lookup (threshold 0.92) that returns in milliseconds. FAQs are generated at ingest time, not per query.

### Role Alignment as a Boost, Not a Filter
User profile is applied as a **boost** in the reranker, not a hard filter. A section relevant to a "Branch Manager" still surfaces for an "Employee" — it just ranks slightly lower. Silently hiding documents from users based on role would undermine trust.

### In-Memory Corpus + Optional OpenSearch
The in-memory corpus (`documents` + `sections` dicts) is always present and always consistent. OpenSearch is the hybrid search layer on top. If OpenSearch is unavailable or disabled, the system degrades gracefully to in-memory cosine similarity with no code changes.

### Frozen Config
Making `Config` a frozen dataclass prevents accidental mutation during a request cycle. All profile-level and domain-level overrides are resolved once at startup. Every component reads the same immutable snapshot.

### Separate UserProfile from DomainProfile
`DomainProfile` describes the domain (prompt language, entity rules, UI copy) — static per deployment. `UserProfile` describes the individual user (role, grade, department) — varies per request. Keeping them separate lets domain configuration stay stable while user context is injected dynamically.

### Evidence Grounding Guard
Rather than trusting the LLM not to hallucinate, a lightweight second LLM call verifies that every factual claim (number, date, name, threshold) in the answer appears in the retrieved evidence. This is the last line of defense against fabricated policy details.

---

*Document reflects the PolicyGPT codebase as of April 2026.*
