# PolicyGPT — Architecture Documentation

## What is PolicyGPT?

PolicyGPT is a **Retrieval-Augmented Generation (RAG) chatbot** built for an insurance company's agency sales channel. Sales agents (FCs, EIMs, ACHs, channel heads) ask natural-language questions about contest policies, reward structures, eligibility rules, thresholds, timelines, and locations. The bot finds the relevant sections across ingested policy documents and uses an LLM to produce a grounded, conversational answer.

**Key design principles:**
- All state is in-memory (Python dicts/lists). No external database in the current POC.
- Answers are grounded strictly in retrieved document evidence — no hallucination.
- Retrieval quality is prioritised over LLM cost at each design decision.
- OpenSearch is the planned next step for hybrid persistent search.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Client (Web / API)                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  HTTP  (FastAPI server)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PolicyGPTBot                                   │
│                           (bot.py)                                      │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐    │
│  │ Conversation │   │  Query       │   │   Answer Cache           │    │
│  │ Manager      │   │  Analyzer    │   │   (in-memory dict)       │    │
│  │(conversations│   │(query_       │   └──────────────────────────┘    │
│  │    .py)      │   │ analyzer.py) │                                    │
│  └──────────────┘   └──────────────┘                                    │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DocumentCorpus                                   │
│                          (corpus.py)                                    │
│                                                                         │
│   Documents ──► Sections ──► Embeddings ──► BM25 indexes               │
│   FAQ Q&A pairs + embeddings                                            │
│   Entity lookup (cross-document)                                        │
│                                                                         │
└───────────────────┬──────────────────────────────────┬──────────────────┘
                    │                                  │
                    ▼                                  ▼
        ┌───────────────────┐              ┌───────────────────┐
        │   AI Service      │              │  File Extractor   │
        │ (Bedrock / OpenAI)│              │ (file_extractor   │
        │                   │              │      .py)         │
        │ - LLM text gen    │              │                   │
        │ - Embeddings      │              │ HTML / TXT / PDF  │
        └───────────────────┘              └───────────────────┘
```

---

## Component Breakdown

### 1. PolicyGPTBot (`bot.py`)
The central orchestrator. Owns the conversation loop and coordinates all other components.

**Responsibilities:**
- Route each user message: conversational vs. policy question
- Manage conversation threads (history, active documents, topic tracking)
- Run the full RAG pipeline for policy questions
- Post-process answers: grounding check, confidence indicator, related questions
- Cache answers for repeated queries within a session

### 2. DocumentCorpus (`corpus.py`)
The in-memory document store and retrieval engine.

**Responsibilities:**
- Ingest documents: extract → summarise → FAQ → entities → embed → index
- Retrieve top documents and sections for a query using hybrid scoring
- LLM re-ranking of candidate sections
- FAQ fast-path lookup (exact match shortcut)
- Cross-corpus FAQ search for aggregate queries
- Evidence extraction from retrieved sections

### 3. QueryAnalyzer (`services/query_analyzer.py`)
Analyses the user's raw question before retrieval.

**Outputs (`QueryAnalysis`):**

| Field | What it contains |
|-------|-----------------|
| `original_question` | Raw user text |
| `canonical_question` | Cleaned, normalised form |
| `focus_terms` | Key search terms extracted |
| `expanded_terms` | Focus terms + synonyms + entity expansions |
| `topic_hints` | Domain topic tags (e.g. `eligibility`, `reward`) |
| `intents` | Detected intents: `aggregate`, `eligibility`, `timeline`, `comparison`, `document_lookup`, etc. |
| `exact_match_expected` | True for threshold/date/name lookups |
| `context_dependent` | True for follow-up questions referencing prior turns |
| `multi_doc_expected` | True for cross-document questions |
| `detail_requested` | True when user asked for "detailed"/"step by step" |

**Also performs:**
- Numeric normalisation: `₹3L` → `300000`, `3,49,000` → `349000`
- Entity expansion via the corpus entity lookup
- Document context injection for follow-up queries

### 4. FileExtractor (`services/file_extractor.py`)
Parses raw files into (title, sections) pairs.

**Supported formats:** HTML, HTM, plain text, PDF

**HTML processing:**
- Walks DOM using BeautifulSoup, emitting semantic units (headings, paragraphs, list items)
- Tables are extracted as atomic sections — never split mid-row — and converted to pipe-separated plain text
- Noise nodes (sidebars, nav, breadcrumbs) are filtered out
- Semantic units are grouped into sections bounded by `target_section_chars` (≈ 1800 chars)

### 5. AI Service (`services/bedrock_service.py` / `services/openai_service.py`)
Thin wrapper around the LLM provider.

**Methods:**
- `llm_text(system_prompt, user_prompt, max_output_tokens)` → string
- `embed_texts(texts)` → list of numpy arrays

**Supported providers:**

| Profile | Model | Embedding |
|---------|-------|-----------|
| `bedrock-120b` | GPT OSS 120B (via AWS Bedrock) | Amazon Titan v2 |
| `bedrock-20b` | GPT OSS 20B (via AWS Bedrock) | Amazon Titan v2 |
| `bedrock-claude-sonnet-4-6` | Claude Sonnet 4.6 (Bedrock) | Amazon Titan v2 |
| `bedrock-claude-opus-4-6` | Claude Opus 4.6 (Bedrock) | Amazon Titan v2 |
| `openai` | GPT-5.4 | text-embedding-3-large |

### 6. EntityExtractor (`services/entity_extractor.py`)
Extracts named entities from each document at ingest time.

**Entity categories:** `role`, `location`, `time_period`, `action`, `reward`, `threshold`, `product`, `contest`, `abbreviation`, `other`

**Output per entity:**
```
Role: FC (Financial Consultant — front-line sales agent; also: agent, advisor)
Contest: Bali Bliss (qualification contest with travel reward to Bali; also: bali contest)
Threshold: ₹3,49,000 FYFP (minimum first-year first-premium to qualify)
```

This enrichment text is embedded alongside the document summary, closing the gap between policy jargon and natural user queries.

### 7. ConversationManager (`conversations.py`)
Manages per-user conversation threads.

**ThreadState fields:**

| Field | Purpose |
|-------|---------|
| `recent_messages` | Last N Q+A turns (verbatim, for LLM context) |
| `display_messages` | Full conversation for UI rendering |
| `conversation_summary` | Compressed older history (kicks in after `summarize_after_turns`) |
| `active_doc_ids` | Documents referenced in the last answer |
| `active_section_ids` | Sections referenced in the last answer |
| `current_topic` | Derived topic label for the current conversation |
| `last_answer_sources` | Source references from last answer (used for follow-up resolution) |

### 8. Redactor (`services/redaction.py`)
Masks and unmasks sensitive brand terms before they reach the LLM.

```python
redaction_rules = {"Kotak": "KKK", "kotak": "KKK", "KOTAK": "KKK"}
```
All LLM calls receive masked text. Responses are unmasked before being shown to the user.

---

## Data Model

```
DocumentRecord
├── doc_id, title, source_path
├── raw_text, masked_text
├── summary (LLM-generated, for retrieval)
├── summary_embedding (numpy, L2-normalised)
├── faq  (raw Q&A text)
├── faq_qa_pairs  [(question, answer), ...]
├── faq_q_embeddings  [numpy, ...]  ← embeds "Q: ...\nA: ..." combined
├── entity_map  (DocumentEntityMap)
├── metadata: type, version, effective_date, tags, audiences, keywords
└── sections  [SectionRecord, ...]
        ├── section_id, title, source_path, doc_id, order_index
        ├── raw_text, masked_text
        ├── summary (LLM-generated)
        ├── summary_embedding (numpy, L2-normalised)
        ├── section_type  (general | table | faq | eligibility | ...)
        ├── metadata_tags, keywords, title_terms
        └── token_counts (for BM25)

ThreadState
├── thread_id, title, created_at, updated_at
├── recent_messages  [Message, ...]
├── display_messages  [Message, ...]
├── conversation_summary (compressed older history)
├── active_doc_ids, active_section_ids
├── current_topic, last_answer_sources
```

---

## Ingestion Pipeline

Triggered at server startup (`ingest_folder`). Processes each file sequentially.

```
File on disk
     │
     ▼
FileExtractor.extract()
     ├── HTML → BeautifulSoup DOM walk
     │         Tables → atomic sections (never split)
     │         Headings group paragraphs into sections
     ├── TXT  → paragraph-based chunking
     └── PDF  → text extraction → paragraph chunking
     │
     ▼
Full document text assembled
     │
     ├──► MetadataExtractor → type, version, effective_date, tags, audiences
     │
     ├──► DocumentSummary (LLM)
     │       "Summarise for retrieval. Capture: eligible roles, thresholds,
     │        reward amounts verbatim, timelines, locations, exceptions."
     │       Large docs → chunk → partial summaries → combine → finalise
     │
     ├──► FAQ Generation (LLM)  [if generate_faq=True]
     │       Up to 30 Q&A pairs a user would realistically ask.
     │       Answers include verbatim numbers, amounts, thresholds.
     │       → faq_qa_pairs parsed
     │       → each pair embedded as "Q: ...\nA: ..." (captures answer content)
     │
     ├──► EntityExtraction (LLM)  [if generate_entity_map=True]
     │       roles, locations, rewards, thresholds, abbreviations, contests
     │       → enrichment text for embedding
     │       → entity_lookup for query-time expansion
     │
     ├──► Document Embedding
     │       Combined text: title + summary + entity enrichment + FAQ + raw excerpt
     │       → L2-normalised numpy vector stored in DocumentRecord
     │
     └──► Per Section:
              ├── SectionSummary (LLM)
              │     "Capture EXACT amounts, thresholds, dates verbatim.
              │      For table sections: describe rows/columns in natural language."
              ├── Section Embedding
              │     Combined: title + summary + raw excerpt → L2-normalised vector
              └── SectionRecord stored with metadata, token_counts for BM25

After all files:
     rebuild_indexes()
          ├── doc_embedding_matrix  (numpy vstack, all doc vectors)
          ├── section_embedding_matrix  (numpy vstack, all section vectors)
          ├── doc_term_doc_freq  (BM25 document frequencies)
          ├── section_term_doc_freq  (BM25 section frequencies)
          └── entity_lookup  (flat cross-doc name → ExtractedEntity)
```

---

## Chat / Query Pipeline

Every user message goes through this flow:

```
User message
     │
     ▼
 ┌─────────────────────────────────────────────────────┐
 │ 1. Conversational Gate (pattern-based, zero cost)   │
 │    greeting / farewell / thanks / identity / chitchat│
 │    → LLM generates a natural short reply, DONE      │
 └──────────────────────────┬──────────────────────────┘
                            │ (policy question)
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 2. QueryAnalyzer.analyze()                          │
 │    • Numeric normalisation (₹3L → 300000)           │
 │    • Intent detection (aggregate / exact / follow-up)│
 │    • Topic hints, focus terms, expanded terms        │
 │    • Entity expansion via corpus entity_lookup       │
 │    • Context injection for follow-up queries         │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 3. Embed user question → query_vec (numpy)          │
 │    (raw question only, not the enriched query)       │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 4. FAQ Fast-Path  (cosine ≥ 0.92)                   │
 │    Scan all document FAQ embeddings.                 │
 │    If near-exact match found → return FAQ answer.   │
 │    Skips full RAG. DONE                             │
 └──────────────────────────┬──────────────────────────┘
                            │ (no fast-path hit)
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 5. Answer Cache Lookup                              │
 │    Key: (normalised_question, frozenset(active_docs))│
 │    Context-dependent queries → skip cache           │
 │    Cache hit → return stored answer. DONE           │
 └──────────────────────────┬──────────────────────────┘
                            │ (cache miss)
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 6. Compound Question Detection                      │
 │    "Who qualifies? And what is the reward?"         │
 │    Split on multiple "?" or "and what/who/how..."   │
 │    → each sub-question retrieved separately          │
 │    → merged context → single LLM answer call        │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 7. Retrieval  (see Retrieval System section)        │
 │    retrieve_top_docs → retrieve_top_sections         │
 │    → heuristic re-rank → LLM re-rank (if needed)   │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 8. Answerability Check                              │
 │    min section score, topic alignment, evidence     │
 │    match, exact evidence count                      │
 │                                                     │
 │    NOT answerable:                                  │
 │    → LLM classify intent (cheap call, 10 tokens)    │
 │    → Non-policy → conversational reply              │
 │    → Ambiguous → clarifying question (LLM, 60 tok)  │
 │    → Policy gap → "couldn't find" message           │
 └──────────────────────────┬──────────────────────────┘
                            │ (answerable)
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 9. Prompt Assembly (_build_answer_context)          │
 │    • Conversation summary + recent turns            │
 │    • Query brief + date context (if timeline query) │
 │    • Answer format guidance (per intent type)       │
 │    • Document index (aliases D1, D2…)              │
 │    • Evidence blocks per section:                   │
 │       [D1:S1 Section Title] [table|list|prose]      │
 │       Summary: ...                                   │
 │       Evidence: D1:S1.E1 ... D1:S1.E2 ...          │
 │    • For AGGREGATE queries: FAQ Q&A pairs from ALL  │
 │      docs (search_faq_questions), grouped by doc    │
 │    • Supplementary facts (injected background)      │
 │    • Negative evidence instruction                  │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 10. LLM Answer Generation (main call)               │
 │     System prompt tailored to model type            │
 │     (open-weight: 26 explicit rules)                │
 │     (claude: compact rules)                         │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 11. Post-Processing                                  │
 │     • Grounding Guard (LLM, if answer has numbers)  │
 │       → UNGROUNDED → adds disclaimer note           │
 │     • Confidence Indicator                          │
 │       score ≥ 0.55 → High (silent)                  │
 │       score ≥ 0.38 → "_Confidence: Medium_"         │
 │       else         → "_Confidence: Low_"            │
 │     • Related Questions (FAQ search, score ≥ 0.55)  │
 │       → "You might also ask: ..."                   │
 │     • Markdown normalisation                        │
 │       Pipe tables → HTML tables                     │
 │       Strip horizontal rules, deep headings         │
 │       Remove internal IDs (D1:S1.E1)               │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │ 12. Thread Update + Cache Store                     │
 │     • Source filtering (drop score < best×0.45)     │
 │     • Append reference links to answer              │
 │     • Update thread state (active docs, topic)      │
 │     • Trigger conversation summary if needed        │
 │     • Store in answer cache if eligible             │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
                      ChatResult to user
```

---

## Retrieval System (Deep Dive)

The retrieval system uses a **three-stage funnel**: broad candidate retrieval → heuristic re-scoring → conditional LLM re-ranking.

### Stage 1 — Document Retrieval

Hybrid score per document:

```
score = 0.48 × semantic_norm
      + 0.24 × bm25_norm
      + 0.16 × title_overlap_norm
      + 0.12 × metadata_score
      + 0.22 × document_lookup_score  (only for "find this document" queries)
      + 0.08  (preferred doc bonus for follow-up queries)
```

- **Semantic** — cosine similarity of query vector against doc embedding matrix (numpy matmul, fast)
- **BM25** — lexical term frequency with inverse document frequency; uses `expanded_terms` for query
- **Title overlap** — query terms appearing in document title or keywords
- **Metadata score** — topic tag alignment, audience match, document type match for intent

### Stage 2 — Section Retrieval

Per-doc candidates + global candidates (prevents missing relevant sections in low-ranked docs).

Hybrid score per section:

```
score = 0.36 × semantic_norm
      + 0.24 × bm25_norm
      + 0.16 × parent_doc_score
      + 0.12 × title_overlap_norm
      + 0.12 × metadata_score
      + 0.08 × focus_term_match
      + 0.12 × precise_focus_match  (exact phrase, multi-word term)
      + 0.08  (preferred section bonus)
```

### Stage 3 — Re-ranking

**Heuristic re-scoring** (zero LLM cost):
```
h_score = 0.72 × base_score
        + 0.14 × snippet_term_overlap
        + 0.08 × section_type_match
        + 0.04 × tag_alignment
        + 0.02 × title_alignment
        + 0.06 × focus_term_match
        + 0.08 × precise_match
```

**Conditional LLM re-ranking** (one call for up to 12 candidates):
- **Skipped** when top heuristic score ≥ 0.82 (already confident — saves cost)
- LLM scores each section 0.0–1.0 for relevance to the question
- Final score: `0.70 × heuristic + 0.30 × llm_score`

### Aggregate Query Path

For "list all X" / cross-document questions, the normal section evidence is replaced by **FAQ Q&A pairs** searched across all documents:

```
search_faq_questions(query_vec, top_k=30)
    → scores all FAQ embeddings across all docs
    → returns top-30 (score, question, answer, doc_title)
    → grouped by doc for clean prompt layout
```

This gives complete cross-document coverage without being limited by per-doc retrieval scoring.

---

## Answer Format Guidance

The system prompt varies by query intent:

| Intent / Condition | Format |
|-------------------|--------|
| User requested detail | Summary → Bullet details → Exceptions (structured) |
| User asked for table | HTML `<table>` output (never markdown pipes) |
| Comparison | Direct difference first, then bullets |
| Exact match | Answer in first line, raw evidence values |
| Aggregate / list all | Direct answer + one bullet per item with description |
| Eligibility | Who is eligible first, then conditions/exclusions |
| Timeline | Lead with date/deadline, then conditions |
| Approval / process | Ordered bullets with bold labels |
| Document lookup | Name the document, brief summary |

**Negative evidence rule** (all query types): if the evidence contains explicit exclusions, exceptions, or disqualifying conditions — include them. Do not silently omit negative conditions.

---

## Key Optimisations Summary

| Optimisation | Where | Benefit |
|-------------|-------|---------|
| FAQ fast-path (cosine ≥ 0.92) | `faq_fastpath_lookup` | Zero RAG for near-exact FAQ hits |
| Answer caching | `_answer_cache` in-memory dict | Zero LLM calls for repeated queries |
| Conditional re-rank skip (h ≥ 0.82) | `_rerank_sections` | Saves one LLM call when retrieval confident |
| Conditional grounding guard | `_check_answer_grounding` | Skip LLM verify for answers without numbers |
| Compound question split | `_split_compound_question` | Separate retrieval per sub-question → better coverage |
| Number normalisation | `normalize_numeric_expressions` | ₹3L and 300000 hit the same BM25 tokens |
| Q+A combined embedding | FAQ ingest | Answer content captured in FAQ vector (not just question phrasing) |
| Table-aware chunking | `FileExtractor` | Tables kept atomic — never split mid-row |
| Evidence content-type hints | `_build_answer_context` | LLM told whether evidence is table/list/prose |
| Source filtering | `chat()` | Drops sections < best×0.45 from reference list |
| Confidence indicator | `_compute_confidence` | User sees Medium/Low when retrieval weak |
| Related questions | `_find_related_questions` | Surfaces connected FAQ questions after each answer |
| Clarifying questions | `_generate_clarifying_question` | Asks one clarifying question for ambiguous queries |
| Date-aware prompting | `_current_date_prompt_line` | LLM flags passed/active/not-started windows |
| Grounding guard | `_check_answer_grounding` | Detects hallucinated numbers/dates |
| LLM intent classifier | `_llm_classify_intent` | Catches social messages that pattern matching missed |
| Structured long answers | `_answer_format_guidance` | Summary → Details → Exceptions for detail requests |
| Entity expansion | `entity_lookup` | User query "FC" expands to "Financial Consultant, agent, advisor" |
| Negative evidence instruction | prompt footer | Exclusions and disqualifications always surfaced |

---

## Configuration (`config.py`)

### Profile Selection

```python
ai_profile       = "bedrock-120b"    # Model stack
accuracy_profile = "vhigh"           # Retrieval breadth / context size
runtime_cost_profile = "standard"    # Per-request cost controls
```

### Key Config Knobs

| Setting | Default | Effect |
|---------|---------|--------|
| `faq_fastpath_enabled` | `True` | Toggle FAQ exact-match shortcut |
| `faq_fastpath_min_score` | `0.92` | Cosine threshold for FAQ fast-path |
| `aggregate_faq_top_k` | `30` | FAQ pairs passed as context for aggregate queries |
| `grounding_guard_enabled` | `True` | Toggle post-answer grounding check |
| `generate_faq` | `True` | Generate Q&A pairs at ingest |
| `generate_entity_map` | `True` | Extract named entities at ingest |
| `faq_max_questions` | `30` | Max Q&A pairs per document |
| `answerability_min_section_score` | `0.24` | Minimum score to consider a query answerable |
| `answerability_high_confidence_score` | `0.50` | Above this → answerable regardless of term overlap |
| `max_recent_messages` | `6–10` | Recent turns passed verbatim to LLM |
| `summarize_after_turns` | `8` | When to compress older history into a summary |

### Accuracy Profiles

| Profile | top_docs | sections/doc | re-rank candidates | Context size |
|---------|----------|-------------|-------------------|-------------|
| `vhigh` | 3 | 5 | 15 | Large |
| `high` | 3 | 3 | 12 | Medium |
| `medium` | 2 | 2 | 8 | Smaller |
| `low` | 2 | 2 | 5 | Minimal |

---

## LLM Call Budget (per user query — typical)

| Step | Calls | Tokens (approx) |
|------|-------|-----------------|
| Conversational reply | 0–1 | ~80 output |
| LLM intent classifier (fallback only) | 0–1 | ~10 output |
| LLM re-ranking (if not skipped) | 0–1 | ~150 output |
| Clarifying question (fallback only) | 0–1 | ~60 output |
| Main answer generation | 1 | up to 6000 output (120B) |
| Grounding guard (if answer has numbers) | 0–1 | ~20 output |
| **Total per query** | **1–4** | |

At ingest (one-time per document):

| Step | LLM Calls |
|------|-----------|
| Document summary | 1–4 (chunked for large docs) |
| FAQ generation | 1 |
| Entity extraction | 1 |
| Section summaries | 1 per section |

---

## Planned Next Step — OpenSearch

The in-memory corpus will be replaced by OpenSearch with:

- **Hybrid search**: keyword (BM25) + vector (k-NN) in a single query
- **Persistent storage**: no re-ingest on restart
- **Document-level access control**: `permitted_roles` / `permitted_branches` field on each document — all corpus operations will accept a `permitted_doc_ids` filter parameter, enforced at the index query level
- **Global FAQ and entity indexes**: separate OpenSearch indexes for cross-document FAQ search and entity lookup, filtered by the same access control

The current in-memory `entity_lookup`, `search_faq_questions`, and aggregate FAQ path all cross document boundaries — these will be gated by the permission filter when OpenSearch is introduced.

---

## Directory Structure

```
policygpt-poc/
├── policygpt/
│   ├── bot.py                  # Main orchestrator, chat logic
│   ├── corpus.py               # Document store, retrieval engine
│   ├── config.py               # All configuration with profiles
│   ├── models.py               # Data classes: DocumentRecord, SectionRecord, ThreadState
│   ├── conversations.py        # Conversation thread management
│   ├── document_links.py       # Source URL builder
│   └── services/
│       ├── base.py             # AIService abstract base
│       ├── bedrock_service.py  # AWS Bedrock LLM + embedding
│       ├── openai_service.py   # OpenAI LLM + embedding
│       ├── file_extractor.py   # HTML / TXT / PDF parsing
│       ├── query_analyzer.py   # Query analysis, intent detection
│       ├── entity_extractor.py # Named entity extraction
│       ├── metadata_extractor.py # Document/section metadata
│       ├── redaction.py        # Sensitive term masking
│       ├── taxonomy.py         # Domain terms, intent patterns, BM25 helpers
│       ├── usage_metrics.py    # LLM token cost tracking
│       ├── pricing_loader.py   # Model pricing data
│       └── debug_logging.py    # LLM call logging
├── web/
│   ├── index.html              # Chat UI
│   ├── app.js                  # Frontend: Markdown rendering, streaming
│   └── styles.css              # Chat and table styling
├── server.py                   # FastAPI server
├── ARCHITECTURE.md             # This document
└── README.md
```
