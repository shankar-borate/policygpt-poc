# Policy GPT Retrieval Logic

## Overview

This POC has been refactored into a small package named `policygpt` so the main responsibilities are separated:

- `policygpt/config.py`
  Holds runtime configuration such as document folder, model provider, model names, retrieval limits, and section sizing.
- `policygpt/services/`
  Contains reusable low-level services:
  `file_extractor.py`, `metadata_extractor.py`, `query_analyzer.py`, `openai_service.py`, `bedrock_service.py`, and `redaction.py`.
- `policygpt/corpus.py`
  Owns document ingestion, section generation, embeddings, and retrieval indexes.
- `policygpt/conversations.py`
  Owns thread state and thread lifecycle.
- `policygpt/bot.py`
  Orchestrates retrieval + answer generation for chat.
- `policygpt/server/`
  Owns FastAPI runtime state, route wiring, and UI injection.
- `policygpt/cli.py`
  Owns the terminal chat experience.

`app.py` is now only the server entrypoint.  
`policy_gpt_poc.py` is now only the CLI/backward-compatible wrapper.

## End-to-End Flow

## Model Providers

The app now supports two AI providers through config:

- `openai`
  Uses the configured OpenAI chat model plus the configured OpenAI embedding model.
- `bedrock`
  Uses Amazon Bedrock for both chat and embeddings.
  By config profile:
  - `ai_profile = "bedrock-20b"` maps to `openai.gpt-oss-20b-1:0`
  - `ai_profile = "bedrock-120b"` maps to `openai.gpt-oss-120b-1:0`
  - `ai_profile = "bedrock-claude-sonnet-4-6"` maps to `global.anthropic.claude-sonnet-4-6`
  - `ai_profile = "bedrock-claude-opus-4-6"` maps to `global.anthropic.claude-opus-4-6-v1`
  - embeddings use `amazon.titan-embed-text-v2:0`
  - Claude Bedrock profiles use the Bedrock `Converse` API; GPT-OSS Bedrock profiles keep using the OpenAI-compatible `InvokeModel` request format

The model switch is `Config.ai_profile` in `policygpt/config.py`.
The shared cost/quality switch is `Config.accuracy_profile`, with `high`, `medium`, and `low` presets that adjust retrieval breadth, evidence/context sizes, and LLM output budgets for every model.
`PolicyGPTBot` chooses the correct low-level AI service based on the resolved `Config.ai_provider`, so retrieval and chat orchestration do not need provider-specific branching.

### 1. Startup

When the web app starts:

1. `app.py` creates a `PolicyApiServer`.
2. `PolicyApiServer` creates a `ServerRuntime`.
3. During FastAPI lifespan startup, `ServerRuntime.start_indexing()` launches indexing in a background thread.

### 2. Indexing Flow

The indexing pipeline is handled by `DocumentCorpus` in `policygpt/corpus.py`.

For each supported policy file:

1. The corpus asks `FileExtractor` to read and parse the file.
2. The extractor:
   - strips scripts/styles/noisy HTML tags
   - extracts text from PDFs when the file is text-based
   - derives a document title
   - splits content into sections
3. The corpus derives metadata for each document and section, such as:
   - normalized titles
   - auto-derived policy/topic tags from titles and filenames
   - document type
   - section type
   - version/effective-date hints
   - lexical keyword terms
   The topic tags are now inferred from the document itself first. A small static synonym map is kept only as fallback for weak cases.
4. The corpus masks sensitive text using `Redactor`.
5. The corpus creates:
   - one retrieval summary for the whole document
   - one retrieval summary per section
6. The active AI service generates embeddings for those summaries.
7. The corpus stores:
   - `DocumentRecord`
   - `SectionRecord`
8. After all files are processed, the corpus builds:
   - a document embedding matrix
   - a section embedding matrix
   - lexical document-frequency stats for hybrid retrieval

### 3. Title Cleanup Logic

Some source HTML files contain misleading `<title>` values such as `Company car scheme`.

To avoid polluting retrieval:

- `_summary` files are excluded from indexing
- `FileExtractor` compares:
  - HTML title
  - first heading
  - file name
- generic or mismatched titles such as `Company car scheme` or `Table of Contents` are rejected
- the extractor falls back to the best title candidate

This keeps the indexed document names closer to the actual policy being queried.

### 4. Chat Flow

User chat is handled by `PolicyGPTBot` in `policygpt/bot.py`.

For every question:

1. The bot loads the current thread from `ConversationManager`.
2. It runs the user question through `QueryAnalyzer`, which infers:
   - likely policy topics from the indexed corpus first
   - answer intent such as checklist/process/eligibility/approval/timeline
   - explicit document-lookup intent for requests like `give me employee exit policy`
   - expanded search terms from matching document titles, tags, and keywords
3. It builds a richer retrieval query using:
   - conversation summary
   - current topic
   - active documents from the previous turn
   - inferred topics/intents
   - current user question
4. That retrieval query is embedded.
5. `DocumentCorpus.retrieve_top_docs()` finds the best matching documents using hybrid scoring.
6. `DocumentCorpus.retrieve_top_sections()` finds the best matching sections from those documents and reranks them.
7. The bot builds an answer prompt using:
   - recent chat
   - retrieved document summaries
   - retrieved section summaries
   - focused evidence snippets instead of dumping whole sections
   - answer-format guidance derived from the question type
8. Before answering, the bot runs an answerability check. If the evidence is weak, it returns a grounded "not clearly stated" style response instead of forcing an answer.
9. The active AI service generates the final answer.
10. The bot appends a `Reference:` line using the top source file names.
11. The thread state is updated with:
   - display messages
   - short-term memory
   - active docs/sections
   - conversation summary when needed

## Long Document Handling

Large PDFs and documents are now split in two places so indexing stays under model token limits:

1. Section chunking
   `FileExtractor` already groups content into sections. If a section is too large, it is split into synthetic parts. Oversized paragraphs are now further broken by sentence or word boundaries, so one giant PDF block does not become one giant LLM call.

2. Document summary reduction
   `DocumentCorpus` no longer sends the full document text to the LLM in one prompt when a file is large. Instead it:
   - splits the document text into summary-safe chunks
   - summarizes each chunk
   - recursively combines those chunk summaries if needed
   - creates one final compact document summary for retrieval

3. Token-limit recovery
   The corpus now uses a conservative token estimate based on both characters and words before sending text to the LLM. If the OpenAI API still reports that a request is too large, the corpus immediately splits that chunk again and retries with smaller pieces.

4. Skip-and-continue behavior
   If one document or one section still cannot be processed after recursive splitting, that item is skipped and indexing continues with the next file. One bad PDF should not stop the whole startup flow.

This means large text-based PDFs can be indexed without relying on a single oversized prompt, and indexing can keep moving even when one file is problematic.

## Retrieval Design

### Retrieval Inputs

The system does not embed the raw user question alone.
It embeds a richer retrieval query that includes conversation context.

This improves follow-up questions like:

- "what about this policy"
- "same policy"
- "what is the eligibility"
- "I am leaving company on 4th April. What process do I follow?"

The query analyzer now does two things:

- it uses the indexed corpus vocabulary to infer which policy family the question is most likely about
- it falls back to a small synonym map only when the corpus signals are weak

This makes the system more scalable as new policy domains are added, because new documents can contribute their own topic vocabulary without a code change.

Examples:

- `leaving company`, `last day`, `resignation`, `offboarding`, `clearance`
  can now match an exit/separation policy through either corpus terms or fallback synonym support
- `who approves`
  adds approval/matrix/delegation intent
- `checklist` or `process`
  biases retrieval toward sections that describe steps and required actions

### Retrieval Levels

Retrieval happens in two stages:

1. Document retrieval
   Finds the most relevant policy documents.
2. Section retrieval
   Finds the most relevant sections within those documents.

This keeps the answer context narrow while still allowing cross-document questions.

Both stages now use hybrid retrieval rather than pure semantic similarity.

### Scoring

Document ranking combines:

- semantic similarity
- lexical/BM25-style keyword matching
- title overlap
- exact title/phrase lookup boost for explicit policy-name requests
- metadata/topic matches
- a small continuity boost for active documents from the same thread

Section ranking combines:

- semantic similarity
- lexical/BM25-style keyword matching
- parent document score
- section-title overlap
- section/document metadata matches
- a reranking pass based on evidence-snippet alignment and section type

This helps maintain context across follow-up questions without locking the model into one document forever.

## Accuracy Controls

Accuracy is controlled in three places:

### 1. Cleaner indexing inputs

- generated summary files are excluded
- noisy HTML titles are corrected
- document and section metadata are extracted and normalized
- topic tags are inferred from document/section titles before falling back to static synonyms
- topic matching is fuzzy rather than exact, so a user topic like `exit` can still align with tags like `employee exit`
- title/version noise is reduced for matching

### 2. Narrow evidence selection

- only top documents are considered
- only top reranked sections from those documents are passed to the answer prompt
- evidence snippets are extracted from those sections instead of sending broad raw text

### 3. Answer prompt rules

The bot prompt explicitly tells the model to:

- answer only from provided evidence
- avoid hallucinating
- ignore semantically similar but off-topic evidence
- say "not clearly stated" when the evidence does not support the answer
- stay concise unless the user explicitly asks for detail

### 4. Answerability guardrail

Before the final answer call, the bot checks whether:

- section scores are strong enough
- evidence snippets actually match the query intent
- the top section still aligns with the inferred policy topic

If not, it returns a grounded fallback instead of guessing.

## Web Server Design

The server side is separated into focused classes:

- `ServerRuntime`
  Handles background indexing state, health status, and readiness.
- `WebUIRenderer`
  Injects small UI customizations into the static HTML.
- `PolicyApiServer`
  Wires routes, serialization, and FastAPI lifecycle together.

This means you can change API behavior without touching retrieval internals, and you can change UI injection without touching runtime or indexing logic.

## CLI Design

`PolicyGPTCli` owns the terminal interaction loop.

That keeps CLI concerns separate from:

- corpus/indexing
- conversation memory
- server concerns

## Best Extension Points

If you want to evolve this POC, these are the cleanest places to modify:

- Change extraction logic:
  `policygpt/services/file_extractor.py`
- Change retrieval sizing/scoring:
  `policygpt/corpus.py`
- Change answer style/prompt behavior:
  `policygpt/bot.py`
- Change thread behavior/memory:
  `policygpt/conversations.py`
- Change API/server lifecycle:
  `policygpt/server/api.py` and `policygpt/server/runtime.py`
- Change UI rendering behavior:
  `policygpt/server/ui.py`

## Current Limitations

This is still a POC, so a few scalability limitations remain:

- Thread state is in-memory only
- Index is rebuilt at startup
- PDF support is for text-based PDFs; scanned/image-only PDFs will not yield useful text without OCR
- No persistent vector store
- No async task queue for indexing
- No caching layer for summaries/embeddings
- No tenant/document access control layer

Those can now be added with much less churn because the code is split into clearer classes and modules.
