# Policy GPT Retrieval Logic

## Overview

This POC has been refactored into a small package named `policygpt` so the main responsibilities are separated:

- `policygpt/config.py`
  Holds runtime configuration such as document folder, model names, retrieval limits, and section sizing.
- `policygpt/services/`
  Contains reusable low-level services:
  `file_extractor.py`, `openai_service.py`, and `redaction.py`.
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
   - derives a document title
   - splits content into sections
3. The corpus masks sensitive text using `Redactor`.
4. The corpus creates:
   - one retrieval summary for the whole document
   - one retrieval summary per section
5. `OpenAIService` generates embeddings for those summaries.
6. The corpus stores:
   - `DocumentRecord`
   - `SectionRecord`
7. After all files are processed, the corpus builds:
   - a document embedding matrix
   - a section embedding matrix

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
2. It builds a retrieval query using:
   - conversation summary
   - current topic
   - active documents from the previous turn
   - current user question
3. That retrieval query is embedded.
4. `DocumentCorpus.retrieve_top_docs()` finds the best matching documents.
5. `DocumentCorpus.retrieve_top_sections()` finds the best matching sections from those documents.
6. The bot builds an answer prompt using:
   - recent chat
   - retrieved document summaries
   - retrieved section summaries
   - original section text
7. `OpenAIService` generates the final answer.
8. The bot appends a `Reference:` line using the top source file names.
9. The thread state is updated with:
   - display messages
   - short-term memory
   - active docs/sections
   - conversation summary when needed

## Retrieval Design

### Retrieval Inputs

The system does not embed the raw user question alone.
It embeds a richer retrieval query that includes conversation context.

This improves follow-up questions like:

- "what about this policy"
- "same policy"
- "what is the eligibility"

### Retrieval Levels

Retrieval happens in two stages:

1. Document retrieval
   Finds the most relevant policy documents.
2. Section retrieval
   Finds the most relevant sections within those documents.

This keeps the answer context narrow while still allowing cross-document questions.

### Scoring

Section ranking combines:

- section similarity
- parent document similarity
- a small continuity boost for active sections/documents from the same thread

This helps maintain context across follow-up questions without locking the model into one document forever.

## Accuracy Controls

Accuracy is controlled in three places:

### 1. Cleaner indexing inputs

- generated summary files are excluded
- noisy HTML titles are corrected

### 2. Narrow evidence selection

- only top documents are considered
- only top sections from those documents are passed to the answer prompt

### 3. Answer prompt rules

The bot prompt explicitly tells the model to:

- answer only from provided evidence
- avoid hallucinating
- ignore semantically similar but off-topic evidence
- say "not clearly stated" when the evidence does not support the answer
- stay concise unless the user explicitly asks for detail

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
- No persistent vector store
- No async task queue for indexing
- No caching layer for summaries/embeddings
- No tenant/document access control layer

Those can now be added with much less churn because the code is split into clearer classes and modules.
