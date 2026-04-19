# PolicyGPT — Improvement Roadmap
## Towards Enterprise-Grade AI Assistant (ChatGPT / Claude Quality)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Context Engineering](#2-context-engineering)
3. [Per Document Type Improvements](#3-per-document-type-improvements)
4. [Retrieval Quality Improvements](#4-retrieval-quality-improvements)
5. [Answer Generation Improvements](#5-answer-generation-improvements)
6. [Explainability & Trust](#6-explainability--trust)
7. [Multi-Modal Capabilities](#7-multi-modal-capabilities)
8. [Agentic & Reasoning Capabilities](#8-agentic--reasoning-capabilities)
9. [Enterprise Platform Features](#9-enterprise-platform-features)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Operational Excellence](#11-operational-excellence)
12. [Prioritised Roadmap](#12-prioritised-roadmap)

---

## 1. Executive Summary

PolicyGPT today is a solid RAG foundation. To reach the quality bar of enterprise ChatGPT or Claude — where every answer is accurate, explainable, fast, and trusted by employees at all levels — the following areas require investment.

The improvements fall into three horizons:

| Horizon | Theme | Expected Outcome |
|---|---|---|
| **H1 — 0–3 months** | Precision & trust | Grounded answers, inline citations, confidence visible to user |
| **H2 — 3–6 months** | Coverage & depth | Multi-modal docs, better chunking, conversational memory |
| **H3 — 6–12 months** | Enterprise platform | SSO, RBAC, audit trail, analytics, evaluation loop |

---

## 2. Context Engineering

Context Engineering is the discipline of deciding **what to put in the LLM's context window, how to structure it, and when to leave things out**. It is the single highest-leverage improvement area — bad context makes even the best LLM produce wrong answers.

### 2.1 Hierarchical Context Assembly

**Current state:** Evidence blocks are assembled by selecting top-N sections and appending them sequentially.

**Problem:** The LLM sees a flat wall of text. It cannot tell which part is most important or how sections relate to each other.

**Improvement:**

```
Context Window Structure
────────────────────────
[IDENTITY]
  You are an enterprise policy assistant for <Company>.

[USER CONTEXT]
  Role: Branch Manager | Department: Retail Banking | Location: Mumbai

[CONVERSATION SUMMARY]
  Previous 8 turns compressed into 3 sentences.

[RECENT TURNS]
  User: "What is the travel policy?" → Assistant: "..."
  User: "What about international travel?" → (current question)

[DOCUMENT INDEX]
  D1: HR Leave Policy v2.1 (effective Jan 2024) — HR domain
  D2: Travel & Expense Policy v1.4 (effective Mar 2024) — Finance domain

[PRIMARY EVIDENCE]  ← highest-scored section
  [D2 § 4.2 International Travel]
  Summary: Employees on international travel are eligible for...
  Evidence: Business class permitted for flights > 8 hours. Daily allowance $200 USD.

[SUPPORTING EVIDENCE]
  [D2 § 4.1 Domestic Travel]
  [D1 § 7 Travel Leave]

[NEGATIVE CONSTRAINT]
  The following sections were retrieved but are NOT relevant to this question.
  Do not draw conclusions from them: [D1 § 3 Annual Leave]

[SUPPLEMENTARY RULES]
  Company standard: All international travel requires VP approval.
```

**Why it works:** LLM models attend strongly to position. Primary evidence first means the most relevant content gets the most attention. Explicit negative constraints prevent the LLM from citing irrelevant sections that happened to be retrieved.

---

### 2.2 Dynamic Context Window Budget Management

**Current state:** Fixed `max_sections_to_llm` regardless of question complexity.

**Improvement:** Budget the context window dynamically based on:
- Question complexity score (simple lookup vs. cross-document comparison)
- Available token budget for the chosen model
- Section relevance score distribution (steep drop-off = fewer sections needed)

```
Available tokens = model_context_limit - system_prompt_tokens
                  - conversation_history_tokens - output_budget
Remaining = fill with evidence sections in score order until budget exhausted
```

This ensures complex multi-document questions get as much evidence as possible, while simple lookups stay lean.

---

### 2.3 Conversation Memory Architecture

**Current state:** Rolling window of last N messages passed verbatim.

**Problem:** After 8–10 turns, older context is silently dropped. The LLM forgets what was established earlier — user gets inconsistent answers.

**Improvement — Three-tier memory:**

| Tier | Content | Retention |
|---|---|---|
| **Working memory** | Last 4–6 turns verbatim | Always in context |
| **Episodic summary** | Turns 7–20 compressed by LLM | In context (condensed) |
| **Long-term facts** | Established facts ("user is asking about Mumbai branch") | Extracted, stored, always prepended |

Long-term facts are extracted after each turn and stored in `ThreadState.established_facts`:

```
"User confirmed they are a Branch Manager in Mumbai."
"User is asking about the Q3 2024 contest, not Q2."
"User has already been told they do not qualify under Scheme A."
```

These facts are injected at the top of every subsequent prompt — the LLM never contradicts an established fact.

---

### 2.4 Query Rewriting for Follow-ups

**Current state:** Follow-up questions like "what about for managers?" are sent to retrieval as-is. They often miss because the standalone question has no context.

**Improvement — Rewrite before embedding:**

```
Turn 1: "What is the annual leave entitlement?"
Turn 2: "What about for managers?"

Rewritten query for retrieval:
  "Annual leave entitlement for managers"
```

A cheap LLM call rewrites ambiguous follow-ups into self-contained retrieval queries. This dramatically improves recall for context-dependent questions.

---

### 2.5 Hypothetical Document Embedding (HyDE)

**Current state:** User question is embedded directly and matched against section embeddings.

**Problem:** Questions and answers live in different embedding spaces. "How many days leave do I get?" may not be close to "Annual leave entitlement: 18 days per year."

**Improvement:**

1. Generate a short hypothetical answer using a fast/cheap LLM call: *"Employees are entitled to 18 days of annual leave per calendar year, subject to service length."*
2. Embed the hypothetical answer (not the question).
3. Use that embedding for kNN retrieval.

The hypothetical answer embedding sits in the same semantic space as real document answers, dramatically improving retrieval precision for factual lookups.

---

### 2.6 Persona-Aware Prompt Calibration

**Current state:** User profile is a single context line injected into the system prompt.

**Improvement:** Calibrate vocabulary, detail level, and format based on role:

| Role | Answer style |
|---|---|
| Executive / VP | 2-sentence summary, no policy jargon, decision-oriented |
| Manager | Structured with conditions, approval flows highlighted |
| Employee | Plain language, step-by-step, "what do I need to do" framing |
| HR / Admin | Full policy detail, references, edge cases included |

The system prompt varies by detected role — the same underlying evidence produces differently formatted answers for different users.

---

## 3. Per Document Type Improvements

Each document type has unique structure challenges that generic extractors handle poorly. Targeted improvements per type will significantly increase extraction quality.

---

### 3.1 PDF Documents

**Current limitations:**
- Multi-column layouts extracted as interleaved text from different columns
- Footnotes and endnotes appended at the end, losing their in-line context
- Headers and footers (page numbers, document title) pollute section text
- Scanned PDFs with no text layer treated as blank
- Tables extracted as raw text without row/column structure

**Improvements:**

| Issue | Solution |
|---|---|
| Multi-column layout | Use layout-aware PDF parser (pdfplumber) that detects column boundaries from x-coordinates |
| Footnotes | Detect superscript numbers; link footnote text back to the inline reference point |
| Headers/footers | Detect repeated text across pages (same text at y < 50 or y > 750); strip from content |
| Scanned PDFs | Auto-detect text layer absence; route to OCR (AWS Textract) automatically |
| Table extraction | Use pdfplumber's table extraction API; preserve cell structure as HTML `<table>` |
| Page number continuity | Track section continuation across page breaks using heading-matching heuristics |

**PDF extraction pipeline:**
```
PDF file
  ↓
Detect: has text layer? → No → AWS Textract OCR
                        → Yes ↓
Detect: multi-column? → Yes → pdfplumber column-aware extraction
                      → No  → standard text extraction
  ↓
Strip: headers, footers, page numbers (repeated text pattern detection)
  ↓
Extract tables → preserve as HTML tables (not pipe text)
  ↓
Reconstruct footnote links
  ↓
Section boundary detection (headings, bold text, font-size changes)
```

---

### 3.2 DOCX (Word Documents)

**Current limitations:**
- Tracked changes (accept/reject markup) leaks old/deleted text into extraction
- Comments and annotations indexed alongside main content
- Styled headings (Heading 1/2/3 styles) sometimes missed if heading is bold paragraph, not Heading style
- Tables with merged cells lose structure

**Improvements:**

| Issue | Solution |
|---|---|
| Tracked changes | Accept all changes before extraction using python-docx `document.revisions` processing |
| Comments / annotations | Strip `w:comment` elements from XML before parsing |
| Heading detection | Check both `paragraph.style.name` AND `paragraph.runs[].bold + font_size` heuristics |
| Merged cell tables | Use python-docx `cell.merge_count`; expand merged cells to proper HTML `colspan`/`rowspan` |
| Text boxes | Extract `txbxContent` from XML — often missed by standard parsers |
| Embedded images | Extract with `docx.part.related_parts`; run OCR on each |
| Document properties | Extract `core_properties`: author, created, modified, company, subject |

---

### 3.3 Excel / XLSX Spreadsheets

**Current limitations:**
- Multiple sheets treated as a single flat document
- Formulas extracted as formula text (`=SUM(B2:B10)`) instead of values
- Named ranges not extracted
- Charts and pivot tables not captured
- Sparse sheets (approval matrices, rate tables) lose their 2D meaning when linearized

**Improvements:**

| Issue | Solution |
|---|---|
| Multiple sheets | Each sheet becomes a separate section with sheet name as title |
| Formulas | Use `data_only=True` in openpyxl to extract computed values, not formulas |
| Named ranges | Extract `workbook.defined_names`; annotate sections with range names |
| Charts | Detect chart objects; generate text description of chart type, axes, series titles |
| Sparse matrices | Detect tables with row/column headers; convert to "Row header + Column header = Value" text |
| Approval matrices | "Grade M3 + Category Travel: Approval limit ₹50,000" — one fact per cell |
| Conditional formatting | Extract rules as policy facts: "Cells highlighted red = threshold exceeded" |

**Excel extraction produces structured facts, not raw text:**

```
Sheet: "Approval Matrix"
  CEO: All categories — Unlimited
  SVP: Travel — ₹5,00,000 | Procurement — ₹10,00,000
  VP:  Travel — ₹2,00,000 | Procurement — ₹5,00,000
  Manager: Travel — ₹50,000 | Procurement — ₹1,00,000
```

This format is directly answerable: "What can a VP approve for procurement? ₹5,00,000."

---

### 3.4 JPEG / PNG / Image Files

**Current limitations:**
- OCR treats all images as plain text — diagrams, charts, org charts are not understood
- Low-confidence OCR text included without qualification
- No understanding of spatial relationships (flow arrows, hierarchy, port tables)
- Image metadata (EXIF, filename date) not extracted

**Improvements:**

| Issue | Solution |
|---|---|
| Diagram understanding | Use multimodal LLM (Claude Vision / GPT-4V) to describe diagrams: "This is a deployment diagram showing Service A connecting to Service B on port 443" |
| Chart understanding | Multimodal LLM extracts: chart type, axis labels, data series names, trend |
| Org charts | Multimodal LLM extracts hierarchy: "CEO → CFO, CTO, CHRO → ..." |
| Port/network tables | Multimodal LLM extracts: "Application X uses ports 80, 443, 8080" |
| Low-confidence OCR | Filter Textract LINE blocks with confidence < `ocr_min_confidence` (80); mark uncertain text |
| Handwritten content | Route to Textract handwriting model; flag as "handwritten — lower confidence" |
| Image metadata | Extract EXIF date, author; use as document metadata |

**Image processing pipeline:**
```
Image file
  ↓
Detect content type: photo | diagram | chart | screenshot | handwritten
  ↓
Route:
  Diagram / chart / org chart → Multimodal LLM description
  Dense text / screenshot    → AWS Textract OCR
  Handwritten               → Textract handwriting model
  ↓
Merge structured description + OCR text
  ↓
Section: title from filename + LLM-generated description + OCR text
```

---

### 3.5 HTML Documents

**Current limitations:**
- JavaScript-rendered content (SPAs) not captured — extractor sees empty `<div>` tags
- Navigation bars, breadcrumbs, sidebars indexed as content
- Cookie banners, pop-up overlays extracted
- Inline CSS and script tags pollute text

**Improvements:**

| Issue | Solution |
|---|---|
| JS-rendered content | Use Playwright headless browser to render before extraction |
| Navigation / sidebars | Detect and strip: `<nav>`, `<aside>`, `role="navigation"`, common class names (`sidebar`, `breadcrumb`) |
| Non-content elements | Strip: `<script>`, `<style>`, `<noscript>`, `<iframe>`, cookie banners |
| Semantic HTML use | Prefer `<article>`, `<section>`, `<main>` as primary content containers |
| `<table>` extraction | Preserve as HTML table; detect if data table vs. layout table (layout tables have no `<th>`) |
| Link extraction | Extract `<a href>` links as related document references |

---

### 3.6 PowerPoint / PPTX

**Current limitations:**
- Speaker notes not captured — often contain important context the slide text omits
- Animations mean some text appears/disappears — static extraction misses sequenced reveals
- SmartArt (org charts, process flows) not extracted

**Improvements:**

| Issue | Solution |
|---|---|
| Speaker notes | Extract `slide.notes_slide.notes_text_frame.text`; append as "Context notes:" after slide content |
| SmartArt | Extract text from all SmartArt nodes; infer structure from shape positions |
| Sequenced content | Extract all animation shapes regardless of trigger order |
| Slide master text | Extract placeholder text from slide master for context |
| Slide numbers + titles | Use slide title as section heading; slide number as order_index |

---

### 3.7 Explainability Text Generation

**What this is:** After ingestion, generate a human-readable "explainability brief" for each document — a structured summary that explains what the document covers, who it applies to, what the key rules are, and what questions it can answer.

**Format:**
```
Document: HR Leave Policy v2.1
Type: HR Policy  |  Effective: 1 Jan 2024  |  Audience: All employees

WHAT IT COVERS
  Annual leave, sick leave, maternity/paternity leave, bereavement leave,
  earned leave encashment, and carry-forward rules.

WHO IT APPLIES TO
  All permanent employees (full-time and part-time).
  Excludes: contract workers, interns, probationary employees (first 6 months).

KEY RULES
  • Annual leave: 18 days per year, prorated for partial year joiners
  • Sick leave: 12 days per year, non-carry-forward
  • Maternity: 26 weeks paid (as per Maternity Benefit Act)
  • Carry-forward: Maximum 30 days; excess lapses on 31 Dec

QUESTIONS THIS DOCUMENT CAN ANSWER
  "How many days annual leave do I get?"
  "Can I carry forward unused leave?"
  "What is the maternity leave policy?"
  "Who approves leave applications?"
```

This brief is:
- Shown in the document viewer alongside the source
- Used as additional retrieval signal (a user asking "who can approve leave" matches the explainability brief even if the exact section uses different phrasing)
- Exported as a searchable admin report

---

## 4. Retrieval Quality Improvements

### 4.1 Cross-Encoder Re-ranking

**Current:** Bi-encoder embeddings (fast, parallel, but coarse).

**Improvement:** After initial retrieval, run a **cross-encoder** that sees both the query AND the candidate section text together. Cross-encoders are far more accurate but slower — used only on the top 12 candidates.

```
Query: "What is the notice period for resignation?"
Section A: "...employees wishing to resign must give notice..."  → cross-encoder score: 0.94
Section B: "...notice period for termination by employer..."   → cross-encoder score: 0.61
```

Cross-encoders outperform bi-encoders on precision by 15–25% on domain-specific corpora.

---

### 4.2 Late Chunking

**Current:** Documents are chunked at ingest time into fixed sections. The section boundary determines what gets retrieved.

**Problem:** Important context often spans section boundaries. A threshold value defined in Section 3 is referenced in Section 7 — retrieving only Section 7 misses the threshold.

**Improvement — Late chunking:**
1. At ingest, embed the full document context as a single vector.
2. At retrieval, dynamically extract the most relevant window (not a pre-determined chunk) from the document.
3. Window size adapts to question complexity — narrow for exact lookups, broad for explanatory questions.

---

### 4.3 Multi-Hop Retrieval

**Current:** Single retrieval pass per query.

**Problem:** Some questions require information from two places: "Does the travel policy cover expenses for employees on notice period?" — answer requires joining travel policy + HR exit policy.

**Improvement:**
1. First retrieval pass → initial evidence
2. LLM identifies what is still missing: "need to know notice period definition"
3. Second targeted retrieval pass for the gap
4. Combined evidence → final answer

This enables genuine cross-document reasoning, not just cross-document search.

---

### 4.4 Query Intent Expansion via LLM

**Current:** Intent detection is pattern-based (regex/substring matching).

**Improvement:** Use a small, fast LLM call to classify intent more accurately:

```
Input: "Can my manager force me to take leave?"
Output: {
  "intent": "eligibility_rights",
  "topic": "annual_leave",
  "entities": ["manager", "forced_leave"],
  "implied_question": "Are employees obligated to take leave when directed by management?"
}
```

The implied question often retrieves better than the literal phrasing.

---

### 4.5 Negative Example Retrieval

**Current:** System retrieves supporting evidence only.

**Improvement:** Also retrieve sections that explicitly say the answer does NOT apply, and inject them as negative context:

```
[RELEVANT EVIDENCE]   "Employees with > 2 years service qualify for..."
[NEGATIVE EVIDENCE]   "Contract employees are NOT eligible for this scheme."
```

Forces the LLM to include exclusions and conditions in the answer — critical for policy documents where knowing who is excluded is as important as knowing who qualifies.

---

### 4.6 Semantic Chunking

**Current:** Sections determined by document headings/structure.

**Problem:** Some documents have no headings. Long sections contain multiple unrelated topics.

**Improvement:** Use embedding similarity between adjacent sentences to detect topic shifts — split where similarity drops below a threshold. This creates semantically coherent chunks regardless of document structure.

---

## 5. Answer Generation Improvements

### 5.1 Inline Citation System

**Current:** Sources shown separately below the answer.

**Improvement:** Inline citations like academic papers or Claude's citation feature:

```
Employees are entitled to 18 days of annual leave per year [HR Leave Policy §3.1].
For employees joining mid-year, leave is prorated from the joining month [HR Leave Policy §3.2].
Contract employees are not eligible [HR Leave Policy §1.4].
```

Each cited claim is a clickable link opening the source document at the exact section. This:
- Makes answers verifiable at a glance
- Builds user trust (they can see the evidence)
- Highlights when the LLM is synthesizing vs. quoting

---

### 5.2 Structured Output Modes

**Current:** Always returns markdown prose.

**Improvement:** Detect output format from question intent:

| Question type | Output format |
|---|---|
| "List all policies that cover X" | Numbered list with document links |
| "Compare policy A and B" | Side-by-side comparison table |
| "What is the approval matrix?" | HTML table |
| "Show me the process for X" | Numbered steps with sub-steps |
| "What are the key dates?" | Timeline / chronological list |
| "Who qualifies for X?" | Eligibility matrix (role vs. condition) |

---

### 5.3 Streaming Responses

**Current:** Full answer generated, then returned as one HTTP response.

**Improvement:** Stream tokens to the UI as they are generated (Server-Sent Events). For a 400-token answer:
- **Current:** User waits 3–5 seconds, then sees the full answer
- **Streaming:** First words appear in < 500ms, answer builds progressively

This is the single most impactful UX improvement for perceived performance. ChatGPT and Claude both stream by default — users now expect it.

---

### 5.4 Answer Versioning & Conflict Detection

**When two documents say different things** (e.g., an old policy and a new one both indexed), the current system picks one and answers from it.

**Improvement:**
- Detect when top-N sections are from different document versions
- Surface the conflict explicitly: "Note: The 2023 policy says X, the 2024 policy says Y. The 2024 version supersedes."
- Compare `effective_date` metadata to determine precedence

---

### 5.5 Answer Re-generation on Low Confidence

**Current:** Low confidence → adds a disclaimer and follow-up question.

**Improvement:** When confidence is LOW and a clarifying follow-up is answered, automatically re-retrieve and re-generate the original answer with the new context — without the user having to re-ask the original question.

```
User: "What is the leave policy?"       ← ambiguous
Bot:  "Which type of leave? (sick / annual / maternity)"
User: "Annual"
Bot:  [auto re-generates answer for "annual leave policy"]   ← no re-ask needed
```

---

### 5.6 Tone & Language Calibration

**Current:** Single answer tone for all users.

**Improvement:** Detect user language and calibrate:
- If user writes in Hindi/Hinglish → respond in the same register
- Formal question → formal answer; casual question → conversational answer
- Technical jargon in question → technical answer; plain question → plain answer

---

## 6. Explainability & Trust

### 6.1 Answer Provenance Panel

Every answer shows a collapsible **"How I answered this"** panel:

```
┌─ How I answered this ──────────────────────────────────────────────┐
│ Sources used:                                                       │
│   1. HR Leave Policy §3.1 (confidence: 94%)  [open]               │
│   2. HR Leave Policy §3.2 (confidence: 81%)  [open]               │
│                                                                     │
│ Search terms used: annual leave, entitlement, days, employee       │
│ Your role context: Employee | All Departments                      │
│ Retrieval method: Hybrid search (BM25 + semantic)                  │
│                                                                     │
│ Was this answer helpful?  👍  👎  [Report issue]                   │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 Confidence Explanation

**Current:** Confidence level shown (High / Medium / Low) with no explanation.

**Improvement:** Explain WHY confidence is low:

```
⚠ Medium confidence — the question matches 2 documents with different
  rules. Showing the most recent version. Open both sources to compare.
```

```
⚠ Low confidence — this topic may not be covered in the indexed documents.
  The answer is based on partial evidence. Consider asking HR directly.
```

### 6.3 "What I don't know" Transparency

When the system cannot answer, instead of a generic "not found":

```
I could not find a clear answer in the indexed documents.
I searched across 47 documents for: "overtime policy", "extra hours", "additional pay"
The closest I found was [HR Policy §8 — Working Hours] which covers standard hours
but does not explicitly address overtime compensation.
Suggested action: Contact HR (hr@company.com) or raise a query in ServiceNow.
```

### 6.4 Source Quality Indicators

Annotate sources with metadata that helps users assess reliability:

```
[HR Leave Policy v2.1]  ✓ Current version  |  Effective: Jan 2024  |  Owner: HR
[HR Leave Policy v1.8]  ⚠ Superseded       |  Effective: Jan 2022  |  Archived
```

### 6.5 Feedback Loop Integration

Every answer has a 👍 / 👎 button. Negative feedback triggers:
1. Log the question, retrieved sections, and answer to a review queue
2. HR/admin reviews and either: (a) corrects the source document, or (b) adds a supplementary fact
3. The correction is available at next ingest

This creates a flywheel: the system gets better from real usage.

---

## 7. Multi-Modal Capabilities

### 7.1 Vision-Based Document Understanding

Use **Claude Vision / GPT-4V** as a pre-processor for visually complex documents:

| Content type | What the model extracts |
|---|---|
| Deployment diagram | Services, connections, protocols, ports |
| Architecture diagram | Components, data flows, dependencies |
| Org chart | Reporting structure, roles, levels |
| Process flowchart | Steps, decision points, conditions |
| Data flow diagram | Systems, data entities, transformations |
| Infographic | Key facts, statistics, relationships |

The extracted structured text is then indexed alongside or instead of OCR output.

---

### 7.2 Chart & Graph Understanding

**Current:** Charts in PDFs / images are ignored or produce garbled OCR.

**Improvement:**

```
Detected: Bar chart
Title: "Monthly Sales Contest Results — Q3 2024"
X-axis: Months (July, August, September)
Y-axis: FYFP achieved (₹ lakhs)
Series 1 (Target): 3.49, 3.49, 3.49
Series 2 (Actual): 2.8, 3.6, 4.1

Extracted fact: In August and September, actuals exceeded target.
                September actual (4.1L) was 17.5% above target.
```

This makes chart data searchable and answerable.

---

### 7.3 Table Structure Preservation

**Current:** Tables converted to pipe-separated text, losing 2D structure meaning.

**Improvement:** Preserve tables as HTML in the index. At retrieval time, render tables as HTML in the answer:

```html
<table>
  <tr><th>Grade</th><th>Travel</th><th>Procurement</th></tr>
  <tr><td>VP</td><td>₹2,00,000</td><td>₹5,00,000</td></tr>
  <tr><td>Manager</td><td>₹50,000</td><td>₹1,00,000</td></tr>
</table>
```

Users can read approval matrices, rate cards, and comparison tables natively — exactly as they appear in the source document.

---

### 7.4 Document Thumbnail Previews

Generate thumbnail images of source document pages at ingest time. When a source is cited, show a thumbnail of the exact page — users can see the physical document layout before clicking through.

---

## 8. Agentic & Reasoning Capabilities

### 8.1 Multi-Step Reasoning (Chain of Thought)

**Current:** Single LLM call with evidence → answer.

**Problem:** Complex policy questions require reasoning chains:
*"I joined in July 2023 with a CTC of ₹12L. I want to resign. What is my notice period and do I get leave encashment?"*

This requires: joining date calculation → service length → notice period lookup → leave encashment eligibility.

**Improvement — ReAct (Reasoning + Acting) pattern:**

```
Think: Question involves notice period and leave encashment.
       Need to know: service length, notice period rules, encashment rules.
Act:   Retrieve "notice period policy"
Observe: Notice period = 30 days for < 2 years, 60 days for ≥ 2 years.
Think: User joined July 2023, current date ~April 2025. Service = ~22 months < 2 years.
       Therefore notice period = 30 days.
Act:   Retrieve "leave encashment on resignation"
Observe: Encashment paid on earned leave balance for employees with > 1 year service.
Think: User has 22 months service → eligible. Earned leave balance needed.
Answer: Notice period is 30 days (< 2 years service). You are eligible for leave
        encashment on your earned leave balance as of resignation date.
```

Each Act step is a real retrieval call — the model only answers from evidence, not training data.

---

### 8.2 Tool Use / Function Calling

Expose structured tools the LLM can call during reasoning:

| Tool | Input | Output |
|---|---|---|
| `search_policy(query)` | Search string | Top 5 sections |
| `lookup_employee_data(field)` | Field name | Value from HR system |
| `calculate_leave_balance(user_id, date)` | User + date | Days remaining |
| `get_policy_version(doc_name)` | Document title | Latest version + effective date |
| `open_service_request(type, description)` | Request details | ServiceNow ticket ID |

With tool use, the bot can answer: *"How many annual leave days do I have left?"* by calling `calculate_leave_balance(user_id)` rather than asking the user.

---

### 8.3 Proactive Information Surfacing

**Current:** Purely reactive — user asks, bot answers.

**Improvement:** During a conversation, detect when related information should be proactively surfaced:

```
User: "What is the travel reimbursement process?"
Bot:  [answers travel reimbursement]
      "You may also want to know: the travel policy was updated in March 2024.
       Key change: International travel now requires VP approval (previously AVP)."
```

The bot detects that a policy version change is relevant and surfaces it unprompted — like a knowledgeable colleague who knows what matters.

---

### 8.4 Clarification Conversation Flow

**Current:** One clarifying question asked, then user must re-state full question.

**Improvement:** Full clarification dialog:

```
User: "What is the policy?"
Bot:  "Which policy are you asking about? I have 47 policies indexed.
       Popular topics: Leave, Travel, IT Security, Expenses, Code of Conduct"
User: "Leave"
Bot:  "Which type of leave? Annual | Sick | Maternity/Paternity | Casual | Other"
User: "Annual"
Bot:  [answers annual leave policy — no need to re-ask]
```

Each clarification step narrows the retrieval scope. Final answer generated automatically.

---

## 9. Enterprise Platform Features

### 9.1 Authentication & Single Sign-On

**Current:** User ID passed as cookie/query param. No real authentication.

**Improvement:**
- SAML 2.0 / OAuth 2.0 / OIDC integration with corporate IdP (Azure AD, Okta)
- JWT tokens carry: user_id, role, grade, department, location
- `resolve_user_profile()` decodes JWT claims — no code change elsewhere needed
- Session management with token refresh

---

### 9.2 Role-Based Access Control (RBAC)

**Current:** Per-user document access list (flat `user_id_access` array).

**Improvement — Role-based ACL:**

| Role | Accessible document groups |
|---|---|
| All employees | HR policies, Code of Conduct, IT policies |
| Managers | + Performance management, Compensation bands |
| HR Business Partners | + HR procedures, Salary structures, PIP guidelines |
| Finance team | + Finance procedures, Approval matrices |
| Executives | All documents |

Document groups managed in admin UI. Role assignment synced from HR system. OpenSearch query filter uses `role_access` field instead of `user_id_access`.

---

### 9.3 Document Access Management UI

Admin interface for:
- Upload / delete / re-index documents
- Set document access groups
- View document ingestion status and health
- Mark documents as archived / superseded
- Set effective date and expiry date on documents (auto-archive after expiry)

---

### 9.4 Audit Trail

Every query logged with:
- User ID, role, department, timestamp
- Question asked
- Documents retrieved + scores
- Answer generated
- Sources cited
- User feedback (👍/👎)
- Confidence level

Stored in OpenSearch audit index. Queryable by:
- HR: "What are employees most commonly asking about leave?"
- Security: "Who accessed the compensation band documents this month?"
- Compliance: "Were any policy questions answered incorrectly (low confidence + 👎)?"

---

### 9.5 Analytics Dashboard

**Admin dashboard showing:**

| Metric | Value |
|---|---|
| Total questions this week | 1,247 |
| Questions answered with high confidence | 73% |
| Unanswered questions (no evidence found) | 8% |
| FAQ fast-path hit rate | 31% |
| Most asked topics | Leave (34%), Travel (22%), IT (18%) |
| Lowest confidence topics | Overtime (42% low), Transfer policy (38% low) |
| Documents with no traffic | 12 documents (possible gaps) |

**Low confidence topics** highlight gaps in the document corpus — HR knows exactly which policies need to be written or updated.

---

### 9.6 Multi-Tenancy

Support multiple independent departments / business units on one deployment:

- Each tenant has isolated document index (separate OpenSearch index prefix)
- Tenant-specific domain profile, UI branding, and prompt configuration
- Cross-tenant queries disabled by default
- Usage and cost tracked per tenant

---

### 9.7 API-First Architecture

Expose all capabilities as documented REST + WebSocket APIs:

```
POST   /api/v1/chat                    Chat with document corpus
GET    /api/v1/threads/{id}            Get conversation thread
POST   /api/v1/documents              Upload document
DELETE /api/v1/documents/{id}          Remove document
GET    /api/v1/search                  Direct search
GET    /api/v1/admin/analytics         Usage analytics
GET    /api/v1/admin/audit             Audit log
POST   /api/v1/admin/feedback          Submit answer feedback
WebSocket /ws/v1/chat                  Streaming chat
```

Enables embedding PolicyGPT into ServiceNow, Microsoft Teams, Slack, or any internal portal.

---

### 9.8 Microsoft Teams / Slack Integration

Deploy PolicyGPT as a bot in Teams or Slack:

```
@PolicyGPT How many sick leaves can I take?
```

Bot responds in the channel thread with answer + source links. No new UI to learn — employees use it where they already work.

---

## 10. Evaluation Framework

Without measurement, improvement is guesswork. An evaluation framework makes quality visible and tracks regression.

### 10.1 RAGAS — Automated RAG Evaluation

RAGAS (Retrieval Augmented Generation Assessment) measures four dimensions:

| Metric | What it measures | Target |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | > 0.90 |
| **Answer Relevancy** | Does the answer address the question asked? | > 0.85 |
| **Context Precision** | What fraction of retrieved context was actually useful? | > 0.75 |
| **Context Recall** | Was all the relevant evidence retrieved? | > 0.80 |

Run RAGAS on a golden test set of 200 question-answer pairs after every deployment.

---

### 10.2 Golden Test Set

Maintain a curated set of 200+ questions with verified correct answers, covering:
- Simple factual lookups ("What is the annual leave entitlement?")
- Multi-condition eligibility ("Am I eligible for maternity leave if I joined 6 months ago?")
- Cross-document questions ("Does the travel policy override the expense policy for meals?")
- Unanswerable questions ("What is the overtime rate?" → should say "not found")
- Adversarial questions (policy jargon, abbreviations, Hindi transliterations)

Scores tracked per deployment. Regression in any metric triggers review before release.

---

### 10.3 Human Feedback Loop

**Weekly review process:**
1. Export all 👎 rated answers from the previous week
2. HR/subject-matter experts review each one
3. Categorize failure: (a) wrong retrieval, (b) wrong answer, (c) incomplete, (d) outdated doc
4. Fix: update document, add supplementary fact, adjust config, or file improvement ticket

Over time, 👎 rate should trend toward < 5%.

---

### 10.4 A/B Testing Framework

Test prompt changes, retrieval configurations, or new features on a subset of traffic:

```
Group A (50%): current retrieval config
Group B (50%): new HyDE embedding approach

Measure after 500 queries:
  Confidence score distribution
  User 👍/👎 rate
  FAQ fast-path hit rate
```

Statistical significance required before promoting any change to 100% traffic.

---

### 10.5 LLM Judge

Use a strong LLM (Claude Opus or GPT-4) as an automated judge to evaluate answer quality on a daily sample:

```
Prompt to LLM judge:
  "Question: [question]
   Evidence: [retrieved sections]
   Answer given: [answer]

   Rate 1–5 on: accuracy, completeness, clarity, groundedness.
   Flag if: hallucination detected, evidence ignored, key condition missed."
```

Automated quality score tracked over time. Drops trigger investigation.

---

## 11. Operational Excellence

### 11.1 Monitoring & Alerting

**Key metrics to monitor:**

| Metric | Alert threshold |
|---|---|
| API p95 latency | > 8 seconds |
| Answer confidence (% HIGH) | < 60% over 1 hour |
| OpenSearch availability | < 99.9% |
| Ingestion failure rate | > 5% of files |
| LLM API error rate | > 2% |
| Cost per query | > 2× baseline |

Alerts via PagerDuty / email to on-call team.

---

### 11.2 Observability Stack

| Layer | Tool |
|---|---|
| Application traces | AWS X-Ray / OpenTelemetry |
| Metrics | CloudWatch / Prometheus |
| Log aggregation | CloudWatch Logs / ELK |
| Retrieval debug logs | Already implemented (local files) → ship to S3 |
| Cost tracking | Already implemented → add to CloudWatch dashboard |

---

### 11.3 Disaster Recovery

| Component | RPO | RTO | Strategy |
|---|---|---|---|
| OpenSearch index | 1 hour | 4 hours | Automated snapshots to S3 every hour |
| In-memory corpus | 0 | 15 min | Re-ingest from document folder on restart |
| Conversation threads | 1 hour | 1 hour | OS-backed persistence with S3 snapshot |
| Configuration | 0 | 5 min | Version-controlled in Git |

---

### 11.4 Horizontal Scaling

**Current:** Single-process FastAPI server. Bot instance is in-memory.

**Improvement for high load:**
- Separate the API server from the ingestion worker (different containers)
- API servers can scale horizontally (stateless — corpus loaded from shared OpenSearch)
- Ingestion worker runs as a single job (avoid duplicate enrichment)
- Load balancer in front of API servers (ALB / nginx)
- Redis for shared answer cache across API server instances

```
              ┌──────────────────────────────────────┐
              │          Load Balancer               │
              └──────┬───────────┬───────────────────┘
                     │           │
              ┌──────▼─┐   ┌─────▼──┐
              │ API #1 │   │ API #2 │   ...  (stateless, scale horizontally)
              └──────┬─┘   └─────┬──┘
                     │           │
              ┌──────▼───────────▼──────┐
              │   OpenSearch Cluster    │   (persistent, HA)
              └─────────────────────────┘
```

---

### 11.5 Rate Limiting & Cost Control

**Per-user rate limits:**
- 20 questions per minute
- 200 questions per day
- Burst allowance for first 5 questions (instant response)

**Cost guardrails:**
- Monthly spend cap per tenant (alert at 80%, hard stop at 100%)
- Automatic downgrade to cheaper model when daily budget is within 20% of limit
- FAQ fast-path aggressively pre-warmed to reduce LLM calls

---

## 12. Prioritised Roadmap

### Phase 1 — Foundation (0–3 months)

| Item | Impact | Effort |
|---|---|---|
| Streaming responses | High UX | Low |
| Inline citation system | High trust | Medium |
| Query rewriting for follow-ups | High recall | Low |
| PDF: header/footer stripping + table extraction | High quality | Medium |
| Excel: multi-sheet + value extraction | High quality | Medium |
| Image: multimodal LLM for diagrams | High coverage | Medium |
| Conversation memory (3-tier) | High quality | Medium |
| Answer confidence explanation | High trust | Low |
| Feedback loop (👍/👎 + review queue) | High quality | Low |

### Phase 2 — Depth (3–6 months)

| Item | Impact | Effort |
|---|---|---|
| SAML/SSO login + JWT profile | Enterprise required | High |
| Role-based document access control | Enterprise required | Medium |
| HyDE embedding | High recall | Low |
| Cross-encoder re-ranking | High precision | Medium |
| Multi-hop retrieval | High reasoning | High |
| Conflict detection (version clash) | High trust | Medium |
| Audit trail | Compliance required | Medium |
| Analytics dashboard | Operations | Medium |
| Golden test set + RAGAS evaluation | Quality measurement | Medium |
| Explainability brief per document | High trust | Medium |

### Phase 3 — Platform (6–12 months)

| Item | Impact | Effort |
|---|---|---|
| Multi-tenancy | Enterprise scale | High |
| Teams / Slack integration | Adoption | Medium |
| ReAct agentic reasoning | Complex Q&A | High |
| Tool use (HR system, leave balance) | High value | High |
| A/B testing framework | Continuous improvement | Medium |
| Horizontal scaling + HA | Enterprise SLA | High |
| Document admin UI | Operations | Medium |
| Proactive information surfacing | Delight | High |
| Tone / language calibration | Personalization | Medium |

---

*Document reflects PolicyGPT improvement analysis — April 2026.*
*Next review: after Phase 1 completion.*
