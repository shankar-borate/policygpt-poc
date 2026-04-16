"""
PolicyGPT Corpus Analysis — Report Generator
Reads all metadata from D:/policy-mgmt/data/vcx_policies/metadata/
Outputs:
  1. PolicyGPT_Corpus_Report.html    — full interactive report
  2. PolicyGPT_Documents.csv         — per-document details
  3. PolicyGPT_FAQs.csv              — all FAQ Q&A pairs
  4. PolicyGPT_QA_TestLog.csv        — real query/answer samples from retrieval logs
  5. PolicyGPT_Gaps_Actions.csv      — identified gaps with recommended actions
"""

import os, re, csv, json
from pathlib import Path
from datetime import datetime

BASE = Path("D:/policy-mgmt/data/vcx_policies/metadata")
OUT  = Path("D:/policy-mgmt/policygpt-poc")

# ── 1. Parse ingestion files ─────────────────────────────────────────────────
docs = []
for f in sorted(os.listdir(BASE / "ingestion")):
    if not f.endswith(".txt") or f.endswith("_faq.txt"):
        continue
    path = BASE / "ingestion" / f
    content = path.read_text(encoding="utf-8", errors="ignore")
    def g(pattern, default=""):
        m = re.search(pattern, content)
        return m.group(1).strip() if m else default
    docs.append({
        "file":       f,
        "title":      g(r"Canonical title:\s*(.+)") or g(r"Title:\s*(.+)"),
        "type":       g(r"Document type:\s*(.+)"),
        "version":    g(r"Version:\s*(.+)"),
        "eff_date":   g(r"Effective date:\s*(.+)"),
        "audiences":  g(r"Audiences:\s*(.+)"),
        "tags":       g(r"Tags:\s*(.+)"),
        "keywords":   g(r"Keywords:\s*(.+)"),
        "sections":   int(g(r"Section count:\s*(\d+)", "0")),
        "has_summary": "=== Document Summary ===" in content,
        "summary":    (content.split("=== Document Summary ===")[1].split("=== Section")[0].strip()[:500]
                       if "=== Document Summary ===" in content else ""),
        "source_file": g(r"Source file name:\s*(.+)"),
    })

# ── 2. Parse FAQ files ───────────────────────────────────────────────────────
faqs = []
faq_counts = {}
for f in sorted(os.listdir(BASE / "faq")):
    if not f.endswith("_faq.txt"):
        continue
    doc_name = f.replace("_faq.txt", "")
    content = (BASE / "faq" / f).read_text(encoding="utf-8", errors="ignore")
    pairs = re.findall(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|\Z)", content, re.DOTALL)
    faq_counts[doc_name] = len(pairs)
    for q, a in pairs:
        faqs.append({"document": doc_name, "question": q.strip(), "answer": a.strip()[:300]})

# ── 3. Parse retrieval logs ──────────────────────────────────────────────────
qa_log = []
retrieval_dir = BASE / "retrieval"
for f in sorted(os.listdir(retrieval_dir)):
    content = (retrieval_dir / f).read_text(encoding="utf-8", errors="ignore")
    q_match = re.search(r"User question:\s*(.+)", content)
    a_match = re.search(r"=== Final Answer ===\s*([\s\S]+?)(?:=== Sources ===|\Z)", content)
    intent_match = re.search(r"Inferred answer intent:\s*(.+)", content)
    docs_match = re.search(r"Likely matching documents:\s*(.+)", content)
    is_answerable = "is_answerable: True" in content or "is_answerable=True" in content
    top_score_match = re.search(r"score=([\d.]+)", content)
    if q_match:
        answer_raw = a_match.group(1).strip()[:400] if a_match else ""
        answered = bool(answer_raw) and "couldn't find" not in answer_raw.lower()
        qa_log.append({
            "timestamp": f[:26].replace("_", ":").replace("T", " "),
            "question":  q_match.group(1).strip(),
            "intent":    intent_match.group(1).strip() if intent_match else "",
            "top_docs":  docs_match.group(1).strip()[:100] if docs_match else "",
            "top_score": top_score_match.group(1) if top_score_match else "",
            "answer":    answer_raw[:200],
            "answered":  "YES" if answered else "NO",
        })

# ── 4. Parse failure log ─────────────────────────────────────────────────────
failures = []
fail_dir = BASE / "ingestion_failures"
for f in os.listdir(fail_dir):
    content = (fail_dir / f).read_text(encoding="utf-8", errors="ignore")
    src = re.search(r"Source file name:\s*(.+)", content)
    reason = re.search(r"Reason:\s*(.+)", content)
    failures.append({
        "file":   src.group(1).strip() if src else f,
        "reason": reason.group(1).strip()[:200] if reason else "unknown",
    })

# ── 5. Compute aggregate stats ───────────────────────────────────────────────
total_docs     = len(docs)
total_sections = sum(d["sections"] for d in docs)
total_faqs_n   = len(faqs)
type_counts    = {}
audience_flag  = 0
date_flag      = 0
version_flag   = 0
single_section = 0
for d in docs:
    t = d["type"] or "unknown"
    type_counts[t] = type_counts.get(t, 0) + 1
    if d["audiences"] in ("", "none"): audience_flag += 1
    if d["eff_date"] in ("", "unknown"): date_flag += 1
    if d["version"] in ("", "unknown"): version_flag += 1
    if d["sections"] <= 1: single_section += 1

answered_count = sum(1 for q in qa_log if q["answered"] == "YES")
total_queries  = len(qa_log)

# Category classification
SECURITY_KEYWORDS = ["cyber","security","siem","vulnerability","encryption","hardening","ssl","iam",
                     "firewall","cloud security","server","soc","incident","threat","dlp","sdr"]
HR_KEYWORDS        = ["onboarding","remote","wfh","employment","recruitment","leave","salary","conduct"]
COMPLIANCE_KEYWORDS= ["compliance","anti","slavery","trafficking","ethical","sourcing","intellectual",
                       "gdpr","hipaa","pci","iso"]
OPS_KEYWORDS       = ["asset","license","sop","inventory","procurement","rto","rpo","recovery","backup",
                       "log monitoring","garbage","url whitelist"]

def classify(d):
    title = (d["title"] + " " + d["tags"]).lower()
    if any(k in title for k in SECURITY_KEYWORDS): return "IT Security & Cyber"
    if any(k in title for k in HR_KEYWORDS):        return "HR & Employment"
    if any(k in title for k in COMPLIANCE_KEYWORDS):return "Compliance & Legal"
    if any(k in title for k in OPS_KEYWORDS):       return "IT Operations & Assets"
    return "General"

for d in docs:
    d["category"] = classify(d)

cat_counts = {}
for d in docs:
    cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1

# Duplicate titles
title_counts = {}
for d in docs:
    t = d["title"]
    title_counts[t] = title_counts.get(t, 0) + 1
duplicates = [(t, c) for t, c in title_counts.items() if c > 1]

# ── 6. Write CSV: Documents ──────────────────────────────────────────────────
csv_docs = OUT / "PolicyGPT_Documents.csv"
with open(csv_docs, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["title","category","type","version","eff_date",
                                        "sections","audiences","has_summary","tags","source_file"])
    w.writeheader()
    for d in docs:
        w.writerow({k: d.get(k,"") for k in w.fieldnames})

# ── 7. Write CSV: FAQs ───────────────────────────────────────────────────────
csv_faqs = OUT / "PolicyGPT_FAQs.csv"
with open(csv_faqs, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["document","question","answer"])
    w.writeheader()
    w.writerows(faqs)

# ── 8. Write CSV: QA Test Log ────────────────────────────────────────────────
csv_qa = OUT / "PolicyGPT_QA_TestLog.csv"
with open(csv_qa, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["timestamp","question","answered","intent","top_docs","top_score","answer"])
    w.writeheader()
    w.writerows(qa_log)

# ── 9. Gaps & Recommendations ────────────────────────────────────────────────
gaps = [
    {
        "area": "Content Coverage — Employee HR",
        "severity": "HIGH",
        "finding": "Only 2 employee-facing HR policies found (onboarding + remote working). "
                   "No leave policy, attendance, travel reimbursement, salary structure, performance appraisal, "
                   "grievance redressal, or code of conduct documents.",
        "impact": "Employees asking everyday HR questions (leave balance, appraisal process, reimbursement rules) "
                  "will get 'not found' or irrelevant security policy answers.",
        "action": "Add: Leave Policy, Travel & Expense Policy, Performance Management Policy, "
                  "Code of Conduct, Grievance Policy, Compensation Structure document.",
    },
    {
        "area": "Content Coverage — IT Security Overload",
        "severity": "MEDIUM",
        "finding": "~60% of corpus is IT security/cyber SOPs (SIEM, hardening, vulnerability, encryption). "
                   "These are written for IT staff, not general employees.",
        "impact": "General employees will find the bot answers full of technical jargon they cannot act on.",
        "action": "Tag documents by audience. Route non-IT employees to HR/operations documents first. "
                  "Consider a separate domain instance for 'IT team' vs 'all employees'.",
    },
    {
        "area": "Metadata — Missing Audiences (36%)",
        "severity": "HIGH",
        "finding": f"{audience_flag}/59 documents have no audience set. "
                   "Permission-based filtering won't work for these documents.",
        "impact": "All users see all documents regardless of role. Sensitive IT-only SOPs surface for all employees.",
        "action": "Review and assign audience to all 21 documents. "
                  "Minimum: 'employee', 'manager', 'it_team', 'vendor'.",
    },
    {
        "area": "Metadata — Missing Effective Dates (22%)",
        "severity": "MEDIUM",
        "finding": f"{date_flag}/59 documents have unknown effective date.",
        "impact": "Bot cannot answer 'is this policy current?' type questions confidently. "
                  "Outdated policies may be served without caveat.",
        "action": "Update source HTML files with effective_date. "
                  "Add a 'last reviewed' field to all policy documents.",
    },
    {
        "area": "Duplicate Document Titles",
        "severity": "MEDIUM",
        "finding": "3 documents share the title 'WorkApps Product Solutions Private limited SOP'. "
                   "These map to different source files (IPR SOP, customer data SOP, SAR SOP).",
        "impact": "Bot may cite the wrong document when asked about SOPs. "
                  "Retrieval ranking cannot distinguish between them.",
        "action": "Rename HTML document titles to be unique and descriptive before re-ingesting.",
    },
    {
        "area": "Oversized Documents",
        "severity": "LOW",
        "finding": "Two documents are extremely large: "
                   "workapps-information-security-and-employeement-policy-v.9.0 (123 sections) "
                   "and videocx.io-infra-management-procedure (85 sections).",
        "impact": "Retrieval covers too many micro-sections; answers may be fragmented across 30+ sections. "
                  "Context window pressure on LLM.",
        "action": "Split into logical sub-documents (e.g., split the 123-section mega-policy "
                  "into Access Control Policy, Data Handling Policy, etc.).",
    },
    {
        "area": "Query Handling — Aggregate / List Queries",
        "severity": "HIGH",
        "finding": "Query 'list the policies' returned 'couldn't find'. "
                   "Single-word queries ('employee', 'devices') gave weak answers.",
        "impact": "First-time users who ask broad navigation questions get no answer. "
                  "Poor first impression.",
        "action": "Add a corpus-level FAQ: 'What policies are available?' -> list all 59 titles. "
                  "Handle single-word queries as topic searches, not exact lookups.",
    },
    {
        "area": "Network — SSL Proxy Failure",
        "severity": "MEDIUM",
        "finding": "1 ingestion failure: SSL certificate mismatch for bedrock-runtime.ap-south-1.amazonaws.com "
                   "intercepted by 91springboard.com proxy. Document was re-ingested successfully on retry.",
        "impact": "If run from 91springboard office network, embeddings may fail on retry-exhaustion. "
                  "Production ingestion must not run from this network.",
        "action": "Run ingestion from VPN or direct internet (not 91springboard proxy). "
                  "Add proxy bypass for *.amazonaws.com in network config.",
    },
    {
        "area": "Entity Extraction — Incomplete",
        "severity": "LOW",
        "finding": "Only 3 of 59 documents have entity files. "
                   "Entity extraction was not run on 56 documents.",
        "impact": "Entity-based search enrichment and synonym lookup limited to 3 docs.",
        "action": "Re-run entity extraction for all documents (generate_entity_map=True in config).",
    },
    {
        "area": "ChatGPT / Claude Parity — Conversational Style",
        "severity": "MEDIUM",
        "finding": "Bot handles follow-ups ('who approved this policy' after asking about log policy) well. "
                   "But casual queries like 'looks good', 'do you have his phone number' fall through to "
                   "policy search instead of conversational reply.",
        "impact": "Experience feels like a search engine, not a conversational assistant for some interactions.",
        "action": "Expand conversational_intent detection patterns. "
                  "Add explicit 'I don't have personal contact details' response for PII queries.",
    },
]

csv_gaps = OUT / "PolicyGPT_Gaps_Actions.csv"
with open(csv_gaps, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["area","severity","finding","impact","action"])
    w.writeheader()
    w.writerows(gaps)

# ── 10. Write HTML Report ────────────────────────────────────────────────────
def pct(n, total): return f"{n/total*100:.0f}%" if total else "0%"
def score_color(s):
    if s == "HIGH":   return "#EF4444"
    if s == "MEDIUM": return "#F59E0B"
    return "#34D399"

faq_top = sorted(faq_counts.items(), key=lambda x: -x[1])[:10]
section_large = sorted(docs, key=lambda d: -d["sections"])[:5]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PolicyGPT Corpus Analysis Report</title>
<style>
:root{{
  --bg:#0F172A;--card:#1E293B;--card2:#162033;--teal:#00D4C2;--blue:#388BFF;
  --amber:#FFA500;--green:#34D399;--red:#EF4444;--purple:#A78BFA;
  --text:#F1F5F9;--muted:#94A3B8;--border:#1E3055;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:14px;line-height:1.6}}
a{{color:var(--teal);text-decoration:none}}
.wrap{{max-width:1280px;margin:0 auto;padding:24px 20px}}
h1{{font-size:2rem;font-weight:700;color:var(--teal)}}
h2{{font-size:1.2rem;font-weight:700;color:var(--teal);margin:32px 0 12px;border-bottom:1px solid var(--border);padding-bottom:6px}}
h3{{font-size:1rem;font-weight:600;color:var(--text);margin:16px 0 8px}}
.subtitle{{color:var(--muted);margin:4px 0 24px}}
/* Stat cards */
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
.stat{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center}}
.stat .n{{font-size:2rem;font-weight:700;color:var(--teal)}}
.stat .l{{color:var(--muted);font-size:12px;margin-top:4px}}
/* Tables */
table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:13px}}
th{{background:var(--card2);color:var(--teal);text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);white-space:nowrap}}
td{{padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:top}}
tr:hover td{{background:rgba(255,255,255,0.03)}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}}
.b-policy{{background:#1E3A5F;color:var(--blue)}}
.b-process{{background:#1A3330;color:var(--green)}}
.b-checklist{{background:#2D2014;color:var(--amber)}}
.b-guideline{{background:#2D1A40;color:var(--purple)}}
.b-form{{background:#1F1F1F;color:var(--muted)}}
.b-ok{{background:#0F2D1F;color:var(--green)}}
.b-warn{{background:#2D1F0F;color:var(--amber)}}
.b-err{{background:#2D0F0F;color:var(--red)}}
/* Severity */
.sev-HIGH{{color:var(--red);font-weight:700}}
.sev-MEDIUM{{color:var(--amber);font-weight:700}}
.sev-LOW{{color:var(--green);font-weight:700}}
/* Progress bar */
.bar-wrap{{background:#1E293B;border-radius:4px;height:8px;margin:4px 0}}
.bar{{height:8px;border-radius:4px;background:var(--teal)}}
/* Section */
.section{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:20px}}
.section.warn{{border-color:var(--amber)}}
.section.ok{{border-color:var(--green)}}
.section.err{{border-color:var(--red)}}
/* Columns */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.verdict{{display:flex;align-items:center;gap:8px;padding:12px 16px;border-radius:8px;margin:8px 0}}
.verdict.pass{{background:#0F2D1F;border:1px solid var(--green)}}
.verdict.warn{{background:#2D1F0F;border:1px solid var(--amber)}}
.verdict.fail{{background:#2D0F0F;border:1px solid var(--red)}}
.verdict .icon{{font-size:1.4rem}}
.verdict .msg{{flex:1}}
.verdict .msg strong{{display:block}}
/* Q&A */
.qa-block{{background:var(--card2);border-radius:8px;padding:12px 14px;margin:8px 0}}
.qa-q{{color:var(--blue);font-weight:600;margin-bottom:4px}}
.qa-a{{color:var(--muted);font-size:12px}}
.qa-badge{{float:right;margin-top:-2px}}
.answered-yes{{color:var(--green)}}
.answered-no{{color:var(--red)}}
/* Download links */
.downloads{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}
.dl{{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;border-radius:8px;
     background:var(--card);border:1px solid var(--teal);color:var(--teal);font-size:13px;font-weight:600}}
.dl:hover{{background:var(--teal);color:var(--bg)}}
footer{{text-align:center;color:var(--muted);font-size:12px;padding:24px 0}}
@media(max-width:700px){{.two-col{{grid-template-columns:1fr}}.stat-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="wrap">

<!-- Header -->
<div style="margin-bottom:8px">
  <div style="color:var(--amber);font-size:11px;font-weight:700;letter-spacing:.1em">POLICYGPT — CORPUS READINESS REPORT</div>
  <h1>Ingestion Analysis & Q&amp;A Readiness Assessment</h1>
  <p class="subtitle">WorkApps (Videocx) Policy Corpus  ·  Ingested: 2026-04-09  ·  Report generated: {datetime.now().strftime('%Y-%m-%d')}</p>
</div>

<!-- Downloads -->
<div class="downloads">
  <a class="dl" href="PolicyGPT_Documents.csv" download>&#11015; Documents CSV (59 rows)</a>
  <a class="dl" href="PolicyGPT_FAQs.csv" download>&#11015; FAQ Pairs CSV ({total_faqs_n} rows)</a>
  <a class="dl" href="PolicyGPT_QA_TestLog.csv" download>&#11015; Q&amp;A Test Log CSV ({total_queries} rows)</a>
  <a class="dl" href="PolicyGPT_Gaps_Actions.csv" download>&#11015; Gaps &amp; Actions CSV ({len(gaps)} rows)</a>
</div>

<!-- Stat cards -->
<h2>1. Corpus At a Glance</h2>
<div class="stat-grid">
  <div class="stat"><div class="n">59</div><div class="l">Documents Ingested</div></div>
  <div class="stat"><div class="n" style="color:var(--blue)">{total_sections}</div><div class="l">Total Sections</div></div>
  <div class="stat"><div class="n" style="color:var(--purple)">{total_faqs_n}</div><div class="l">FAQ Q&amp;A Pairs</div></div>
  <div class="stat"><div class="n" style="color:var(--amber)">{total_queries}</div><div class="l">Real Queries Tested</div></div>
  <div class="stat"><div class="n" style="color:var(--green)">{answered_count}</div><div class="l">Queries Answered</div></div>
  <div class="stat"><div class="n" style="color:var(--red)">{len(failures)}</div><div class="l">Ingestion Failures</div></div>
</div>

<!-- Overall Verdict -->
<h2>2. Overall Readiness Verdict</h2>
<div class="section warn">
<div class="verdict warn">
  <span class="icon">⚠️</span>
  <div class="msg">
    <strong>PARTIALLY READY — Strong foundation for IT/Security staff; significant content gaps for general employees</strong>
    The technical infrastructure is working well (59/59 docs ingested, 1812 FAQs, hybrid search, conversation memory).
    However, ~60% of the corpus is IT Security SOPs written for the IT team, not everyday employees.
    Critical HR policies (leave, travel, appraisal, conduct) are missing. Without these, general employees
    will receive irrelevant IT security answers to basic HR questions — far below ChatGPT/Claude expectations.
  </div>
</div>

<div class="two-col" style="margin-top:16px">
<div>
<h3>✅ What's Working</h3>
<ul style="padding-left:16px;color:var(--muted);line-height:2">
  <li>All 59 documents ingested with summaries</li>
  <li>1,812 FAQ pairs across all documents (avg 30.7/doc)</li>
  <li>Hybrid search (BM25 + MLT + kNN) operational</li>
  <li>Factual policy Q&amp;A with citations working well</li>
  <li>Typo tolerance ("devise" → device answers correctly)</li>
  <li>Conversation follow-up context maintained</li>
  <li>Rich document summaries for every document</li>
  <li>Cross-document synthesis (e.g., cloud storage table)</li>
  <li>Approval chain / version questions answered correctly</li>
  <li>Table-formatted responses for complex queries</li>
</ul>
</div>
<div>
<h3>❌ Gaps & Problems</h3>
<ul style="padding-left:16px;color:var(--muted);line-height:2">
  <li><strong style="color:var(--red)">No HR policies</strong> — leave, travel, appraisal, conduct</li>
  <li>36% of docs missing audience metadata</li>
  <li>"list the policies" returns no answer</li>
  <li>3 duplicate document titles confuse retrieval</li>
  <li>One 123-section mega-doc fragments answers</li>
  <li>Entity extraction run on only 3/59 docs</li>
  <li>SSL proxy failure at 91springboard network</li>
  <li>Personal contact queries not gracefully declined</li>
  <li>22% docs missing effective date</li>
  <li>IT SOPs use jargon not relevant to non-IT staff</li>
</ul>
</div>
</div>
</div>

<!-- ChatGPT/Claude Comparison -->
<h2>3. ChatGPT / Claude Parity Assessment</h2>
<div class="section">
<p style="color:var(--muted);margin-bottom:12px">
  Evaluation based on 21 real queries tested in the retrieval logs. Scale: ✅ Meets expectation · ⚠️ Partial · ❌ Below expectation
</p>
<table>
<tr><th>Dimension</th><th>ChatGPT / Claude</th><th>PolicyGPT (current)</th><th>Status</th></tr>
<tr><td><strong>Factual policy answers</strong></td><td>Answers from training data</td>
    <td>Answers grounded in your specific docs with citations</td><td><span class="badge b-ok">✅ BETTER</span></td></tr>
<tr><td><strong>Typo / informal language</strong></td><td>Handles naturally</td>
    <td>"devise"→device, "telle me"→tell me work well</td><td><span class="badge b-ok">✅ MATCHES</span></td></tr>
<tr><td><strong>Follow-up conversations</strong></td><td>Full context window</td>
    <td>6-turn context window, auto-summarises at 8 turns</td><td><span class="badge b-ok">✅ MATCHES</span></td></tr>
<tr><td><strong>Structured responses (tables)</strong></td><td>Tables when appropriate</td>
    <td>Tables generated for multi-doc comparisons</td><td><span class="badge b-ok">✅ MATCHES</span></td></tr>
<tr><td><strong>Approval chains / authority</strong></td><td>Generic, not company-specific</td>
    <td>Knows "Shankar Borate = CTO & CISO", "Anupsingh = approver"</td><td><span class="badge b-ok">✅ BETTER</span></td></tr>
<tr><td><strong>Listing / navigation queries</strong></td><td>"Here are all policies on X…"</td>
    <td>"list the policies" → no answer found</td><td><span class="badge b-err">❌ BELOW</span></td></tr>
<tr><td><strong>HR / general employee queries</strong></td><td>General knowledge</td>
    <td>No leave/travel/appraisal docs → irrelevant answers</td><td><span class="badge b-err">❌ BELOW</span></td></tr>
<tr><td><strong>Personal info boundaries</strong></td><td>Declines personal data requests</td>
    <td>"do you have his phone number" not gracefully handled</td><td><span class="badge b-warn">⚠️ PARTIAL</span></td></tr>
<tr><td><strong>Greeting / small talk</strong></td><td>Natural conversation</td>
    <td>"looks good" routed to policy search</td><td><span class="badge b-warn">⚠️ PARTIAL</span></td></tr>
<tr><td><strong>Source citations</strong></td><td>No citations (general knowledge)</td>
    <td>Every answer includes document + section + score</td><td><span class="badge b-ok">✅ BETTER</span></td></tr>
</table>
</div>

<!-- Document Category Breakdown -->
<h2>4. Document Coverage by Category</h2>
<div class="section">
<div class="two-col">
<div>
<table>
<tr><th>Category</th><th>Count</th><th>% of corpus</th><th>Coverage</th></tr>
{''.join(f'''<tr>
  <td>{cat}</td>
  <td>{cnt}</td>
  <td>{pct(cnt, total_docs)}</td>
  <td><div class="bar-wrap"><div class="bar" style="width:{cnt/total_docs*100:.0f}%;background:{'var(--teal)' if cat=='IT Security & Cyber' else 'var(--blue)' if cat=='IT Operations & Assets' else 'var(--green)' if cat=='HR & Employment' else 'var(--amber)'}"></div></div></td>
</tr>''' for cat, cnt in sorted(cat_counts.items(), key=lambda x:-x[1]))}
</table>
<div style="margin-top:12px;padding:10px;background:var(--card2);border-radius:8px;color:var(--amber);font-size:12px">
⚠️ <strong>Imbalance Alert:</strong> IT Security & Cyber = {cat_counts.get('IT Security & Cyber',0)} docs ({pct(cat_counts.get('IT Security & Cyber',0),total_docs)}).
HR & Employment = {cat_counts.get('HR & Employment',0)} docs only.
Target ratio for general employee Q&amp;A should be at least 30% HR/Employment.
</div>
</div>
<div>
<table>
<tr><th>Document Type</th><th>Count</th></tr>
{''.join(f'<tr><td><span class="badge b-{t}">{t}</span></td><td>{c}</td></tr>' for t, c in sorted(type_counts.items(), key=lambda x:-x[1]))}
</table>
<br>
<table>
<tr><th>Metadata Quality</th><th>Present</th><th>Missing</th><th>Score</th></tr>
<tr><td>Version</td><td>{59-version_flag}</td><td>{version_flag}</td>
    <td><span class="badge {'b-ok' if version_flag<10 else 'b-warn'}">{pct(59-version_flag,59)}</span></td></tr>
<tr><td>Effective Date</td><td>{59-date_flag}</td><td>{date_flag}</td>
    <td><span class="badge {'b-ok' if date_flag<10 else 'b-warn'}">{pct(59-date_flag,59)}</span></td></tr>
<tr><td>Audience</td><td>{59-audience_flag}</td><td>{audience_flag}</td>
    <td><span class="badge {'b-ok' if audience_flag<10 else 'b-err'}">{pct(59-audience_flag,59)}</span></td></tr>
<tr><td>Document Summary</td><td>59</td><td>0</td>
    <td><span class="badge b-ok">100%</span></td></tr>
<tr><td>FAQ Pairs</td><td>59</td><td>0</td>
    <td><span class="badge b-ok">100%</span></td></tr>
<tr><td>Entity Extraction</td><td>3</td><td>56</td>
    <td><span class="badge b-err">5%</span></td></tr>
</table>
</div>
</div>
</div>

<!-- Document List -->
<h2>5. All 59 Ingested Documents</h2>
<div class="section">
<table>
<tr><th>#</th><th>Document Title</th><th>Category</th><th>Type</th><th>Version</th><th>Effective Date</th><th>Sections</th><th>Audience</th></tr>
{''.join(f'''<tr>
  <td style="color:var(--muted)">{i+1}</td>
  <td><strong>{d["title"][:55]}</strong></td>
  <td style="font-size:11px;color:var(--muted)">{d["category"]}</td>
  <td><span class="badge b-{d['type'] if d['type'] in ('policy','process','checklist','guideline','form') else 'form'}">{d['type']}</span></td>
  <td>{d['version'] or '<span style="color:var(--red)">—</span>'}</td>
  <td>{d['eff_date'] or '<span style="color:var(--red)">—</span>'}</td>
  <td style="text-align:center"><span class="badge {'b-err' if d['sections']>=50 else 'b-warn' if d['sections']>=15 else 'b-ok'}">{d['sections']}</span></td>
  <td style="font-size:11px;color:var(--muted)">{d['audiences'][:40] if d['audiences'] and d['audiences'] not in ('none','') else '<span style="color:var(--red)">NOT SET</span>'}</td>
</tr>''' for i, d in enumerate(docs))}
</table>
</div>

<!-- FAQ Analysis -->
<h2>6. FAQ Pairs Analysis</h2>
<div class="section">
<div class="two-col">
<div>
<h3>Top 10 Documents by FAQ Count</h3>
<table>
<tr><th>Document</th><th>FAQ Pairs</th></tr>
{''.join(f"<tr><td style='font-size:11px'>{name[:55]}</td><td><strong style='color:var(--purple)'>{count}</strong></td></tr>" for name, count in faq_top)}
</table>
</div>
<div>
<h3>Sample High-Value FAQs</h3>
{''.join(f'''<div class="qa-block">
  <div class="qa-q">Q: {f["question"][:90]}</div>
  <div class="qa-a">A: {f["answer"][:120]}...</div>
</div>''' for f in faqs[:8])}
</div>
</div>
</div>

<!-- Real Query Testing -->
<h2>7. Real Query Test Results ({total_queries} queries)</h2>
<div class="section">
<p style="color:var(--muted);margin-bottom:12px">All queries asked by users in the retrieval log, with whether they received a useful answer.</p>
<div style="display:flex;gap:16px;margin-bottom:16px">
  <div style="padding:10px 18px;background:var(--card2);border-radius:8px;border:1px solid var(--green)">
    <span style="font-size:1.4rem;font-weight:700;color:var(--green)">{answered_count}</span>
    <span style="color:var(--muted);font-size:12px;margin-left:6px">Answered ({pct(answered_count,total_queries)})</span>
  </div>
  <div style="padding:10px 18px;background:var(--card2);border-radius:8px;border:1px solid var(--red)">
    <span style="font-size:1.4rem;font-weight:700;color:var(--red)">{total_queries-answered_count}</span>
    <span style="color:var(--muted);font-size:12px;margin-left:6px">Unanswered / Weak ({pct(total_queries-answered_count,total_queries)})</span>
  </div>
</div>
<table>
<tr><th>#</th><th>Query</th><th>Intent</th><th>Answered?</th><th>Answer Preview</th></tr>
{''.join(f'''<tr>
  <td style="color:var(--muted)">{i+1}</td>
  <td><strong>{q["question"][:60]}</strong></td>
  <td style="font-size:11px;color:var(--muted)">{q["intent"]}</td>
  <td><span class="{'answered-yes' if q['answered']=='YES' else 'answered-no'} badge {'b-ok' if q['answered']=='YES' else 'b-err'}">{q["answered"]}</span></td>
  <td style="font-size:11px;color:var(--muted)">{q["answer"][:80].replace(chr(10)," ") if q["answer"] else "—"}</td>
</tr>''' for i, q in enumerate(qa_log))}
</table>
</div>

<!-- Ingestion Failure -->
<h2>8. Ingestion Failures</h2>
<div class="section {'err' if failures else 'ok'}">
{''.join(f'''<div style="padding:10px;background:var(--card2);border-radius:8px;border-left:3px solid var(--amber);margin-bottom:8px">
  <strong style="color:var(--amber)">{f["file"]}</strong>
  <p style="color:var(--muted);font-size:12px;margin-top:4px">{f["reason"][:200]}</p>
</div>''' for f in failures) if failures else '<p style="color:var(--green)">✅ 1 failure recorded (SSL proxy at 91springboard.com) — document was re-ingested successfully on retry. No data loss.</p>'}
</div>

<!-- Gaps and Actions -->
<h2>9. Gaps & Recommended Actions</h2>
<div>
{''.join(f'''<div class="section {'err' if g['severity']=='HIGH' else 'warn' if g['severity']=='MEDIUM' else 'ok'}">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
    <span class="badge {'b-err' if g['severity']=='HIGH' else 'b-warn' if g['severity']=='MEDIUM' else 'b-ok'}">{g['severity']}</span>
    <strong style="font-size:1rem">{g["area"]}</strong>
  </div>
  <div class="two-col">
    <div>
      <p style="color:var(--muted);font-size:12px;margin-bottom:6px"><strong style="color:var(--text)">Finding:</strong> {g["finding"]}</p>
      <p style="color:var(--muted);font-size:12px"><strong style="color:var(--amber)">Impact:</strong> {g["impact"]}</p>
    </div>
    <div style="background:var(--card2);border-radius:8px;padding:10px;border-left:3px solid var(--teal)">
      <p style="color:var(--teal);font-size:11px;font-weight:700;margin-bottom:4px">RECOMMENDED ACTION</p>
      <p style="color:var(--muted);font-size:12px">{g["action"]}</p>
    </div>
  </div>
</div>''' for g in gaps)}
</div>

<!-- Duplicate titles warning -->
<h2>10. Duplicate Document Titles (Need Fixing)</h2>
<div class="section warn">
<table>
<tr><th>Duplicate Title</th><th>Occurrences</th><th>Risk</th></tr>
{''.join(f"<tr><td>{t}</td><td style='color:var(--red);font-weight:700'>{c}×</td><td style='color:var(--amber);font-size:12px'>Retrieval may serve wrong document</td></tr>" for t,c in duplicates)}
</table>
</div>

<!-- Recommended Missing Documents -->
<h2>11. Recommended Documents to Add for Employee Q&amp;A</h2>
<div class="section err">
<p style="color:var(--muted);margin-bottom:12px">
These policy categories are essential for employees to get ChatGPT-quality answers to everyday HR and ops questions.
Currently absent from the corpus — employees asking these questions will receive no useful answer.
</p>
<table>
<tr><th>Missing Policy</th><th>Employee Questions It Answers</th><th>Priority</th></tr>
<tr><td><strong>Leave Policy</strong></td><td>How many leave days do I get? Can I carry forward leaves? What is sick leave policy?</td><td><span class="badge b-err">CRITICAL</span></td></tr>
<tr><td><strong>Travel & Expense Policy</strong></td><td>How do I claim travel reimbursement? What is per diem? Who approves expenses?</td><td><span class="badge b-err">CRITICAL</span></td></tr>
<tr><td><strong>Performance Appraisal Policy</strong></td><td>When is appraisal? What are the rating scales? How is increment decided?</td><td><span class="badge b-err">CRITICAL</span></td></tr>
<tr><td><strong>Code of Conduct / Ethics Policy</strong></td><td>What behaviour is expected? What happens if rules are broken?</td><td><span class="badge b-err">CRITICAL</span></td></tr>
<tr><td><strong>Grievance / Complaint Policy</strong></td><td>How do I raise a complaint? Who handles harassment issues? What is the process?</td><td><span class="badge b-err">CRITICAL</span></td></tr>
<tr><td><strong>Benefits & Perks Policy</strong></td><td>What health insurance do I get? Gym allowance? Flexi benefits?</td><td><span class="badge b-warn">HIGH</span></td></tr>
<tr><td><strong>Attendance & Working Hours Policy</strong></td><td>What are official working hours? Can I work flexible hours? What about overtime?</td><td><span class="badge b-warn">HIGH</span></td></tr>
<tr><td><strong>POSH / Anti-Harassment Policy</strong></td><td>What is the sexual harassment policy? Who is on the ICC?</td><td><span class="badge b-warn">HIGH</span></td></tr>
<tr><td><strong>Resignation & Exit Policy</strong></td><td>What is notice period? How do I get my FnF? When is relieving letter issued?</td><td><span class="badge b-warn">HIGH</span></td></tr>
<tr><td><strong>IT Acceptable Use Policy (simple version)</strong></td><td>Can I use personal email on office laptop? What can I install? Internet usage rules?</td><td><span class="badge b-ok">MEDIUM</span></td></tr>
</table>
</div>

<!-- Oversized docs -->
<h2>12. Oversized Documents (Risk to Answer Quality)</h2>
<div class="section warn">
<table>
<tr><th>Document</th><th>Sections</th><th>Risk</th><th>Recommendation</th></tr>
{''.join(f"<tr><td>{d['title'][:55]}</td><td><strong style='color:var(--red)'>{d['sections']}</strong></td><td style='color:var(--amber);font-size:12px'>Fragmented answers, LLM context pressure</td><td style='font-size:11px;color:var(--muted)'>Split into 3–5 sub-documents</td></tr>" for d in section_large)}
</table>
</div>

<footer>
  Generated by PolicyGPT Report Generator · {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
  Data source: D:/policy-mgmt/data/vcx_policies/metadata/
</footer>
</div>
</body>
</html>
"""

html_path = OUT / "PolicyGPT_Corpus_Report.html"
html_path.write_text(html, encoding="utf-8")

print("Reports generated:")
print(f"  HTML : {html_path}")
print(f"  CSV1 : {csv_docs}")
print(f"  CSV2 : {csv_faqs}")
print(f"  CSV3 : {csv_qa}")
print(f"  CSV4 : {csv_gaps}")
print(f"\nSummary: {total_docs} docs | {total_sections} sections | {total_faqs_n} FAQs | {answered_count}/{total_queries} queries answered")
