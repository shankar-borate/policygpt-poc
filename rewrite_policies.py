#!/usr/bin/env python3
"""
Policy Rewrite Script — WorkApps Product Solutions Pvt Ltd
==========================================================
Reads each HTML policy from SOURCE_DIR, sends it to Claude with the
SaaS-for-banking production-grade policy prompt, writes the rewritten
HTML to DEST_DIR.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python rewrite_policies.py

Optional flags (edit CONFIG below):
    SKIP_EXISTING  = True   → skip files already in improved/ (resume after interruption)
    MAX_FILES      = None   → process all; set to e.g. 3 for a test run
    MODEL          = "claude-opus-4-6"  or "claude-sonnet-4-6"
"""

import os
import re
import time
import csv
import html as html_lib
from pathlib import Path
from datetime import datetime

import anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
SOURCE     = Path("D:/policy-mgmt/data/vcx_policies")
DEST       = SOURCE / "improved"
DEST.mkdir(exist_ok=True)

MODEL          = "claude-opus-4-6"          # best quality; swap to claude-sonnet-4-6 to save cost
MAX_TOKENS     = 8000                        # max output tokens per policy
SKIP_EXISTING  = True                        # set False to re-process everything
MAX_FILES      = None                        # None = all 59; set to 3 for a test run
RETRY_LIMIT    = 3
RETRY_DELAY    = 10                          # seconds between retries

# Files to skip entirely (superseded stubs handled by improve_policies.py)
SKIP_FILES = {
    "log-monitoring-and-management-policy.html",
    "workapps---network-vulnerability-management-process-procedures.html",
}

# ── PROMPT ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a senior policy writer for a B2B SaaS technology company whose customers
are regulated financial institutions — banks, NBFCs, and insurance companies in
India. Your product (VideoCX.io) is deployed within bank/insurer environments,
making you a critical IT service provider subject to their outsourcing oversight.

Your task is to rewrite the policy document provided below into a production-grade,
audit-ready, customer-auditable policy that satisfies:
  (a) Internal governance for the SaaS company itself
  (b) Due diligence requirements of bank and insurer customers
  (c) Regulatory expectations when your company is reviewed as a third-party vendor
"""

USER_PROMPT_TEMPLATE = """\
Rewrite the policy document below following ALL rules precisely.

════════════════════════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════════════════════════

1. OUTPUT FORMAT
   - Return a single complete HTML document (<!DOCTYPE html> … </html>)
   - Clean semantic HTML: h1–h4, p, ul/ol/li, table/thead/tbody/tr/td/th
   - Inline CSS only — no external stylesheets, no JavaScript

2. VISUAL DIFF — COLOR CODING (no text labels, no "(modified)" markers)
   - Content you ADD that was absent in the original:
     <span style="background:#fff9c4"> … </span>   ← yellow
   - Content you REWRITE for clarity / compliance language:
     <span style="background:#e8f5e9"> … </span>   ← green
   - Original content kept verbatim: no colour, no wrapper

3. DOCUMENT STRUCTURE — in this exact order:

   COVER PAGE
     Document Title
     Document ID | Version | Classification
     Effective Date | Next Review Date | Review Cycle
     Policy Owner | Approved By | Approval Authority
     Audience | Department
     Company: WorkApps Product Solutions Pvt Ltd
     Product: VideoCX.io

   DOCUMENT CONTROL
     Change History table: Ver | Date | Author | Reviewed By | Change Summary | Approval Ref
     Distribution table: Role | Department | Copy Type

   TABLE OF CONTENTS (anchor links)

   1. PURPOSE — business need + which customer regulatory obligation this satisfies
   2. SCOPE — internal (employees, contractors, vendors) + external (bank/insurer customers)
   3. REGULATORY & COMPLIANCE FRAMEWORK — all applicable laws, regulations, standards
   4. DEFINITIONS — Term | Definition | Source
   5. POLICY STATEMENTS — numbered clauses 5.1, 5.2, 5.2.1 …
   6. ROLES & RESPONSIBILITIES — Role | Accountability | Key Duties
      Always include: Board of Directors, CEO, CTO, CISO, DPO, Compliance Manager,
      Engineering Lead, Customer Success, Internal Audit
   7. OPERATING PROCEDURE — numbered steps, decision points, customer-facing vs internal
   8. RISK & CONTROL MATRIX — Risk | Likelihood | Impact | Control | Owner | Type
   9. CUSTOMER OBLIGATIONS & SLA COMMITMENTS
   10. EXCEPTION MANAGEMENT — approval authority, time-bound, customer notification Y/N
   11. NON-COMPLIANCE & CONSEQUENCES — internal disciplinary + external notification
   12. REVIEW & GOVERNANCE — cycle, trigger events, approval authority
   13. RELATED DOCUMENTS & REFERENCES

4. LANGUAGE RULES
   - Formal professional English — customer auditor-ready
   - Mandatory controls: "must" / "shall" — never "should" for controls
   - Active voice throughout
   - Sentences ≤ 25 words
   - Hierarchical clause numbering: 5.1 → 5.1.1 → 5.1.1.a
   - Define every acronym on first use

5. REGULATORY MAPPING
   After any clause driven by regulation/standard add:
   <span style="color:#6b7280;font-size:9pt">[Ref: …]</span>

   As a company (internal governance):
     ISO 27001:2022, SOC 2 Type II, PCI-DSS v4.0, IT Act 2000,
     CERT-In Directions 2022, DPDP Act 2023

   As a vendor to banks:
     RBI Guidelines on IT Outsourcing 2023,
     RBI Cyber Security Framework for Banks 2016,
     RBI Master Direction on IT Governance 2023

   As a vendor to insurers:
     IRDAI Information & Cyber Security Guidelines 2023,
     IRDAI Outsourcing of Activities Regulations 2017

   Cross-cutting:
     PMLA 2002, FEMA 1999, SEBI Circular on Cyber Security 2023

6. CLASSIFICATION BANNER — first element inside <body>:
   INTERNAL USE ONLY      → background:#e3f2fd; border-left:5px solid #1565c0
   CONFIDENTIAL           → background:#fff9c4; border-left:5px solid #f59e0b
   STRICTLY CONFIDENTIAL  → background:#fce4ec; border-left:5px solid #c62828
   CUSTOMER SHAREABLE     → background:#e8f5e9; border-left:5px solid #2e7d32
   Infer from content. Default to CONFIDENTIAL.
   Add text: "This document may be shared with customer auditors upon request under NDA."
   where classification allows.

7. STYLE SPECIFICATION
   - Max width 900px, margin 0 auto, padding 24px 40px
   - Body: Arial/Helvetica, 11pt, line-height 1.8, color #1e293b, background #ffffff
   - Cover block: background #f0f4ff; border-left 6px solid #1a3a8f; border-radius 4px; padding 20px
   - h1: 18pt, color #1a3a8f
   - h2: 13pt, color #1a3a8f, border-bottom 2px solid #c8d5f0, padding-bottom 4px
   - h3: 11pt, color #1a56db
   - th: background #1a3a8f; color #ffffff; padding 8px 10px
   - td: padding 6px 10px; border-bottom 1px solid #e2e8f0
   - tr:nth-child(even): background #f8fafc
   - Table: border-collapse collapse; width 100%; font-size 10pt

8. DO NOT CHANGE
   - Existing policy decisions, thresholds, or penalties
   - Document ID, version numbers, or dates already present
   - Well-written content (keep verbatim, no colour)
   - Missing section → add in yellow + <!-- TODO: confirm with Policy Owner -->

════════════════════════════════════════════════════════════
SOURCE POLICY DOCUMENT
════════════════════════════════════════════════════════════

{policy_content}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(html_content: str, max_chars: int = 60_000) -> str:
    """Strip HTML tags and return clean text (capped to avoid token overflow)."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.S|re.I)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.S|re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&#xa0;', ' ').replace('&nbsp;', ' ')
    text = re.sub(r'\s{3,}', '\n\n', text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + '\n\n[... document truncated for length ...]'
    return text

def call_claude(client: anthropic.Anthropic, policy_text: str, filename: str) -> str:
    prompt = USER_PROMPT_TEMPLATE.format(policy_content=policy_text)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            print(f"    Rate limit hit — waiting {RETRY_DELAY}s (attempt {attempt}/{RETRY_LIMIT})")
            time.sleep(RETRY_DELAY)
        except anthropic.APIError as e:
            print(f"    API error: {e} (attempt {attempt}/{RETRY_LIMIT})")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                raise
    raise RuntimeError(f"Failed after {RETRY_LIMIT} attempts for {filename}")

def ensure_html_doc(response: str, filename: str) -> str:
    """If the model returned HTML wrapped in a code block, unwrap it."""
    m = re.search(r'```html\s*(<!DOCTYPE.*?)</s*```', response, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*(<!DOCTYPE.*?)```', response, re.S | re.I)
    if m:
        return m.group(1).strip()
    if '<!DOCTYPE' in response or '<html' in response.lower():
        # Find start of HTML
        start = response.find('<!DOCTYPE')
        if start == -1:
            start = response.lower().find('<html')
        return response[start:].strip()
    # Fallback: wrap in minimal HTML
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{filename}</title></head><body>'
            f'{response}</body></html>')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Set it with:  set ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = anthropic.Anthropic(api_key=api_key)

    html_files = sorted(SOURCE.glob("*.html"))
    if MAX_FILES:
        html_files = html_files[:MAX_FILES]

    total = len(html_files)
    print(f"Model   : {MODEL}")
    print(f"Files   : {total}")
    print(f"Output  : {DEST}\n")

    report = []
    ok = skipped = errors = 0

    for i, src in enumerate(html_files, 1):
        fn = src.name
        out_path = DEST / fn

        if fn in SKIP_FILES:
            print(f"[{i:>2}/{total}] SKIP (superseded) — {fn}")
            skipped += 1
            continue

        if SKIP_EXISTING and out_path.exists():
            print(f"[{i:>2}/{total}] SKIP (exists)     — {fn}")
            skipped += 1
            continue

        print(f"[{i:>2}/{total}] Processing — {fn}")
        raw = src.read_text(encoding='utf-8', errors='ignore')
        policy_text = extract_text(raw)
        char_count  = len(policy_text)
        print(f"         chars={char_count:,}  sending to {MODEL} …")

        t0 = time.time()
        try:
            response = call_claude(client, policy_text, fn)
            improved = ensure_html_doc(response, fn)
            out_path.write_text(improved, encoding='utf-8')
            elapsed = round(time.time() - t0, 1)
            out_kb  = out_path.stat().st_size // 1024
            print(f"         Done in {elapsed}s — {out_kb} KB written")
            report.append({'file': fn, 'status': 'OK',
                           'chars': char_count, 'elapsed_s': elapsed,
                           'output_kb': out_kb})
            ok += 1
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"         ERROR: {e}")
            report.append({'file': fn, 'status': f'ERROR: {e}',
                           'chars': char_count, 'elapsed_s': elapsed,
                           'output_kb': 0})
            errors += 1

        # Small pause between calls to respect rate limits
        if i < total:
            time.sleep(2)

    # Write report
    report_path = DEST / 'rewrite_report.csv'
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file','status','chars','elapsed_s','output_kb'])
        w.writeheader()
        w.writerows(report)

    print(f"\n{'='*60}")
    print(f"Complete.  OK={ok}  Skipped={skipped}  Errors={errors}")
    print(f"Report: {report_path}")

if __name__ == '__main__':
    main()
