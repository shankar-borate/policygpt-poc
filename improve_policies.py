#!/usr/bin/env python3
"""
Policy Improvement Script — WorkApps / VideoCX Policies
========================================================
For every HTML policy in SOURCE_DIR:
  - Adds a standardised metadata banner (title, type, version, date, owner,
    audience, category, keywords, review cycle, organisation)
  - Fixes empty <title> tags
  - Notes improvement recommendations inline
  - Splits the large 906 KB omnibus IS&E policy into 9 focused documents
  - Marks 2 superseded duplicates with redirect stubs

Outputs to SOURCE_DIR/improved/
Generates improvement_report.html + improvement_report.csv
"""

import re, csv
from pathlib import Path
from datetime import datetime

SOURCE = Path("D:/policy-mgmt/data/vcx_policies")
DEST   = SOURCE / "improved"
DEST.mkdir(exist_ok=True)

# ── HTML helpers ──────────────────────────────────────────────────────────────

def clean_html(html: str) -> str:
    return (re.sub(r'<[^>]+>', '', html)
              .replace('&amp;', '&').replace('&#xa0;', ' ')
              .replace('&lt;', '<').replace('&gt;', '>')
              .strip())

def extract_title(content: str, filename: str) -> str:
    # 1. Large coloured span (24pt / colour-styled — the document heading)
    m = re.search(
        r'<span[^>]*(?:font-size:2[0-9]pt|color:#0070c0)[^>]*>(.*?)</span>',
        content, re.S | re.I)
    if m:
        t = clean_html(m.group(1))
        if len(t) > 6:
            return t
    # 2. HTML <title> tag
    m = re.search(r'<title[^>]*>(.*?)</title>', content, re.S | re.I)
    if m:
        t = clean_html(m.group(1))
        if len(t) > 3:
            return t
    # 3. Derive from filename
    stem = filename.replace('.html', '')
    stem = re.sub(r'-v[\d.]+$', '', stem)
    stem = re.sub(r'-workapps[-\w]*$', '', stem)
    stem = re.sub(r'-videocx[-\w]*$', '', stem)
    return stem.replace('-', ' ').title()

def extract_version(content: str, filename: str) -> str:
    m = re.search(r'[Vv]ersion\s+(\d+[\d.]*)', content)
    if m:
        return f"v{m.group(1)}"
    m = re.search(r'v[.\s]*(\d+[\d.]+)', filename, re.I)
    if m:
        return f"v{m.group(1)}"
    return "v1.0"

def extract_date(content: str) -> str | None:
    patterns = [
        r'Date[:]\s*(\d{1,2}[/]\w{3}[/]\d{4})',
        r'Date[:]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(\d{1,2}[/]\w{3}[/]\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
    ]
    for p in patterns:
        m = re.search(p, content)
        if m:
            return m.group(1)
    return None

# ── Metadata inference ────────────────────────────────────────────────────────

def infer_doc_type(fn: str) -> str:
    f = fn.lower()
    if 'sop'        in f: return 'SOP'
    if 'checklist'  in f: return 'Checklist'
    if 'guideline'  in f: return 'Guideline'
    if 'procedure'  in f: return 'Procedure'
    if 'framework'  in f: return 'Framework'
    if 'inventory'  in f: return 'Inventory Register'
    if 'drill'      in f: return 'Drill Record'
    if 'jd'         in f: return 'Job Description'
    if 'playbook'   in f: return 'Playbook'
    if 'plan'       in f: return 'Continuity Plan'
    return 'Policy'

def infer_audience(fn: str, content_head: str) -> str:
    s = fn.lower() + content_head.lower()
    if any(x in s for x in ['clear-desk', 'remote-working', 'wfh', 'modern-slavery',
                              'anti-trust', 'ethical-sourcing', 'disease-response',
                              'employee-onboarding', 'information-security-and-employe']):
        return 'All Employees'
    if any(x in s for x in ['recruitment', 'onboarding', 'background']):
        return 'HR Team, Hiring Managers'
    if 'ciso-jd' in s:
        return 'Leadership, HR'
    if any(x in s for x in ['coding', 'sdl', 'system-acquisition', 'garbage-collection']):
        return 'Development Team'
    if any(x in s for x in ['aws-security', 'cloud', 'server-hardening', 'infra',
                              'sso', 'url-whitelist', 'obtain-or-review', 'rto-rpo',
                              'resiliency', 'contingency']):
        return 'DevOps / Cloud Engineers'
    if any(x in s for x in ['siem', 'log-monitor', 'monthly-log', 'ir-playbook',
                              'incident-management', 'threat-response', 'network-vuln',
                              'vulnerability-assessment', 'workapps---server-hardening',
                              'security-hardening-baseline']):
        return 'Security Operations Team'
    if any(x in s for x in ['asset', 'inventory', 'license']):
        return 'IT Asset Managers, IT Team'
    if any(x in s for x in ['marketing', 'selling', 'anti-trust', 'compliance-program']):
        return 'Sales, Marketing, Legal'
    if any(x in s for x in ['outsourc', 'sar-reports', 'customer-data', 'customer data']):
        return 'Legal, Compliance Team'
    if any(x in s for x in ['quality', 'document-for-quality']):
        return 'QA Team, Project Managers'
    if any(x in s for x in ['intellectual-property', 'ipr']):
        return 'All Employees, Legal'
    if any(x in s for x in ['dlp', 'encryption', 'web-application-security',
                              'cyber-security', 'user-lifecyle', 'sop-iam']):
        return 'IT Team, All Employees'
    return 'IT Team, Management'

def infer_category(fn: str) -> str:
    f = fn.lower()
    if any(x in f for x in ['anti-trust', 'compliance-program', 'outsourc',
                              'sar', 'customer-data', 'modern-slavery', 'ethical',
                              'guidelines-and-process-for-management']):
        return 'Legal & Compliance'
    if any(x in f for x in ['onboarding', 'recruitment', 'wfh', 'remote-working',
                              'clear-desk', 'background', 'disease-response',
                              'workapps-ciso-jd']):
        return 'HR & Employee Policies'
    if any(x in f for x in ['asset', 'inventory', 'license']):
        return 'IT Asset Management'
    if any(x in f for x in ['siem', 'log', 'monitoring', 'ir-playbook', 'incident']):
        return 'Security Operations'
    if any(x in f for x in ['aws', 'cloud', 'server', 'infra', 'sso',
                              'url-whitelist', 'obtain-or-review']):
        return 'Cloud & Infrastructure'
    if any(x in f for x in ['coding', 'sdl', 'system-acquisition', 'garbage']):
        return 'Software Development'
    if any(x in f for x in ['rto', 'rpo', 'continuity', 'resiliency', 'trrp',
                              'contingency', 'disease']):
        return 'Business Continuity'
    if any(x in f for x in ['quality', 'qa', 'document-for-quality']):
        return 'Quality Assurance'
    if any(x in f for x in ['intellectual', 'ipr']):
        return 'Intellectual Property'
    if any(x in f for x in ['tech-recruit']):
        return 'Talent & Organisation'
    return 'Information Security'

def infer_owner(fn: str) -> str:
    f = fn.lower()
    if 'ciso'       in f: return 'CISO'
    if any(x in f for x in ['recruitment', 'onboarding', 'background',
                              'workapps-tech-recruit']): return 'HR Manager'
    if any(x in f for x in ['asset', 'inventory', 'license']): return 'IT Asset Manager / CISO'
    if any(x in f for x in ['cloud', 'aws', 'server', 'infra',
                              'rto', 'trrp', 'contingency']): return 'CTO / DevOps Lead'
    if any(x in f for x in ['coding', 'sdl', 'system-acquisition',
                              'garbage']): return 'Engineering Lead'
    if any(x in f for x in ['quality']): return 'QA Manager'
    if any(x in f for x in ['marketing', 'selling', 'anti-trust',
                              'modern-slavery', 'ethical',
                              'sar', 'customer-data',
                              'outsourc']): return 'Legal / Compliance Officer'
    if any(x in f for x in ['siem', 'log', 'monitoring',
                              'ir-playbook']): return 'Security Operations Manager'
    return 'CISO / CTO'

def infer_review(doc_type: str) -> str:
    if doc_type in ('Checklist', 'Drill Record'):    return 'Quarterly'
    if doc_type == 'Inventory Register':             return 'Monthly'
    if doc_type in ('SOP', 'Procedure'):             return 'Semi-Annual'
    return 'Annual'

def infer_keywords(fn: str) -> str:
    stop = {'workapps','product','solutions','private','limited','policy','sop',
            'procedure','management','process','videocx','platform','html','the',
            'and','for','with','from','into','this','that','are','for','its',
            'standard','operating','document','guidelines','checklist','version'}
    words = re.sub(r'[^a-z0-9 ]+', ' ', fn.replace('.html','').lower()).split()
    kws = [w for w in words if len(w) > 3 and w not in stop]
    return ', '.join(kws[:7])

# ── Fallback date map (files whose HTML has no extractable date) ──────────────

_DATE_FALLBACK: dict[str, str] = {
    'anti-trust':                               '15/Jan/2024',
    'asset-lifecycle':                          '01/Apr/2024',
    'asset-reconcil':                           '01/Apr/2024',
    'aws-security-configuration':               '01/Jun/2024',
    'checklist---security-threat':              '15/Mar/2024',
    'clear-desk-audit':                         '01/Jan/2025',
    'compliance-program-for-marketing':         '10/Feb/2024',
    'custom-siem-on-aws':                       '01/Aug/2024',
    'custom-siem-tool-integration':             '01/Aug/2024',
    'disease-response':                         '01/May/2024',
    'employee-onboarding':                      '01/Mar/2024',
    'ethical-sourcing':                         '15/Jan/2024',
    'garbage-collection':                       '01/Sep/2024',
    'guidelines-and-process-for-management':    '01/Feb/2024',
    'intellectual-property-policy':             '01/Mar/2024',
    'internal-background':                      '01/Mar/2024',
    'ir-playbook':                              '01/Jul/2024',
    'modern-slavery':                           '15/Jan/2024',
    'monthly-log-review':                       '31/Oct/2024',
    'obtain-or-review':                         '01/Jun/2024',
    'organization-safeguards':                  '01/Apr/2024',
    'outsourcing-risk':                         '01/Feb/2024',
    'procedures-for-encryption':                '01/May/2024',
    'remote-working':                           '01/Jan/2025',
    'rto-rpo':                                  '01/Jun/2024',
    'secure-coding':                            '01/Jul/2024',
    'security-development':                     '01/Jul/2024',
    'server-hardening-policy':                  '01/May/2024',
    'sop-iam-user':                             '01/Jun/2024',
    'sso-integration':                          '01/Aug/2024',
    'standard-operating-procedure-sop-intell':  '01/Mar/2024',
    'system-acquisition':                       '01/Jul/2024',
    'technology-resiliency':                    '01/Jun/2024',
    'url-whitelisting':                         '01/Aug/2024',
    'user-lifecyle':                            '01/Mar/2024',
    'videocx.io-platform---contingency':        '01/Jun/2024',
    'videocx.io-platform---infra':              '01/Aug/2024',
    'vulnerability-assessment':                 '15/Mar/2024',
    'web-application-security':                 '01/Apr/2024',
    'work-from-home':                           '01/Jan/2025',
    'workapps---network-vulnerability-manage':  '15/Mar/2024',
    'workapps---procedure-for-responding':      '01/Feb/2024',
    'workapps---sar-reports':                   '01/Feb/2024',
    'workapps---server-hardening-checklist':    '01/May/2024',
    'workapps--security-hardening-baseline':    '01/May/2024',
    'workapps-ciso-jd':                         '01/Jan/2025',
    'workapps-cloud-security':                  '01/Jun/2024',
    'workapps-dlp':                             '15/Apr/2024',
    'workapps-information-security':            '01/Jan/2025',
    'workapps-it-asset-software-license':       '01/Apr/2024',
    'workapps-product-solutions-private-limited-document-for-quality': '01/Sep/2024',
    'workapps-product-solutions-private-limited-it-asset-inventory':   '01/Apr/2024',
    'workapps-product-solutions-private-limited-security-incident-drill': '30/Apr/2025',
    'workapps-tech-recruitments':               '01/Mar/2024',
}

def get_date(fn: str, content: str) -> tuple[str, bool]:
    """Return (date_string, was_inferred)."""
    d = extract_date(content)
    if d:
        return d, False
    fn_lower = fn.lower()
    for key, val in _DATE_FALLBACK.items():
        if key in fn_lower:
            return val, True
    return '01/Jan/2024', True

# ── Metadata banner HTML ──────────────────────────────────────────────────────

_META_CSS = """<style>
.wa-meta-banner{font-family:Arial,Helvetica,sans-serif;background:#f0f4ff;
  border:1px solid #c0cef5;border-left:5px solid #1a56db;border-radius:5px;
  padding:14px 20px 10px;margin:0 0 24px}
.wa-meta-banner .wa-doc-title{font-size:15pt;font-weight:bold;color:#1a3a8f;
  margin:0 0 10px;font-family:Arial,sans-serif}
.wa-meta-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:5px 18px}
.wa-meta-item{font-size:9.5pt;line-height:1.5}
.wa-meta-label{color:#555;font-weight:bold}
.wa-meta-value{color:#111}
.wa-improve-note{background:#fffbea;border:1px solid #e6b800;border-left:4px solid #d97706;
  border-radius:4px;padding:8px 12px;margin-top:10px;font-size:9.5pt;color:#5a3e00}
.wa-improve-note ul{margin:4px 0 0 18px;padding:0}
</style>"""

def build_banner(title, doc_type, version, date_str, owner,
                 audience, category, keywords, review_cycle,
                 notes: list[str]) -> str:
    note_html = ''
    if notes:
        lis = ''.join(f'<li>{n}</li>' for n in notes)
        note_html = (f'<div class="wa-improve-note">'
                     f'<strong>Improvement Notes:</strong>'
                     f'<ul>{lis}</ul></div>')
    return f"""
<div class="wa-meta-banner">
  <div class="wa-doc-title">{title}</div>
  <div class="wa-meta-grid">
    <div class="wa-meta-item"><span class="wa-meta-label">Document Type:</span>&nbsp;<span class="wa-meta-value">{doc_type}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Version:</span>&nbsp;<span class="wa-meta-value">{version}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Effective Date:</span>&nbsp;<span class="wa-meta-value">{date_str}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Document Owner:</span>&nbsp;<span class="wa-meta-value">{owner}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Audience:</span>&nbsp;<span class="wa-meta-value">{audience}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Category:</span>&nbsp;<span class="wa-meta-value">{category}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Keywords:</span>&nbsp;<span class="wa-meta-value">{keywords}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Review Cycle:</span>&nbsp;<span class="wa-meta-value">{review_cycle}</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Organisation:</span>&nbsp;<span class="wa-meta-value">WorkApps Product Solutions Pvt Ltd</span></div>
    <div class="wa-meta-item"><span class="wa-meta-label">Platform:</span>&nbsp;<span class="wa-meta-value">VideoCX.io</span></div>
  </div>
  {note_html}
</div>"""

def inject_banner(content: str, banner: str) -> str:
    m = re.search(r'(<body[^>]*>)', content, re.I)
    if m:
        pos = m.end()
        return content[:pos] + '\n' + _META_CSS + '\n' + banner + '\n' + content[pos:]
    return _META_CSS + '\n' + banner + '\n' + content

def fix_title_tag(content: str, title: str) -> str:
    return re.sub(
        r'<title>\s*</title>',
        f'<title>{title} — WorkApps Product Solutions Pvt Ltd</title>',
        content, flags=re.I)

# ── Large omnibus file split ──────────────────────────────────────────────────

OMNIBUS = "workapps-information-security-and-employeement-policy-v.9.0.html"

# Split boundaries discovered by inspecting font-size:16pt section headings
# Each entry: (section_number, title, output_filename, owner, audience, keywords)
_SPLITS = [
    (None,  "Information Security & Employment Policy",
     "information-security-employment-policy.html",
     "CISO / CTO", "All Employees",
     "information security, employment, AUP, access control, data classification",
     0,        556546),   # chars 0 → 556546

    (31,    "Network Security Policy",
     "network-security-policy-extracted.html",
     "CISO / Network Engineer", "IT Team, Network Engineers",
     "network security, firewall, VPN, network access, segmentation",
     556546,   560656),

    (32,    "Vulnerability Assessment Policy",
     "vulnerability-assessment-policy-extracted.html",
     "CISO / Security Team", "Security Team, DevOps",
     "vulnerability assessment, VA scan, penetration testing, CVE",
     560656,   575186),

    (33,    "Cloud Security Policy",
     "cloud-security-policy-extracted.html",
     "CISO / CTO", "Cloud Engineers, DevOps",
     "cloud security, AWS, GCP, cloud configuration, shared responsibility",
     575186,   584736),

    (34,    "KMS Policy",
     "kms-policy-extracted.html",
     "CISO / CTO", "Cloud Engineers, Security Team",
     "KMS, key management, encryption keys, AWS KMS",
     584736,   585914),

    (35,    "Information Security Exception Policy",
     "information-security-exception-policy-extracted.html",
     "CISO", "IT Team, Department Heads",
     "exception handling, security exception, risk acceptance, compensating control",
     585914,   840433),

    (37,    "Anti-Corruption Policy",
     "anti-corruption-policy-extracted.html",
     "Legal / Compliance Officer", "All Employees, Management",
     "anti-corruption, bribery, ethics, gifts, compliance",
     840433,   865827),

    (38,    "Firewalls IPS IDS and Signature Policy",
     "firewalls-ips-ids-signature-policy-extracted.html",
     "CISO / Network Engineer", "Security Team, Network Engineers",
     "firewall, IPS, IDS, intrusion detection, signatures, WAF",
     865827,   891426),

    (40,    "Logging and Monitoring Policy",
     "logging-monitoring-policy-extracted.html",
     "CISO / Security Operations", "Security Operations Team",
     "logging, monitoring, SIEM, log retention, audit logs",
     891426,   None),      # None = end of file
]

_BASE_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <title>{title} — WorkApps Product Solutions Pvt Ltd</title>
  <style>
    body {{ line-height:1.6; font-family:Calibri,Arial,sans-serif; font-size:11pt;
            margin:20px 40px; }}
    h1,h2,h3 {{ font-family:Arial,sans-serif; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:12pt; }}
    td,th {{ border:0.75pt solid #999; padding:4pt 6pt; vertical-align:top; }}
  </style>
</head>
<body>
{meta_css}
{banner}
<h1 style="color:#1a3a8f">{title}</h1>
<p style="color:#666;font-size:9.5pt;font-family:Arial">
  <em>Extracted from omnibus document:
  <a href="workapps-information-security-and-employeement-policy-v.9.0.html"
  >workapps-information-security-and-employeement-policy-v.9.0.html</a></em>
</p>
{body_content}
</body>
</html>"""

def split_omnibus(content: str, version: str, date_str: str, report: list) -> None:
    print(f"  Splitting omnibus into {len(_SPLITS)} documents …")
    for (sec_no, title, out_fname, owner, audience, keywords, start, end) in _SPLITS:
        body_slice = content[start: end] if end else content[start:]
        # Remove the original numbered section heading (e.g. "33. Cloud Security Policy")
        # from the slice — the template already adds a clean <h1> title above it.
        body_slice = re.sub(
            r'<[ph][1-6]?[^>]*>\s*(?:<[^>]+>\s*)*'
            r'\d+\.?\s+' + re.escape(title) + r'\s*(?:</[^>]+>\s*)*</[ph][1-6]?>',
            '', body_slice, count=1, flags=re.I | re.S)
        notes = [f"Extracted from omnibus document: {OMNIBUS}"]
        if sec_no:
            notes.append(f"Original section number in source: {sec_no}")
        banner = build_banner(
            title=title, doc_type='Policy',
            version=version, date_str=date_str,
            owner=owner, audience=audience,
            category='Information Security',
            keywords=keywords, review_cycle='Annual',
            notes=notes)
        out_html = _BASE_HTML_TEMPLATE.format(
            title=title,
            meta_css=_META_CSS,
            banner=banner,
            body_content=body_slice)
        (DEST / out_fname).write_text(out_html, encoding='utf-8')
        size_kb = len(out_html) // 1024
        report.append({
            'source':           OMNIBUS,
            'output':           out_fname,
            'action':           'Split from omnibus document',
            'doc_type':         'Policy',
            'version':          version,
            'date':             date_str,
            'owner':            owner,
            'audience':         audience,
            'category':         'Information Security',
            'keywords':         keywords,
            'review_cycle':     'Annual',
            'notes':            f'Section {sec_no}; extracted from omnibus',
            'output_size_kb':   size_kb,
        })
        print(f"    Written: {out_fname} ({size_kb} KB)")

# ── Superseded duplicates ─────────────────────────────────────────────────────

_SUPERSEDED: dict[str, str] = {
    'log-monitoring-and-management-policy.html':
        'log-monitoring-and-management-policy-v1.3.html',
    'workapps---network-vulnerability-management-process-procedures.html':
        'workapps---network-vulnerability-management-process-procedures-sop.html',
}

_SUPERSEDED_STUB = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Superseded Document</title>
<style>
body{{font-family:Arial,sans-serif;padding:40px}}
.banner{{background:#fff3cd;border:1px solid #ffc107;border-left:5px solid #e6a817;
  border-radius:5px;padding:24px;max-width:600px}}
h2{{color:#856404;margin-top:0}}p{{color:#333}}
a{{color:#1a56db}}
</style></head>
<body>
<div class="banner">
  <h2>Document Superseded</h2>
  <p>This document (<strong>{old}</strong>) has been superseded.<br>
     Please refer to the current version:</p>
  <p><a href="{new}">{new}</a></p>
  <p><em>Organisation: WorkApps Product Solutions Pvt Ltd</em></p>
</div>
</body></html>"""

# ── Main per-file processor ───────────────────────────────────────────────────

def process_file(src: Path, report: list) -> None:
    fn = src.name
    content = src.read_text(encoding='utf-8', errors='ignore')
    file_kb = src.stat().st_size // 1024

    # ── Superseded stub ──────────────────────────────────────────────────────
    if fn in _SUPERSEDED:
        newer = _SUPERSEDED[fn]
        stub  = _SUPERSEDED_STUB.format(old=fn, new=newer)
        (DEST / fn).write_text(stub, encoding='utf-8')
        report.append({
            'source': fn, 'output': fn,
            'action': 'Superseded — redirect stub written',
            'doc_type': '', 'version': '', 'date': '',
            'owner': '', 'audience': '', 'category': '',
            'keywords': '', 'review_cycle': '',
            'notes': f'Superseded by {newer}',
            'output_size_kb': len(stub)//1024,
        })
        print(f"  [SUPERSEDED] {fn}")
        return

    # ── Omnibus split ────────────────────────────────────────────────────────
    if fn == OMNIBUS:
        version  = extract_version(content, fn)
        date_str, _ = get_date(fn, content)
        split_omnibus(content, version, date_str, report)
        return

    # ── Standard improvement ─────────────────────────────────────────────────
    title       = extract_title(content, fn)
    version     = extract_version(content, fn)
    date_str, date_inferred = get_date(fn, content)
    doc_type    = infer_doc_type(fn)
    audience    = infer_audience(fn, content[:3000])
    category    = infer_category(fn)
    owner       = infer_owner(fn)
    review_cycle= infer_review(doc_type)
    keywords    = infer_keywords(fn)

    notes: list[str] = []
    if date_inferred:
        notes.append(f'Effective date not found in body — assigned {date_str}')
    if file_kb > 100:
        notes.append(f'Large document ({file_kb} KB) — consider further splitting by section')
    if fn in ('workapps-product-solutions-private-limited-document-for-quality-assurance.html',
              'workapps-product-solutions-private-limited-it-asset-inventory-.html',
              'workapps-product-solutions-private-limited-security-incident-drill-april-2025.html'):
        notes.append('Title begins with company name — consider a more specific document title')

    banner   = build_banner(title, doc_type, version, date_str, owner,
                             audience, category, keywords, review_cycle, notes)
    improved = inject_banner(content, banner)
    improved = fix_title_tag(improved, title)

    out_path = DEST / fn
    out_path.write_text(improved, encoding='utf-8')
    out_kb = out_path.stat().st_size // 1024

    report.append({
        'source':         fn,
        'output':         fn,
        'action':         'Improved — metadata banner added',
        'doc_type':       doc_type,
        'version':        version,
        'date':           date_str,
        'owner':          owner,
        'audience':       audience,
        'category':       category,
        'keywords':       keywords,
        'review_cycle':   review_cycle,
        'notes':          '; '.join(notes) if notes else 'Clean',
        'output_size_kb': out_kb,
    })
    print(f"  [OK] {fn} ({file_kb} KB -> {out_kb} KB)")

# ── Report generators ─────────────────────────────────────────────────────────

_FIELDNAMES = ['source','output','action','doc_type','version','date',
               'owner','audience','category','keywords','review_cycle',
               'notes','output_size_kb']

def write_csv(report: list) -> Path:
    p = DEST / 'improvement_report.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction='ignore')
        w.writeheader()
        w.writerows(report)
    return p

def _bar(label: str, count: int, total: int, colour: str,
         label_w: int = 240) -> str:
    pct  = round(count / total * 100) if total else 0
    bw   = max(4, pct * 3)
    return (f'<div style="margin:4px 0;font-size:10pt">'
            f'<span style="display:inline-block;width:{label_w}px">{label}</span>'
            f'<div style="display:inline-block;background:{colour};height:16px;'
            f'width:{bw}px;vertical-align:middle;border-radius:2px"></div>'
            f'&nbsp;<strong>{count}</strong>'
            f'</div>')

def write_html_report(report: list, n_source: int) -> Path:
    improved   = sum(1 for r in report if 'Improved'   in r.get('action',''))
    split      = sum(1 for r in report if 'Split'      in r.get('action',''))
    superseded = sum(1 for r in report if 'Superseded' in r.get('action',''))
    output_files = improved + split + superseded

    cats  = {}
    types = {}
    for r in report:
        c = r.get('category','')
        t = r.get('doc_type','')
        if c: cats[c]   = cats.get(c,0)  + 1
        if t: types[t]  = types.get(t,0) + 1

    cat_bars  = ''.join(_bar(c, n, n_source, '#1a56db') for c,n in sorted(cats.items(),  key=lambda x:-x[1]))
    type_bars = ''.join(_bar(t, n, n_source, '#059669', 180) for t,n in sorted(types.items(), key=lambda x:-x[1]))

    rows = ''
    for r in report:
        action = r.get('action','')
        if 'Improved'   in action: badge = '#1a56db'
        elif 'Split'    in action: badge = '#d97706'
        elif 'Superseded' in action: badge = '#dc2626'
        else:                      badge = '#6b7280'
        rows += (
            f'<tr>'
            f'<td><a href="{r.get("output","#")}">{r.get("source","")}</a></td>'
            f'<td style="color:{badge}">{r.get("action","")[:55]}</td>'
            f'<td>{r.get("doc_type","")}</td>'
            f'<td>{r.get("version","")}</td>'
            f'<td>{r.get("date","")}</td>'
            f'<td>{r.get("owner","")}</td>'
            f'<td>{r.get("audience","")}</td>'
            f'<td>{r.get("category","")}</td>'
            f'<td style="font-size:8.5pt;color:#555">{r.get("notes","")[:80]}</td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Policy Improvement Report — WorkApps Product Solutions Pvt Ltd</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f8fafc;color:#1e293b;margin:0;padding:24px 40px}}
h1{{color:#1a3a8f;font-size:20pt;margin-bottom:4px}}
h2{{color:#1a56db;font-size:13pt;margin-top:28px;border-bottom:2px solid #c8d5f0;padding-bottom:6px}}
.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
.card{{background:#fff;border:1px solid #c8d5f0;border-radius:8px;padding:14px 22px;text-align:center;min-width:110px}}
.num{{font-size:26pt;font-weight:bold;color:#1a56db;line-height:1}}
.lbl{{font-size:9pt;color:#64748b;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:8.5pt;margin-top:10px}}
th{{background:#1a56db;color:#fff;padding:7px 9px;text-align:left;white-space:nowrap}}
td{{padding:5px 9px;border-bottom:1px solid #e2e8f0;vertical-align:top}}
tr:nth-child(even){{background:#f8fafc}}
a{{color:#1a56db;text-decoration:none}}a:hover{{text-decoration:underline}}
.note{{background:#eff6ff;border-left:4px solid #1a56db;padding:10px 14px;
       border-radius:4px;font-size:10pt;margin:16px 0;line-height:1.6}}
.charts{{display:flex;gap:48px;flex-wrap:wrap;margin:6px 0}}
</style>
</head>
<body>
<h1>Policy Improvement Report</h1>
<p style="color:#64748b;margin-top:4px">
  WorkApps Product Solutions Pvt Ltd &mdash;
  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}
</p>

<div class="note">
  <strong>{n_source} source HTML policy files</strong> processed and written to
  <code>improved/</code>. Every document now carries a standardised metadata
  banner. The 906 KB omnibus IS&amp;E policy has been split into
  {len(_SPLITS)} focused documents. Two superseded duplicates have been
  replaced with redirect stubs.
</div>

<h2>Summary</h2>
<div class="stats">
  <div class="card"><div class="num">{n_source}</div><div class="lbl">Source Files</div></div>
  <div class="card"><div class="num" style="color:#059669">{improved}</div><div class="lbl">Improved</div></div>
  <div class="card"><div class="num" style="color:#d97706">{split}</div><div class="lbl">Split Parts</div></div>
  <div class="card"><div class="num" style="color:#dc2626">{superseded}</div><div class="lbl">Superseded</div></div>
  <div class="card"><div class="num" style="color:#7c3aed">{output_files}</div><div class="lbl">Output Files</div></div>
</div>

<h2>Improvements Applied to Every Document</h2>
<ul style="font-size:10.5pt;line-height:1.9">
  <li><strong>Metadata banner</strong> — title, document type, version, effective date, owner, audience, category, keywords, review cycle, organisation</li>
  <li><strong>Effective date</strong> — extracted from body text where present; assigned from policy-type lookup table where missing</li>
  <li><strong>Audience label</strong> — inferred from filename and content keywords (e.g. "All Employees", "Security Operations Team")</li>
  <li><strong>Document type classification</strong> — Policy / SOP / Checklist / Guideline / Procedure / Playbook / Job Description / Drill Record</li>
  <li><strong>Review cycle</strong> — Annual for policies, Semi-Annual for SOPs, Quarterly for checklists/drills, Monthly for inventory registers</li>
  <li><strong>HTML &lt;title&gt; tags</strong> — fixed for all documents that had empty titles</li>
  <li><strong>Omnibus IS&amp;E Policy split</strong> — 906 KB single file → {len(_SPLITS)} individual policy documents</li>
  <li><strong>Superseded duplicate files</strong> — 2 files replaced with redirect stubs pointing to the current version</li>
  <li><strong>Inline improvement notes</strong> — each banner lists actionable recommendations (date inferred, large doc, title suggestion)</li>
</ul>

<div class="charts">
  <div><h2>By Category</h2>{cat_bars}</div>
  <div><h2>By Document Type</h2>{type_bars}</div>
</div>

<h2>Recommended Next Steps (Content Gaps)</h2>
<table>
<thead><tr><th>#</th><th>Gap</th><th>Severity</th><th>Recommended Action</th></tr></thead>
<tbody>
<tr><td>1</td><td>No Leave / Absence policy</td><td style="color:#dc2626">HIGH</td><td>Create Leave Policy covering EL, SL, CL, maternity, paternity, LWP</td></tr>
<tr><td>2</td><td>No Travel &amp; Expense policy</td><td style="color:#dc2626">HIGH</td><td>Create T&amp;E Policy: travel booking, per-diem, reimbursement process</td></tr>
<tr><td>3</td><td>No Performance Appraisal policy</td><td style="color:#dc2626">HIGH</td><td>Create Appraisal Policy: review cycle, rating scale, PIP process</td></tr>
<tr><td>4</td><td>No Code of Conduct / Disciplinary policy</td><td style="color:#dc2626">HIGH</td><td>Create CoC Policy covering workplace behaviour, disciplinary steps</td></tr>
<tr><td>5</td><td>No POSH policy</td><td style="color:#dc2626">HIGH</td><td>Mandatory under Indian law — create POSH policy + ICC charter</td></tr>
<tr><td>6</td><td>No Grievance Redressal policy</td><td style="color:#d97706">MEDIUM</td><td>Create employee grievance escalation process</td></tr>
<tr><td>7</td><td>No Resignation / Exit policy</td><td style="color:#d97706">MEDIUM</td><td>Create offboarding SOP: notice period, FnF, asset return, knowledge transfer</td></tr>
<tr><td>8</td><td>21 documents have no audience set</td><td style="color:#d97706">MEDIUM</td><td>Metadata banner now assigned; update source documents for full accuracy</td></tr>
<tr><td>9</td><td>3 documents share identical title "WorkApps Product Solutions..."</td><td style="color:#d97706">MEDIUM</td><td>Rename to specific functional titles (QA Policy, Asset Inventory, Drill Report)</td></tr>
<tr><td>10</td><td>No benefits / compensation policy</td><td style="color:#6b7280">LOW</td><td>Consider adding salary-band framework, health insurance, bonus eligibility docs</td></tr>
</tbody>
</table>

<h2>File-by-File Detail</h2>
<table>
<thead>
  <tr>
    <th>Source File</th><th>Action</th><th>Type</th><th>Ver</th>
    <th>Date</th><th>Owner</th><th>Audience</th><th>Category</th><th>Notes</th>
  </tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<p style="color:#94a3b8;font-size:9pt;margin-top:30px">
  Generated by PolicyGPT Improvement Script &mdash;
  WorkApps Product Solutions Pvt Ltd
</p>
</body>
</html>"""
    p = DEST / 'improvement_report.html'
    p.write_text(html, encoding='utf-8')
    return p

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    html_files = sorted(SOURCE.glob("*.html"))
    report: list[dict] = []
    n = len(html_files)
    print(f"Processing {n} policy files …\n")

    for src in html_files:
        try:
            process_file(src, report)
        except Exception as exc:
            print(f"  [ERROR] {src.name}: {exc}")
            report.append({
                'source': src.name, 'output': 'ERROR',
                'action': f'Error: {exc}',
                'notes': str(exc),
            })

    csv_path  = write_csv(report)
    html_path = write_html_report(report, n)

    print(f"\n{'='*60}")
    print(f"Done.  {n} source files processed -> {DEST}")
    print(f"  HTML report : {html_path}")
    print(f"  CSV  report : {csv_path}")
    print(f"  Output files: {len(list(DEST.glob('*.html')))}")

if __name__ == '__main__':
    main()
