"""PolicyRewriter — HTML policy pre-processor for PolicyGPT ingestion.

Sits as the FIRST step in the ingestion pipeline (before extraction).
For every HTML document it:
  1. Checks the cache — skips if improved file is already newer than source.
  2. Parses the HTML and adds structure improvements:
       - Classification banner
       - Metadata block  (title, type, version, date, owner, audience,
                          category, keywords, review cycle, organisation)
       - Table of Contents  (built from existing h1/h2/h3)
       - Overview paragraph (extracted from Purpose / Scope sections)
       - Roles & Responsibilities table (if present in content)
       - Inline regulatory tags  (ISO 27001 / RBI / PCI-DSS)
       - Related Policies list
  3. Saves the improved HTML to  {debug_log_dir}/improved/{filename}
  4. Returns the improved file path so the pipeline can re-point
     IngestMessage.source_path to the improved version.

Design principles
-----------------
- NEVER rewrites or paraphrases original policy text.
- Only ADDS structure and metadata around the verbatim content.
- All additions wrapped in yellow  (background:#fff9c4)  so they are
  visually distinct and search tools hit clean original text.
- Falls back to the original path silently if anything fails, so a
  rewriter bug never blocks ingestion.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── CSS injected once per document ───────────────────────────────────────────

_GLOBAL_CSS = """<style id="wa-rewriter-styles">
.wa-classification{font-family:Arial,sans-serif;font-size:10pt;font-weight:bold;
  padding:9px 16px;border-radius:4px;margin-bottom:18px}
.wa-meta{font-family:Arial,sans-serif;background:#f0f4ff;border-left:5px solid #1a56db;
  border-radius:4px;padding:14px 18px;margin-bottom:24px;font-size:10pt}
.wa-meta table{border-collapse:collapse;width:100%;font-size:10pt;margin-top:6px}
.wa-meta th{background:#1a56db;color:#fff;padding:6px 10px;text-align:left;font-size:9.5pt}
.wa-meta td{padding:5px 10px;border-bottom:1px solid #c8d5f0;vertical-align:top}
.wa-toc{background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;
  padding:12px 18px;margin-bottom:24px;font-size:10pt}
.wa-toc h3{color:#1a3a8f;margin-top:0;font-size:11pt;font-family:Arial,sans-serif}
.wa-toc ol{margin:4px 0 0 18px;padding:0}
.wa-toc li{margin-bottom:3px}
.wa-toc a{color:#1a56db;text-decoration:none}
.wa-overview{background:#f0f4ff;border-left:4px solid #1a56db;border-radius:3px;
  padding:10px 14px;margin-bottom:20px;font-size:10.5pt;font-family:Arial,sans-serif;
  line-height:1.7}
.wa-roles table{border-collapse:collapse;width:100%;font-size:10pt;margin-top:8px}
.wa-roles th{background:#1a3a8f;color:#fff;padding:7px 10px;text-align:left}
.wa-roles td{padding:6px 10px;border-bottom:1px solid #e2e8f0;vertical-align:top}
.wa-roles tr:nth-child(even) td{background:#f8fafc}
.wa-related ul{margin:6px 0 0 20px;padding:0;font-size:10.5pt}
.wa-related li{margin-bottom:4px}
.wa-reg-ref{color:#9ca3af;font-size:9pt;margin-left:5px}
.wa-add{background:#fff9c4;padding:0 2px;border-radius:2px}
</style>"""

# ── Metadata inference (same logic as improve_policies.py) ───────────────────

def _infer_doc_type(fn: str) -> str:
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

def _infer_audience(fn: str) -> str:
    f = fn.lower()
    if any(x in f for x in ['clear-desk','remote-working','wfh','modern-slavery',
                              'anti-trust','ethical-sourcing','disease-response',
                              'employee-onboarding','information-security-and-employe']):
        return 'All Employees'
    if any(x in f for x in ['recruitment','onboarding','background']):
        return 'HR Team, Hiring Managers'
    if 'ciso-jd' in f:
        return 'Leadership, HR'
    if any(x in f for x in ['coding','sdl','system-acquisition','garbage']):
        return 'Development Team'
    if any(x in f for x in ['aws','cloud','server','infra','sso',
                              'url-whitelist','obtain-or-review']):
        return 'DevOps / Cloud Engineers'
    if any(x in f for x in ['siem','log-monitor','monthly-log','ir-playbook',
                              'incident','threat-response','network-vuln',
                              'vulnerability-assessment']):
        return 'Security Operations Team'
    if any(x in f for x in ['asset','inventory','license']):
        return 'IT Asset Managers, IT Team'
    if any(x in f for x in ['marketing','selling','anti-trust']):
        return 'Sales, Marketing, Legal'
    if any(x in f for x in ['outsourc','sar','customer-data']):
        return 'Legal, Compliance Team'
    if any(x in f for x in ['quality','qa']):
        return 'QA Team, Project Managers'
    return 'IT Team, Management'

def _infer_category(fn: str) -> str:
    f = fn.lower()
    if any(x in f for x in ['anti-trust','compliance-program','outsourc',
                              'sar','customer-data','modern-slavery','ethical']):
        return 'Legal & Compliance'
    if any(x in f for x in ['onboarding','recruitment','wfh','remote-working',
                              'clear-desk','background','disease-response']):
        return 'HR & Employee Policies'
    if any(x in f for x in ['asset','inventory','license']):
        return 'IT Asset Management'
    if any(x in f for x in ['siem','log','monitoring','ir-playbook','incident']):
        return 'Security Operations'
    if any(x in f for x in ['aws','cloud','server','infra','sso','url-whitelist']):
        return 'Cloud & Infrastructure'
    if any(x in f for x in ['coding','sdl','system-acquisition','garbage']):
        return 'Software Development'
    if any(x in f for x in ['rto','rpo','continuity','resiliency','trrp','contingency']):
        return 'Business Continuity'
    if any(x in f for x in ['quality','qa']):
        return 'Quality Assurance'
    if any(x in f for x in ['intellectual','ipr']):
        return 'Intellectual Property'
    return 'Information Security'

def _infer_owner(fn: str) -> str:
    f = fn.lower()
    if 'ciso' in f:                                          return 'CISO'
    if any(x in f for x in ['recruitment','onboarding','background']): return 'HR Manager'
    if any(x in f for x in ['asset','inventory','license']): return 'IT Asset Manager / CISO'
    if any(x in f for x in ['cloud','aws','server','infra','rto','trrp']): return 'CTO / DevOps Lead'
    if any(x in f for x in ['coding','sdl','system-acquisition']): return 'Engineering Lead'
    if any(x in f for x in ['quality']): return 'QA Manager'
    if any(x in f for x in ['marketing','selling','anti-trust','modern-slavery',
                              'ethical','sar','customer-data','outsourc']): return 'Legal / Compliance Officer'
    if any(x in f for x in ['siem','log','monitoring','ir-playbook']): return 'Security Operations Manager'
    return 'CISO / CTO'

def _infer_review(doc_type: str) -> str:
    if doc_type in ('Checklist', 'Drill Record'):  return 'Quarterly'
    if doc_type == 'Inventory Register':           return 'Monthly'
    if doc_type in ('SOP', 'Procedure'):           return 'Semi-Annual'
    return 'Annual'

def _infer_keywords(fn: str) -> str:
    stop = {'workapps','product','solutions','private','limited','policy','sop',
            'procedure','management','process','videocx','platform','the','and',
            'for','with','from','this','standard','operating','document',
            'guidelines','checklist','version'}
    words = re.sub(r'[^a-z0-9 ]+', ' ', fn.replace('.html','').lower()).split()
    return ', '.join(w for w in words if len(w) > 3 and w not in stop)[:120]

# Files larger than this are skipped by _inject_reg_tags (size guard for
# catastrophic-backtracking safety — the regex is already fixed but belt+braces)
_MAX_REG_INJECT_BYTES = 400_000

_DATE_FALLBACKS: dict[str, str] = {
    'anti-trust': '15/Jan/2024', 'asset-lifecycle': '01/Apr/2024',
    'asset-reconcil': '01/Apr/2024', 'aws-security': '01/Jun/2024',
    'checklist---security-threat': '15/Mar/2024', 'clear-desk': '01/Jan/2025',
    'compliance-program-for-marketing': '10/Feb/2024',
    'custom-siem': '01/Aug/2024', 'disease-response': '01/May/2024',
    'employee-onboarding': '01/Mar/2024', 'ethical-sourcing': '15/Jan/2024',
    'garbage-collection': '01/Sep/2024',
    'guidelines-and-process-for-management': '01/Feb/2024',
    'intellectual-property': '01/Mar/2024', 'internal-background': '01/Mar/2024',
    'ir-playbook': '01/Jul/2024', 'modern-slavery': '15/Jan/2024',
    'monthly-log-review': '31/Oct/2024', 'obtain-or-review': '01/Jun/2024',
    'organization-safeguards': '01/Apr/2024', 'outsourcing-risk': '01/Feb/2024',
    'procedures-for-encryption': '01/May/2024', 'remote-working': '01/Jan/2025',
    'rto-rpo': '01/Jun/2024', 'secure-coding': '01/Jul/2024',
    'security-development': '01/Jul/2024', 'server-hardening-policy': '01/May/2024',
    'sop-iam-user': '01/Jun/2024', 'sso-integration': '01/Aug/2024',
    'standard-operating-procedure-sop-intell': '01/Mar/2024',
    'system-acquisition': '01/Jul/2024', 'technology-resiliency': '01/Jun/2024',
    'url-whitelisting': '01/Aug/2024', 'user-lifecyle': '01/Mar/2024',
    'videocx.io-platform': '01/Jun/2024', 'vulnerability-assessment': '15/Mar/2024',
    'web-application-security': '01/Apr/2024', 'work-from-home': '01/Jan/2025',
    'workapps---network-vulnerability': '15/Mar/2024',
    'workapps---procedure-for-responding': '01/Feb/2024',
    'workapps---sar-reports': '01/Feb/2024',
    'workapps---server-hardening': '01/May/2024',
    'workapps--security-hardening': '01/May/2024',
    'workapps-ciso-jd': '01/Jan/2025', 'workapps-cloud-security': '01/Jun/2024',
    'workapps-dlp': '15/Apr/2024', 'workapps-information-security': '01/Jan/2025',
    'workapps-it-asset-software-license': '01/Apr/2024',
    'workapps-product-solutions-private-limited-document-for-quality': '01/Sep/2024',
    'workapps-product-solutions-private-limited-it-asset-inventory': '01/Apr/2024',
    'workapps-product-solutions-private-limited-security-incident-drill': '30/Apr/2025',
    'workapps-tech-recruitments': '01/Mar/2024',
}

def _get_date(fn: str, text: str) -> str:
    m = re.search(r'Date[:]\s*(\d{1,2}[/]\w{3,}[/]\d{4})', text)
    if m: return m.group(1)
    m = re.search(r'(\d{1,2}[/]\w{3}[/]\d{4})', text)
    if m: return m.group(1)
    fn_l = fn.lower()
    for key, val in _DATE_FALLBACKS.items():
        if key in fn_l:
            return val
    return '01/Jan/2024'

def _get_version(fn: str, text: str) -> str:
    m = re.search(r'[Vv]ersion\s+(\d+[\d.]*)', text)
    if m: return f"v{m.group(1)}"
    m = re.search(r'v[.\s]*(\d+[\d.]+)', fn, re.I)
    if m: return f"v{m.group(1)}"
    return 'v1.0'

# ── Regex-based regulatory tag injection ─────────────────────────────────────

_REG_PATTERNS: list[tuple[str, str]] = [
    (r'\b(?:VPN|virtual private network)\b',          'ISO 27001 A.6.7 / RBI Cyber Security Framework 2016'),
    (r'\b(?:password|passphrase|credential)\b',       'ISO 27001 A.8.5 / PCI-DSS v4.0 Req 8'),
    (r'\b(?:encrypt|encryption)\b',                   'ISO 27001 A.8.24 / PCI-DSS v4.0 Req 3'),
    (r'\b(?:access control|role.based access|RBAC)\b','ISO 27001 A.8.2 / RBI IT Governance 2023'),
    (r'\b(?:incident|security breach|security event)\b', 'ISO 27001 A.5.24 / CERT-In Directions 2022'),
    (r'\b(?:log|audit trail|audit log)\b',            'ISO 27001 A.8.15 / RBI Cyber Security 2016'),
    (r'\b(?:backup|data recovery|restore)\b',         'ISO 27001 A.8.13 / RBI BCP Guidelines'),
    (r'\b(?:patch|vulnerability|CVE)\b',              'ISO 27001 A.8.8 / CERT-In Directions 2022'),
    (r'\b(?:personal data|PII|DPDP|data privacy)\b',  'DPDP Act 2023 / ISO 27001 A.5.34'),
    (r'\b(?:third.party|vendor|outsourc)\b',          'RBI IT Outsourcing 2023 / IRDAI Outsourcing 2017'),
    (r'\b(?:cloud|AWS|Azure|GCP|SaaS)\b',             'ISO 27001 A.5.23 / RBI Cloud Framework'),
    (r'\b(?:firewall|IDS|IPS|intrusion)\b',           'ISO 27001 A.8.20 / RBI Cyber Security 2016'),
    (r'\b(?:MFA|multi.factor|two.factor|2FA)\b',      'ISO 27001 A.8.5 / PCI-DSS v4.0 Req 8.4'),
    (r'\b(?:DLP|data loss|data leakage)\b',           'ISO 27001 A.8.12 / CERT-In Directions 2022'),
]

def _clean(html: str) -> str:
    return re.sub(r'<[^>]+>', '', html).replace('&amp;', '&').replace('&#xa0;', ' ').strip()

# ── Main rewriter class ───────────────────────────────────────────────────────

class PolicyRewriter:
    """Pre-processes HTML policy files for optimised PolicyGPT ingestion.

    Parameters
    ----------
    output_dir :
        Directory where improved HTML files are written.
        Typically  {debug_log_dir}/improved/
        Ignored when save_to_disk=False.
    save_to_disk :
        When True (default), the improved HTML is written to output_dir and
        subsequent ingestion runs use the cached file (skip if newer).
        When False, the rewrite runs in-memory only — nothing is written to
        disk and source_path on IngestMessage is not changed.
    skip_if_cached :
        When save_to_disk=True, skip rewriting if the improved file already
        exists and is newer than the source file.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        save_to_disk: bool = True,
        skip_if_cached: bool = True,
    ) -> None:
        self._save_to_disk    = save_to_disk
        self._skip_if_cached  = skip_if_cached
        if save_to_disk and output_dir:
            self._out = Path(output_dir)
            self._out.mkdir(parents=True, exist_ok=True)
        else:
            self._out = None

    # ── Public API ────────────────────────────────────────────────────────────

    def rewrite(self, source_path: str) -> tuple[str, str]:
        """Rewrite the HTML at source_path.

        Returns
        -------
        (improved_path, improved_content)
            improved_path    — path to improved file when save_to_disk=True,
                               otherwise the original source_path unchanged.
            improved_content — the rewritten HTML string (always populated
                               so the pipeline can use it in-memory even when
                               save_to_disk=False).
            On any error both values fall back to the originals silently.
        """
        src = Path(source_path)
        if src.suffix.lower() not in {'.html', '.htm'}:
            original = src.read_text(encoding='utf-8', errors='ignore')
            return source_path, original

        try:
            original = src.read_text(encoding='utf-8', errors='ignore')

            # ── Cache hit (save_to_disk mode only) ───────────────────────────
            if self._save_to_disk and self._out:
                out_path = self._out / src.name
                if self._skip_if_cached and out_path.exists():
                    if out_path.stat().st_mtime >= src.stat().st_mtime:
                        logger.debug("PolicyRewriter: cache hit %s", src.name)
                        return str(out_path), out_path.read_text(encoding='utf-8', errors='ignore')

            improved = self._improve(original, src.name)

            # ── Persist to disk ───────────────────────────────────────────────
            if self._save_to_disk and self._out:
                out_path = self._out / src.name
                out_path.write_text(improved, encoding='utf-8')
                logger.info("PolicyRewriter: saved %s", out_path)
                return str(out_path), improved

            # ── In-memory only ────────────────────────────────────────────────
            logger.debug("PolicyRewriter: in-memory rewrite %s", src.name)
            return source_path, improved

        except Exception as exc:
            logger.warning("PolicyRewriter: failed for %s — %s", src.name, exc, exc_info=True)
            original_content = src.read_text(encoding='utf-8', errors='ignore')
            return source_path, original_content

    # ── Core improvement logic ────────────────────────────────────────────────

    def _improve(self, html: str, filename: str) -> str:
        text = _clean(html)

        # ── Derive metadata ──────────────────────────────────────────────────
        fn         = filename
        title      = self._extract_title(html, fn)
        version    = _get_version(fn, text)
        date_str   = _get_date(fn, text)
        doc_type   = _infer_doc_type(fn)
        audience   = _infer_audience(fn)
        category   = _infer_category(fn)
        owner      = _infer_owner(fn)
        review     = _infer_review(doc_type)
        keywords   = _infer_keywords(fn)
        classif, cls_color, cls_border = self._infer_classification(fn, text)

        # ── Build add-on blocks ──────────────────────────────────────────────
        classification_banner = self._build_classification_banner(
            classif, cls_color, cls_border)
        meta_block  = self._build_meta_block(
            title, doc_type, version, date_str, owner,
            audience, category, keywords, review)
        toc         = self._build_toc(html)
        overview    = self._build_overview(html, title)
        roles_block = self._build_roles_block(html)

        # ── Inject CSS + blocks after <body> ─────────────────────────────────
        body_m = re.search(r'(<body[^>]*>)', html, re.I)
        if body_m:
            insert_pos = body_m.end()
            prefix = (
                '\n' + _GLOBAL_CSS + '\n'
                + classification_banner + '\n'
                + meta_block + '\n'
                + toc + '\n'
                + overview + '\n'
            )
            html = html[:insert_pos] + prefix + html[insert_pos:]
        else:
            html = _GLOBAL_CSS + '\n' + classification_banner + '\n' + meta_block + '\n' + overview + '\n' + html

        # ── Append roles block before </body> ────────────────────────────────
        # Related policies are NOT added to body — listing other policy names
        # as body text would pollute BM25/vector scores for those documents.
        close_body = re.search(r'</body>', html, re.I)
        if close_body:
            pos = close_body.start()
            html = html[:pos] + '\n' + roles_block + '\n' + html[pos:]
        else:
            html += '\n' + roles_block

        # ── Fix empty <title> tag ────────────────────────────────────────────
        html = re.sub(
            r'<title>\s*</title>',
            f'<title>{title} — WorkApps Product Solutions Pvt Ltd</title>',
            html, flags=re.I)

        # ── Add inline regulatory tags ───────────────────────────────────────
        html = self._inject_reg_tags(html)

        return html

    # ── Block builders ────────────────────────────────────────────────────────

    def _build_classification_banner(
        self, label: str, bg: str, border: str) -> str:
        return (
            f'<div class="wa-classification wa-add" '
            f'style="background:{bg};border-left:5px solid {border}">'
            f'{label} &nbsp;|&nbsp; '
            f'WorkApps Product Solutions Pvt Ltd &nbsp;|&nbsp; VideoCX.io'
            f'</div>'
        )

    def _build_meta_block(
        self, title: str, doc_type: str, version: str, date_str: str,
        owner: str, audience: str, category: str, keywords: str,
        review: str) -> str:
        return f"""<div class="wa-meta wa-add">
<table>
<tr><th>Document Title</th><td>{title}</td>
    <th>Document Type</th><td>{doc_type}</td></tr>
<tr><th>Version</th><td>{version}</td>
    <th>Effective Date</th><td>{date_str}</td></tr>
<tr><th>Next Review</th><td>Annual — {review}</td>
    <th>Document Owner</th><td>{owner}</td></tr>
<tr><th>Audience</th><td>{audience}</td>
    <th>Category</th><td>{category}</td></tr>
<tr><th>Keywords</th><td colspan="3" style="font-size:9.5pt">{keywords}</td></tr>
<tr><th>Company</th><td>WorkApps Product Solutions Pvt Ltd</td>
    <th>Platform</th><td>VideoCX.io</td></tr>
</table>
</div>"""

    def _build_toc(self, html: str) -> str:
        # re.S removed — headings in these policy files are always single-line;
        # dotall mode causes slow matching on large documents.
        headings = re.findall(
            r'<h([23])[^>]*>(.*?)</h\1>', html, re.I)
        if len(headings) < 3:
            return ''
        items = ''
        for level, raw in headings:
            text = _clean(raw).strip()
            if not text or len(text) < 3:
                continue
            anchor = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
            indent = ' style="margin-left:18px"' if level == '3' else ''
            items += f'<li{indent}><a href="#{anchor}">{text}</a></li>\n'
        if not items:
            return ''
        return (
            f'<div class="wa-toc wa-add">'
            f'<h3>Contents</h3><ol>{items}</ol></div>'
        )

    def _build_overview(self, html: str, title: str) -> str:
        """Extract text from Purpose / Scope / Introduction sections."""
        # Find first substantial paragraph after Purpose/Scope heading
        patterns = [
            r'(?:purpose|scope|introduction|overview)[^>]*</h[1-4]>\s*<p[^>]*>(.*?)</p>',
        ]
        for pat in patterns:
            m = re.search(pat, html, re.S | re.I)
            if m:
                text = _clean(m.group(1)).strip()[:400]
                if len(text) > 40:
                    return (
                        f'<div class="wa-overview wa-add">'
                        f'<strong>Overview:</strong> {text}'
                        f'<br><em style="font-size:9pt;color:#64748b">'
                        f'This summary is added for search navigation. '
                        f'Authoritative policy text follows below.</em>'
                        f'</div>'
                    )
        return ''

    def _build_roles_block(self, html: str) -> str:
        """Build Roles & Responsibilities table if not already present."""
        # Check if a roles/responsibilities section already exists
        if re.search(r'roles?\s*(?:and|&amp;)\s*responsibilit', html, re.I):
            return ''  # already has one — don't duplicate

        # Detect mentioned roles from the text and build a generic table
        text = _clean(html).lower()
        rows = []
        role_map = [
            ('ciso',           'CISO',             'Overall policy ownership; security enforcement; exception approvals'),
            ('cto',            'CTO',              'Technology infrastructure; tool provisioning; policy ratification'),
            ('hr manager',     'HR Manager',       'Employee communication; onboarding; disciplinary process'),
            ('employee',       'All Employees',    'Read, understand, and comply with this policy'),
            ('supervisor',     'Supervisor / Manager', 'Approve requests; monitor team compliance; escalate incidents'),
            ('it team',        'IT / DevOps Team', 'Implement technical controls; maintain systems per policy'),
            ('compliance',     'Compliance Manager','Audit compliance; report to leadership; manage exceptions'),
            ('internal audit', 'Internal Audit',   'Independent verification of policy adherence'),
        ]
        for keyword, role_label, duty in role_map:
            if keyword in text:
                rows.append(f'<tr><td>{role_label}</td><td>{duty}</td></tr>')
        if len(rows) < 2:
            return ''
        table_rows = '\n'.join(rows)
        return (
            f'<div class="wa-roles wa-add">'
            f'<h2 style="font-size:12pt;color:#1a3a8f;border-bottom:2px solid #c8d5f0;'
            f'padding-bottom:4px;font-family:Arial,sans-serif">'
            f'Roles and Responsibilities</h2>'
            f'<table><thead><tr>'
            f'<th>Role</th><th>Key Responsibility</th>'
            f'</tr></thead><tbody>{table_rows}</tbody></table>'
            f'</div>'
        )

    def _build_related_policies(self, fn: str, category: str) -> str:
        related_map: dict[str, list[str]] = {
            'Information Security': [
                'Cyber Security Policy',
                'Information Security & Employment Policy',
                'Data Loss Prevention (DLP) Policy',
                'User Lifecycle Policy',
                'Vulnerability Assessment Policy',
            ],
            'HR & Employee Policies': [
                'Employee Onboarding Policy',
                'Remote Working Policy',
                'Work From Home Setup & Security Checklist',
                'Clear Desk Audit Checklist',
                'Information Security & Employment Policy',
            ],
            'Cloud & Infrastructure': [
                'Server Hardening Policy',
                'AWS Security Configuration Guidelines',
                'Cloud Security Policy',
                'RTO / RPO Policy',
                'Technology Resiliency & Recovery Plan',
            ],
            'Security Operations': [
                'IT Incident Management Policy',
                'Log Monitoring & Management Policy',
                'Vulnerability Assessment Policy',
                'IR Playbook',
                'Custom SIEM Policy',
            ],
            'IT Asset Management': [
                'IT Asset Software License Management Policy',
                'IT Asset Inventory',
                'User Lifecycle Policy',
                'SSO Integration Policy',
            ],
            'Legal & Compliance': [
                'Anti-Trust and Anti-Competitive Practices Policy',
                'Modern Slavery & Human Trafficking Prevention Policy',
                'Ethical Sourcing Policy',
                'Customer Data Request SOP',
                'SAR Reports SOP',
            ],
            'Software Development': [
                'Secure Coding Practices',
                'Security Development Lifecycle (SDL)',
                'System Acquisition & Development Policy',
                'Garbage Collection Procedure',
            ],
            'Business Continuity': [
                'RTO / RPO Policy',
                'Technology Resiliency & Recovery Plan',
                'Disease Response & Continuity Plan',
                'VideoCX.io Contingency Plan',
            ],
        }
        policies = related_map.get(category, [])
        # Remove self-reference
        fn_clean = fn.replace('.html', '').replace('-', ' ').lower()
        policies = [p for p in policies if p.lower()[:20] not in fn_clean[:20]]
        if not policies:
            return ''
        items = '\n'.join(f'<li>{p}</li>' for p in policies)
        return (
            f'<div class="wa-related wa-add">'
            f'<h2 style="font-size:12pt;color:#1a3a8f;border-bottom:2px solid #c8d5f0;'
            f'padding-bottom:4px;font-family:Arial,sans-serif">'
            f'Related Policies</h2>'
            f'<ul>{items}</ul>'
            f'</div>'
        )

    def _inject_reg_tags(self, html: str) -> str:
        """Add lightweight regulatory ref tags after matching terms in <p>/<li> tags.

        re.S is intentionally NOT used — dotall mode causes catastrophic
        backtracking on large HTML files (906 KB+).  Policy HTML in this
        corpus has one <p>/<li> element per line, so line-mode is correct.
        Files that exceed _MAX_REG_INJECT_BYTES are skipped entirely as an
        additional safety net.
        """
        if len(html.encode('utf-8', errors='ignore')) > _MAX_REG_INJECT_BYTES:
            logger.debug(
                "PolicyRewriter: skipping reg-tag injection — file too large (%d bytes)",
                len(html),
            )
            return html

        def _replace_in_tag(m: re.Match) -> str:
            tag_html = m.group(0)
            # Only inject once per tag — check if tag already has a reg ref
            if 'wa-reg-ref' in tag_html:
                return tag_html
            for pattern, ref in _REG_PATTERNS:
                pm = re.search(pattern, tag_html, re.I)
                if pm:
                    ref_tag = (f'<span class="wa-reg-ref wa-add">'
                               f'[{ref}]</span>')
                    # Insert ref tag just before the closing tag
                    tag_html = re.sub(r'(</(?:p|li)>)', ref_tag + r'\1',
                                      tag_html, count=1, flags=re.I)
                    break  # one tag per element
            return tag_html

        # re.I only — no re.S.  Each <p>/<li> must open and close on the
        # same line; multi-line elements are silently skipped (acceptable).
        return re.sub(
            r'<(?:p|li)[^>]*>[^<]*(?:<(?!/?(?:p|li)\b)[^>]*>[^<]*)*</(?:p|li)>',
            _replace_in_tag,
            html,
            flags=re.I,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(html: str, filename: str) -> str:
        m = re.search(
            r'<span[^>]*(?:font-size:2[0-9]pt|color:#0070c0)[^>]*>(.*?)</span>',
            html, re.S | re.I)
        if m:
            t = _clean(m.group(1))
            if len(t) > 6: return t
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.S | re.I)
        if m:
            t = _clean(m.group(1))
            if len(t) > 3: return t
        stem = filename.replace('.html', '')
        stem = re.sub(r'-v[\d.]+$', '', stem)
        stem = re.sub(r'-workapps.*', '', stem)
        return stem.replace('-', ' ').title()

    @staticmethod
    def _infer_classification(
        fn: str, text: str
    ) -> tuple[str, str, str]:
        """Return (label, background_color, border_color)."""
        tl = (fn + text[:500]).lower()
        if any(x in tl for x in ['salary','compensation','personal','posh',
                                   'disciplinary','termination','strictly']):
            return ('STRICTLY CONFIDENTIAL', '#fce4ec', '#c62828')
        if any(x in tl for x in ['security','password','vpn','encryption',
                                   'incident','vulnerability','firewall']):
            return ('CONFIDENTIAL', '#fff9c4', '#f59e0b')
        if any(x in tl for x in ['public','customer shareable','external']):
            return ('CUSTOMER SHAREABLE', '#e8f5e9', '#2e7d32')
        return ('INTERNAL USE ONLY', '#e3f2fd', '#1565c0')
