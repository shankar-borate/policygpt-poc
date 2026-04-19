import re


STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "am",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "follow",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "so",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


GENERIC_POLICY_TERMS: set[str] = {
    "approval",
    "approvals",
    "checklist",
    "checklists",
    "clause",
    "clauses",
    "company",
    "companies",
    "compliance",
    "doc",
    "document",
    "documents",
    "employee",
    "employees",
    "faq",
    "final",
    "form",
    "forms",
    "general",
    "guidance",
    "guideline",
    "guidelines",
    "handbook",
    "major",
    "manual",
    "matrix",
    "page",
    "pages",
    "policy",
    "policies",
    "procedure",
    "procedures",
    "process",
    "processes",
    "purpose",
    "requirement",
    "requirements",
    "role",
    "roles",
    "rule",
    "rules",
    "scope",
    "section",
    "sections",
    "standard",
    "standards",
    "template",
    "overview",
    "introduction",
    "definition",
    "definitions",
    "owner",
    "owners",
    "responsibility",
    "responsibilities",
    "applicability",
    "version",
    "versions",
    "workflow",
}


DOMAIN_TOPIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "exit": (
        "exit",
        "offboarding",
        "off boarding",
        "separation",
        "resignation",
        "resign",
        "leaving company",
        "last day",
        "clearance",
        "full and final",
        "fnf",
        "handover",
        "relieving",
    ),
    "internship": (
        "internship",
        "intern",
        "trainee",
        "summer intern",
        "graduate trainee",
    ),
    "reimbursement": (
        "reimbursement",
        "reimburse",
        "expense claim",
        "claim",
        "expense",
        "repayment",
    ),
    "approval": (
        "approval",
        "approve",
        "approver",
        "sanction",
        "authority",
        "delegation",
        "matrix",
    ),
    "car_scheme": (
        "car scheme",
        "company car",
        "vehicle policy",
        "lease car",
        "car lease",
        "vehicle reimbursement",
    ),
    "leave": (
        "leave",
        "time off",
        "vacation",
        "annual leave",
        "sick leave",
        "pto",
        "absence",
    ),
    "travel": (
        "travel",
        "trip",
        "domestic travel",
        "international travel",
        "journey",
        "tour",
    ),
    "security": (
        "information security",
        "infosec",
        "cyber security",
        "security",
        "data protection",
        "password",
        "access control",
    ),
    "payroll": (
        "payroll",
        "salary",
        "compensation",
        "wages",
        "pay slip",
    ),
    "benefits": (
        "benefits",
        "insurance",
        "medical",
        "wellness",
        "allowance",
    ),
    "attendance": (
        "attendance",
        "working hours",
        "late coming",
        "shift",
        "timesheet",
    ),
    "remote_work": (
        "remote work",
        "work from home",
        "hybrid",
        "wfh",
        "telework",
    ),
}


INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "aggregate": (
        "list all", "all types", "types of", "what are all", "give me all",
        "all available", "how many types", "every type", "what all",
        "complete list", "all the types", "all policies", "all leaves",
        "all benefits", "all options",
        # Contest / scheme listing (date-sensitive — needs current date context)
        "all contests", "active contests", "ongoing contests", "current contests",
        "list contests", "show contests", "which contests", "what contests",
        "all schemes", "active schemes", "current schemes",
        "all rewards", "active rewards", "current rewards",
        "show me all", "tell me all",
    ),
    "checklist": ("checklist", "what should i do", "what do i need to do", "list of steps"),
    "process": (
        "process", "procedure", "how do i", "how do", "steps", "workflow", "formalities",
        "how to", "how does", "what happens", "what is the process", "what are the steps",
        "how should i",
    ),
    "eligibility": (
        "eligibility", "eligible", "who can", "criteria", "qualification",
        "am i eligible", "can i apply", "do i qualify", "who is eligible",
        "who qualifies", "can i get", "am i entitled", "can i avail",
    ),
    "approval": (
        "who approves", "approval", "approver", "who can approve", "sanction",
        "need approval", "requires approval", "authorized by", "who signs",
        "who needs to approve",
    ),
    "timeline": (
        "when", "last day", "timeline", "how long", "notice period", "by when",
        "deadline", "how many days", "within how many", "time limit", "duration",
        "how soon",
    ),
    "documents_required": (
        "documents", "document", "forms", "form", "submit", "paperwork",
        "what do i need to submit", "what documents", "supporting documents",
    ),
    "contact": ("who should i contact", "contact", "reach out", "whom should", "who do i contact"),
    "comparison": ("difference", "compare", "versus", "vs", "better", "different", "which is better"),
    "scope": ("scope", "applies to", "covered", "who is covered", "applicable to", "does it apply"),
    "exceptions": ("exception", "exceptions", "out of policy", "special case", "exemption", "exempt"),
}


DOCUMENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "policy": ("policy",),
    "process": ("process", "procedure", "workflow"),
    "checklist": ("checklist",),
    "matrix": ("matrix", "delegation"),
    "guideline": ("guideline", "guide"),
    "faq": ("faq", "frequently asked questions"),
    "form": ("form", "template"),
}


SECTION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "eligibility": ("eligibility", "eligible", "criteria", "who can", "qualification"),
    "approval": ("approval", "approver", "authority", "delegation", "sanction"),
    "process": ("process", "procedure", "steps", "workflow"),
    "checklist": ("checklist", "required actions", "to do"),
    "timeline": ("timeline", "when", "notice period", "effective date"),
    "documents_required": ("documents", "forms", "submit", "paperwork"),
    "scope": ("scope", "applies to", "covered"),
    "exceptions": ("exception", "exceptions", "out of policy"),
    "definitions": ("definition", "definitions", "meaning"),
    "contact": ("contact", "helpdesk", "reach out"),
    "responsibilities": ("responsibilities", "role of", "owner"),
}


AUDIENCE_KEYWORDS: tuple[str, ...] = (
    "employee",
    "employees",
    "intern",
    "interns",
    "manager",
    "managers",
    "contractor",
    "contractors",
    "consultant",
    "consultants",
    "vendor",
    "vendors",
)


def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    return compact.casefold()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def keywordize_text(text: str) -> list[str]:
    tokens = [
        token
        for token in tokenize_text(text)
        if (len(token) >= 2 or token.isdigit()) and token not in STOPWORDS
    ]
    if len(tokens) < 2:
        return tokens

    terms = list(tokens)
    for left, right in zip(tokens, tokens[1:]):
        terms.append(f"{left}_{right}")
    return terms


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def humanize_term(term: str) -> str:
    return term.replace("_", " ").strip()


def is_informative_term(term: str) -> bool:
    normalized = term.strip().casefold()
    if not normalized:
        return False

    parts = [part for part in normalized.split("_") if part]
    if not parts:
        return False
    if any(len(part) < 2 and not part.isdigit() for part in parts):
        return False
    if all(part in STOPWORDS for part in parts):
        return False

    meaningful_parts = [part for part in parts if part not in STOPWORDS]
    if not meaningful_parts:
        return False

    if len(parts) == 1:
        if normalized.isdigit():
            return False
        return normalized not in GENERIC_POLICY_TERMS

    return any(part not in GENERIC_POLICY_TERMS for part in meaningful_parts)


def detect_matching_labels(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = normalize_text(text)
    text_tokens = set(tokenize_text(text))
    matches: list[str] = []
    for label, phrases in mapping.items():
        if any(phrase in normalized for phrase in phrases):
            matches.append(label)
            continue

        for phrase in phrases:
            phrase_tokens = [
                token
                for token in tokenize_text(phrase)
                if token not in STOPWORDS and len(token) >= 3
            ]
            if not phrase_tokens:
                continue

            overlap = len(set(phrase_tokens).intersection(text_tokens))
            required_overlap = 1 if len(phrase_tokens) == 1 else min(2, len(phrase_tokens))
            if overlap >= required_overlap:
                matches.append(label)
                break
    return unique_preserving_order(matches)
