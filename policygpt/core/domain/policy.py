"""Domain profile — enterprise employee policies."""

from policygpt.core.domain.base import DomainProfile, register

PROFILE = DomainProfile(
    domain_context=(
        "Documents are enterprise employee policy documents covering HR, IT, finance, "
        "and other corporate functions. Users are employees, managers, and staff at "
        "all levels asking about leave entitlements, attendance, code of conduct, "
        "travel and expense reimbursement, IT acceptable use, data security, "
        "procurement, benefits, grievance procedures, and other workplace policies."
    ),

    # ── Bot ───────────────────────────────────────────────────────────────────
    persona_description="an enterprise company's employees",
    greeting_reply="Hello! Ready to help with any HR, IT, finance, or workplace policy questions.",
    identity_reply=(
        "I'm an enterprise policy assistant — ask me about HR policies, IT policies, "
        "finance procedures, workplace rules, and more."
    ),
    intent_user_description="enterprise employees",
    intent_policy_description=(
        "HR policies, IT policies, finance procedures, workplace rules, benefits, "
        "eligibility, approvals, timelines, or any other policy topic"
    ),
    doc_type_label="policy documents",

    # ── Corpus: document summaries ────────────────────────────────────────────
    doc_summary_focus=(
        "purpose, scope, applicable roles, key rules and procedures, "
        "thresholds and amounts, timelines, exceptions, and definitions"
    ),
    chunk_summary_capture=(
        "policy names, applicable roles, key rules, thresholds, "
        "amounts, timelines, exceptions, and definitions"
    ),
    combine_summary_retain=(
        "policy names, applicable roles, key rules, amounts, thresholds, "
        "timelines, exceptions, and key definitions"
    ),
    finalize_summary_focus=(
        "applicable roles, policy scope, key rules, thresholds, "
        "timelines, exceptions, and definitions"
    ),

    # ── Corpus: section summaries ─────────────────────────────────────────────
    section_summary_capture=(
        "applicable roles, key rules and procedures, thresholds with EXACT amounts, "
        "timelines with EXACT dates or periods, exceptions, "
        "definitions, and rules specific to this section"
    ),
    section_combine_preserve=(
        "applicable roles, key rules, thresholds, amounts, "
        "timelines, exceptions, and definitions"
    ),
    user_label="employee",

    # ── Corpus: FAQ ───────────────────────────────────────────────────────────
    faq_cover=(
        "eligibility criteria, key thresholds, timelines, procedures, "
        "exceptions, role definitions, approval requirements, and policy scope"
    ),

    # ── Bot: aggregate queries ────────────────────────────────────────────────
    aggregate_response_hint=(
        "Prefer standalone policy names from document titles or sections that explicitly describe the policy. "
        "Do not treat incidental mentions or cross-references as main policy items. "
        "Do not output bare names without context."
    ),
    aggregate_positive_markers=(
        "policy name",
        "policy scope",
        "policy overview",
        "policy purpose",
        "who it applies to",
        "applicable to",
        "policy is titled",
        "the policy is named",
        "policy identity",
        "policy summary",
    ),

    # ── Entity extraction ─────────────────────────────────────────────────────
    entity_categories=frozenset({
        "role",         # employee, manager, HR business partner, department head
        "location",     # office, region, country — any named place
        "time_period",  # Q1, FY2025, notice period, probation window
        "action",       # apply, approve, submit, request, report
        "benefit",      # annual leave, medical coverage, bonus, allowance
        "threshold",    # 85%, 30 days, ≥2 years service — numeric criteria
        "process",      # onboarding, appraisal, grievance, travel expense claim
        "policy",       # Code of Conduct, IT Acceptable Use, Travel Policy
        "abbreviation", # HR, IT, PIP, HRBP, EAP, SLA, T&C
        "other",
    }),
    entity_global_categories=frozenset({"role", "abbreviation", "policy", "time_period"}),
    entity_extraction_rules=(
        "- Extract ALL meaningful entities: roles, locations, time periods, actions, "
        "benefits, thresholds, processes, policy names, and abbreviations.\n"
        "- For roles: describe the role's level and responsibilities in this policy context.\n"
        "- For locations: describe the office, region, or country and its relevance in the policy.\n"
        "- For benefits: describe the benefit type (leave, allowance, coverage) and its value if stated.\n"
        "- For thresholds: describe what the number means (eligibility criterion, "
        "minimum service period, approval limit) and who it applies to.\n"
        "- For time periods: describe the window, what it covers, and any deadlines within it.\n"
        "- For abbreviations: spell them out fully in context.\n"
        "- For policy names: describe the policy's purpose and who it applies to.\n"
        "- Synonyms must reflect how a non-expert employee would phrase a question "
        "(natural, conversational English — the way someone types in a chat).\n"
        "- Be exhaustive: include every entity a user in this domain might ask about."
    ),
    entity_examples=(
        '[\n'
        '  {"name":"Annual Leave","category":"benefit","context":"Paid time off entitlement '
        'granted to full-time employees each calendar year","synonyms":["yearly leave","paid leave",'
        '"vacation days","AL"]},\n'
        '  {"name":"PIP","category":"abbreviation","context":"Performance Improvement Plan — '
        'a formal process initiated by HR when an employee\'s performance falls below expectations, '
        'outlining required improvements and timelines","synonyms":["performance plan","improvement plan",'
        '"performance review plan"]}\n'
        ']'
    ),

    # ── Web UI ────────────────────────────────────────────────────────────────
    ui_assistant_label="Enterprise Policy Assistant",
    ui_eyebrow="Ask policy questions",
    ui_description=(
        "Get instant answers on HR, IT, finance, and workplace policies — "
        "grounded in your indexed policy documents."
    ),
    ui_prompt_chips=(
        ("Leave entitlements", "What types of leave are available and what is the entitlement for each?"),
        ("Travel policy", "What is the travel policy — booking process, eligibility, and limits?"),
        ("Expense reimbursement", "How does the expense reimbursement process work and what can be claimed?"),
        ("Approval matrix", "What is the approval matrix — who approves what and up to which limit?"),
    ),
)

register("policy", PROFILE)
