"""Domain profile — reward-and-recognition contest policies."""

from policygpt.core.domain.base import DomainProfile, register

PROFILE = DomainProfile(
    domain_context=(
        "Documents are reward-and-recognition contest policies for an insurance "
        "company's agency sales channel. Users are sales agents (FCs, EIMs, ACHs, "
        "and other channel roles) asking about contest eligibility, qualification "
        "criteria, reward details, payout timelines, locations, role definitions, "
        "product thresholds (FYFP, persistency, etc.), and contest rules."
    ),

    # ── Bot ───────────────────────────────────────────────────────────────────
    persona_description="an insurance company's agency sales team",
    greeting_reply="Hello! Ready to help with any contest or policy questions.",
    identity_reply=(
        "I'm a policy assistant for the agency sales team — ask me about contests, "
        "rewards, eligibility, thresholds, timelines, and more."
    ),
    intent_user_description="insurance sales agents",
    intent_policy_description=(
        "contest rules, rewards, eligibility, thresholds, timelines, locations, roles, "
        "or any other policy topic"
    ),
    doc_type_label="contest policy documents",

    # ── Corpus: document summaries ────────────────────────────────────────────
    doc_summary_focus=(
        "purpose, scope, eligible roles, contest/reward structure, "
        "key thresholds and amounts, timelines, locations, exceptions, and definitions"
    ),
    chunk_summary_capture=(
        "contest/reward names, eligible roles, qualification thresholds, "
        "FYFP amounts, reward values, locations, timelines, exceptions, and definitions"
    ),
    combine_summary_retain=(
        "contest names, eligible roles, reward amounts, locations, thresholds, "
        "timelines, exceptions, and key definitions"
    ),
    finalize_summary_focus=(
        "eligible roles, contest/reward structure, thresholds, "
        "locations, timelines, exceptions, and definitions"
    ),

    # ── Corpus: section summaries ─────────────────────────────────────────────
    section_summary_capture=(
        "eligible roles, qualification thresholds with EXACT amounts, reward details "
        "with EXACT values, locations, timelines with EXACT dates or periods, exceptions, "
        "definitions, and rules specific to this section"
    ),
    section_combine_preserve=(
        "eligible roles, reward amounts, qualification thresholds, locations, "
        "timelines, exceptions, and definitions"
    ),
    user_label="sales agent",

    # ── Corpus: FAQ ───────────────────────────────────────────────────────────
    faq_cover=(
        "eligibility criteria, reward amounts, key thresholds, timelines, locations, "
        "exceptions, role definitions, qualification rules, and contest structure"
    ),

    # ── Bot: aggregate queries ────────────────────────────────────────────────
    aggregate_response_hint=(
        "Prefer standalone contest names from document titles or sections that explicitly name the contest. "
        "Do not treat comparison-only mentions, incomplete-sentence mentions, or verification notes as main contest items. "
        "Do not output bare names without context."
    ),
    aggregate_positive_markers=(
        "contest name",
        "name and purpose",
        "contest identity",
        "contest overview",
        "contest structure",
        "structure summary",
        "contest is named",
        "contest is titled",
        "the contest is named",
        "the contest is titled",
    ),

    # ── Entity extraction ─────────────────────────────────────────────────────
    entity_categories=frozenset({
        "role",         # FC, EIM, ACH, channel head
        "location",     # Goa, Bali, Singapore — any named place
        "time_period",  # JFM, Q4, March 2026, contest window
        "action",       # travel, lunch, meet, achieve, qualify
        "reward",       # Rolex, Samsung Galaxy, cash, trip, voucher
        "threshold",    # 85%, ₹3,49,000, ≥₹25,000 FYFP
        "product",      # Non-UL, Term, ULIP, life insurance product
        "contest",      # Bali Bliss, Power League, Dhurandhar
        "abbreviation", # FYFP, DPPM, HDFC, ACE, ESL, T&C
        "other",
    }),
    entity_global_categories=frozenset({"role", "abbreviation", "contest", "time_period"}),
    entity_extraction_rules=(
        "- Extract ALL meaningful entities: roles, locations, time periods, actions, "
        "rewards, thresholds, products, contest names, and abbreviations.\n"
        "- For roles: describe the role's level, responsibilities, and which contests they qualify for.\n"
        "- For locations: describe whether domestic or international, the country, "
        "and what the location represents (travel reward, event venue, etc.).\n"
        "- For rewards: describe the reward type (cash, travel, voucher, product) and its value if stated.\n"
        "- For thresholds: describe what the number means (eligibility criterion, "
        "minimum FYFP, persistency rate, slab amount) and who it applies to.\n"
        "- For time periods: describe the window, what it covers, and any deadlines within it.\n"
        "- For abbreviations: spell them out fully in context.\n"
        "- For contest names: describe the contest purpose and who can participate.\n"
        "- Synonyms must reflect how a non-expert sales agent would phrase a question "
        "(natural, conversational English — the way someone types in a chat).\n"
        "- Be exhaustive: include every entity a user in this domain might ask about."
    ),
    entity_examples=(
        '[\n'
        '  {"name":"Goa","category":"location","context":"Domestic travel reward destination '
        'in India offered to qualifying agents","synonyms":["goa trip","india trip",'
        '"domestic travel","goa event"]},\n'
        '  {"name":"FYFP","category":"abbreviation","context":"First Year First Premium — '
        'the total new business premium collected in the first policy year, used as the '
        'contest performance metric","synonyms":["first year premium","first year first premium",'
        '"new premium","production"]}\n'
        ']'
    ),

    # ── Web UI ────────────────────────────────────────────────────────────────
    ui_assistant_label="Agency Sales Assistant",
    ui_eyebrow="Ask contest & policy questions",
    ui_description=(
        "Get instant answers on contest eligibility, FYFP thresholds, rewards, and timelines — "
        "grounded in your indexed policy documents."
    ),
    ui_prompt_chips=(
        ("List all contests", "List all active contests and who is eligible for each — FC, EIM, or ACH."),
        ("FYFP thresholds", "What is the minimum FYFP threshold for FC to qualify for each contest?"),
        ("Travel rewards", "Which contests offer travel rewards and what are the destinations?"),
        ("Contest timelines", "What are the contest period dates and enrollment deadlines for all current contests?"),
    ),
)

register("contest", PROFILE)
