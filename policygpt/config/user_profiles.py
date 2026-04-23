"""User profile utilities for per-request and per-thread personalization.

Structure
---------
UserProfile             — request/thread context describing the current user
DOMAIN_DEFAULT_PROFILES — optional domain baselines kept for future integrations

Today the bot should *not* silently assume a default identity for anonymous
users.  When we do not know who the user is, we return an empty profile and let
the conversation flow ask for the missing context explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class UserProfile:
    """Describes the user making a request.

    Fields are all optional — only populated fields influence retrieval and
    answer generation.  Unknown or anonymous users get an empty profile so
    retrieval is unaffected.
    """
    role: str = ""          # e.g. "Branch Manager", "Sales Agent", "Software Engineer"
    grade: str = ""         # e.g. "M3", "E4", "Senior"
    department: str = ""    # e.g. "Retail Banking", "Agency Sales", "Platform Engineering"
    location: str = ""      # e.g. "Mumbai", "Delhi", "Remote"
    # Free-form tags — union of role/grade/department/location tokens plus any
    # domain-specific labels.  Built automatically by ``build_tags()``.
    tags: tuple[str, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not any([self.role, self.grade, self.department, self.location])

    def build_tags(self) -> "UserProfile":
        """Return a new UserProfile with tags populated from the other fields."""
        raw: list[str] = []
        for value in (self.role, self.grade, self.department, self.location):
            if value:
                # Split multi-word values so "Branch Manager" → ["branch", "manager"]
                raw.extend(token.lower() for token in value.split())
                raw.append(value.lower())
        seen: list[str] = []
        for t in raw:
            if t and t not in seen:
                seen.append(t)
        return UserProfile(
            role=self.role,
            grade=self.grade,
            department=self.department,
            location=self.location,
            tags=tuple(seen),
        )

    def context_line(self) -> str:
        """One-line summary for injection into LLM prompts."""
        parts = [p for p in (self.role, self.grade, self.department, self.location) if p]
        return " | ".join(parts) if parts else ""

    def merged_with(self, fallback: "UserProfile | None" = None) -> "UserProfile":
        """Return a profile where this profile's non-empty fields win."""
        fallback = fallback or UserProfile()
        return UserProfile(
            role=self.role or fallback.role,
            grade=self.grade or fallback.grade,
            department=self.department or fallback.department,
            location=self.location or fallback.location,
        ).build_tags()


# ---------------------------------------------------------------------------
# Domain-level default profiles
# Used until real per-user login is wired in.  Represents the typical user
# population for each domain so retrieval is already role-aware by default.
# ---------------------------------------------------------------------------

_POLICY_DEFAULT = UserProfile(
    # Enterprise HR / IT / Finance policy domain
    # Covers the full employee population — from individual contributors to
    # senior managers — across HR, Finance, IT, and Operations.
    role="Manager",
    grade="",
    department="Engineering",
    location="",
).build_tags()

_CONTEST_DEFAULT = UserProfile(
    # Insurance agency sales contest domain
    # Primary users: FCs (Financial Consultants), EIMs (Enterprise Insurance Managers),
    # ACHs (Agency Channel Heads), and Branch Managers in the agency sales channel.
    role="Financial Consultant",
    grade="",
    department="Agency Sales",
    location="",
).build_tags()

_PRODUCT_TECHNICAL_DEFAULT = UserProfile(
    # Software / cloud infrastructure technical documentation domain
    # Primary users: backend engineers, DevOps/SRE, solution architects, and
    # technical leads working on AWS-based platforms.
    role="Software Architect",
    grade="",
    department="Platform Engineering",
    location="",
).build_tags()

DOMAIN_DEFAULT_PROFILES: dict[str, UserProfile] = {
    "policy":            _POLICY_DEFAULT,
    "contest":           _CONTEST_DEFAULT,
    "product_technical": _PRODUCT_TECHNICAL_DEFAULT,
}


def merge_user_profiles(primary: UserProfile | None, fallback: UserProfile | None = None) -> UserProfile:
    """Return a merged profile where primary values override fallback values."""
    return (primary or UserProfile()).merged_with(fallback)


def _clean_profile_value(value: str) -> str:
    compact = " ".join((value or "").strip(" \t\r\n,;:-").split())
    compact = re.sub(r"^(?:a|an|the)\s+", "", compact, flags=re.IGNORECASE)
    return compact.strip()


def _looks_like_question(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if "?" in normalized:
        return True
    return bool(re.match(r"^(?:who|what|when|where|why|how|am|can|do|does|did|is|are|should|could|would)\b", normalized))


def parse_user_profile_text(text: str) -> UserProfile:
    """Parse a short free-form profile reply into a UserProfile.

    Supports explicit labeled input such as:
        Role: Branch Manager; Department: Retail Banking; Grade: M3; Location: Mumbai

    and lightweight natural language such as:
        I'm a Branch Manager in Retail Banking at Mumbai
    """
    raw_text = (text or "").strip()
    if not raw_text:
        return UserProfile()

    fields: dict[str, str] = {
        "role": "",
        "grade": "",
        "department": "",
        "location": "",
    }
    pattern_map: dict[str, tuple[str, ...]] = {
        "role": (
            r"\b(?:role|designation|job title|title)\s*[:=-]\s*([^;\n,]+)",
            r"\b(?:i am|i'm|im|my role is|role is|designation is)\s+(?:an?\s+)?(.+?)(?=\s+(?:in|at|from|based in|located in)\b|[.;]|$)",
        ),
        "grade": (
            r"\b(?:grade|level|band)\s*[:=-]\s*([^;\n,]+)",
        ),
        "department": (
            r"\b(?:department|dept|team|function|business unit)\s*[:=-]\s*([^;\n,]+)",
            r"\b(?:i am|i'm|im).+?\bin\s+(.+?)(?=\s+(?:at|from|based in|located in)\b|[.;]|$)",
        ),
        "location": (
            r"\b(?:location|city|office|region|country)\s*[:=-]\s*([^;\n,]+)",
            r"\b(?:at|from|based in|located in)\s+(.+?)(?=[.;]|$)",
        ),
    }
    for field_name, patterns in pattern_map.items():
        for pattern in patterns:
            match = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if not match:
                continue
            fields[field_name] = _clean_profile_value(match.group(1))
            if fields[field_name]:
                break

    if not any(fields.values()):
        compact = _clean_profile_value(raw_text)
        if compact and len(compact.split()) <= 6 and ":" not in raw_text and not _looks_like_question(raw_text):
            fields["role"] = compact

    return UserProfile(
        role=fields["role"],
        grade=fields["grade"],
        department=fields["department"],
        location=fields["location"],
    ).build_tags()


def resolve_user_profile(domain_type: str, user_id: str | int | None = None) -> UserProfile:
    """Return the UserProfile for a request.

    Today: anonymous or unresolved users intentionally return an empty profile.
    Later: look up user_id in a profile store / decode from JWT and return the
    real profile.
    """
    # TODO: replace with real profile lookup when login is wired
    # e.g.:  profile = ProfileService.get(user_id)
    #        if profile: return profile.build_tags()
    return UserProfile()
