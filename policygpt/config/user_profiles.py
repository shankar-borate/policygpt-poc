"""User profile — role/grade/department context injected at request time.

Structure
---------
UserProfile          — per-request context (populated by API layer from session/token)
DOMAIN_DEFAULT_PROFILES — domain-keyed defaults used until real login is wired

At request time the API layer calls ``resolve_user_profile(domain_type, request)``
which returns the real profile (from JWT / session) when available, falling back
to the domain default.  No other code needs to change when real login is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def resolve_user_profile(domain_type: str, user_id: str | int | None = None) -> UserProfile:
    """Return the UserProfile for a request.

    Today: always returns the domain default.
    Later: look up user_id in a profile store / decode from JWT and return the
    real profile; fall back to the domain default only when lookup fails.
    """
    # TODO: replace with real profile lookup when login is wired
    # e.g.:  profile = ProfileService.get(user_id)
    #        if profile: return profile.build_tags()
    return DOMAIN_DEFAULT_PROFILES.get(domain_type, UserProfile())
