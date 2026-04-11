"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.domain import DomainProfile, get_domain_profile
import policygpt.core.domain.contest   # noqa: F401 — registers "contest"
import policygpt.core.domain.policy    # noqa: F401 — registers "policy"

__all__ = ["DomainProfile", "get_domain_profile"]
