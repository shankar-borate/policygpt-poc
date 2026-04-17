"""policygpt.core.domain — domain profile registry.

Import this package to register all built-in domain profiles.
Use get_domain_profile(name) to retrieve a registered profile.

Built-in domains: "contest" | "policy" | "product_technical"
"""

from policygpt.core.domain.base import DomainProfile, get_domain_profile, register

# Trigger registration of built-in profiles
import policygpt.core.domain.contest           # noqa: F401
import policygpt.core.domain.policy            # noqa: F401
import policygpt.core.domain.product_technical  # noqa: F401

__all__ = ["DomainProfile", "register", "get_domain_profile"]
