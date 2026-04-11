"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.domain.base import DomainProfile, register, get_domain_profile

__all__ = ["DomainProfile", "register", "get_domain_profile"]
