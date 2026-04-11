"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.document_links import build_document_open_url, build_document_view_url

__all__ = ["build_document_open_url", "build_document_view_url"]
