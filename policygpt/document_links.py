from urllib.parse import quote


def _base_prefix(base_url: str) -> str:
    return (base_url or "").rstrip("/")


def build_document_open_url(base_url: str, source_path: str) -> str:
    prefix = _base_prefix(base_url)
    encoded_path = quote(source_path, safe="")
    return f"{prefix}/api/documents/open?path={encoded_path}" if prefix else f"/api/documents/open?path={encoded_path}"


def build_document_view_url(
    base_url: str,
    source_path: str,
    section_index: int | None = None,
    section_title: str = "",
) -> str:
    prefix = _base_prefix(base_url)
    query_parts = [f"path={quote(source_path, safe='')}"]
    if section_index is not None:
        query_parts.append(f"section_index={section_index}")
    if section_title:
        query_parts.append(f"section_title={quote(section_title[:200], safe='')}")
    query_string = "&".join(query_parts)
    return f"{prefix}/api/documents/view?{query_string}" if prefix else f"/api/documents/view?{query_string}"
