"""OpenSearch HTTPS client factory.

Creates a lazily-connected OpenSearch client using username/password auth
over HTTPS.  SSL certificate verification is optional (useful for self-signed
certs in dev/staging environments).
"""

from __future__ import annotations


def create_client(
    host: str,
    port: int,
    username: str,
    password: str,
    use_ssl: bool = True,
    verify_certs: bool = False,
):
    """Return a connected OpenSearch client.

    The opensearch-py SDK is imported here rather than at module level so the
    rest of the application does not fail to import when the package is absent
    (i.e. when opensearch_enabled=False).
    """
    try:
        from opensearchpy import OpenSearch, RequestsHttpConnection
    except ImportError as exc:
        raise ImportError(
            "OpenSearch support requires the 'opensearch-py' package. "
            "Install it with:  pip install opensearch-py"
        ) from exc

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(username, password),
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
