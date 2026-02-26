"""Semantic Scholar API client for citation network data."""

import asyncio
import logging
import os
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("arxiv_mcp.semantic_scholar")

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 30.0

# Fields for papers returned inside citation/reference lists
_PAPER_FIELDS = ",".join([
    "paperId", "externalIds", "title", "authors", "year",
    "citationCount", "abstract", "url", "venue",
])

_client: Optional[httpx.AsyncClient] = None
_client_loop: Optional[asyncio.AbstractEventLoop] = None


def _strip_version(arxiv_id: str) -> str:
    """Strip version suffix from arXiv IDs for Semantic Scholar lookups.

    S2 indexes papers by base ID (e.g., '2412.19437'), not versioned IDs
    (e.g., '2412.19437v1'). Passing a version suffix causes 404 errors.
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def _get_client() -> httpx.AsyncClient:
    """Lazy-init a shared async HTTP client.

    Handles checking if the event loop has changed (e.g. in tests) to ensure
    we don't use a client bound to a closed loop.
    """
    global _client, _client_loop

    # Get the current running loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    # Check if we have a client but it's bound to a different (or closed) loop
    if _client is not None:
        if _client.is_closed:
            _client = None
        elif current_loop is not None and _client_loop != current_loop:
            logger.debug("Event loop changed, resetting Semantic Scholar client")
            _client = None

    if _client is None:
        headers = {"User-Agent": "arxiv-scout/0.1.0"}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        _client = httpx.AsyncClient(
            base_url=S2_BASE,
            timeout=S2_TIMEOUT,
            headers=headers,
        )
        _client_loop = current_loop

    return _client


async def close_client() -> None:
    """Close the shared HTTP client. Call during server shutdown."""
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _client_loop = None


def _normalize_paper(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize an S2 paper object into a consistent shape."""
    external = raw.get("externalIds") or {}
    authors = raw.get("authors") or []
    return {
        "s2_id": raw.get("paperId"),
        "arxiv_id": external.get("ArXiv"),
        "doi": external.get("DOI"),
        "title": raw.get("title") or "Unknown Title",
        "authors": [a.get("name", "") for a in authors],
        "year": raw.get("year"),
        "citation_count": raw.get("citationCount"),
        "venue": raw.get("venue") or None,
        "abstract": raw.get("abstract"),
        "url": raw.get("url"),
    }


async def get_references(
    arxiv_id: str,
    limit: int = 200,
    api_key: Optional[str] = None,
) -> Optional[list[dict[str, Any]]]:
    """Get papers cited BY this paper (outbound references).

    Returns a list of normalized paper dicts, or None if the API call fails.
    """
    arxiv_id = _strip_version(arxiv_id)
    client = _get_client()

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = await client.get(
            f"/paper/ARXIV:{arxiv_id}/references",
            params={"fields": f"citedPaper.{_PAPER_FIELDS}", "limit": limit},
            headers=headers,
        )
        if resp.status_code == 404:
            logger.debug("Paper %s not found in Semantic Scholar", arxiv_id)
            return None
        resp.raise_for_status()

        data = resp.json().get("data", [])
        references = []
        for entry in data:
            cited = entry.get("citedPaper")
            if cited and cited.get("title"):
                references.append(_normalize_paper(cited))
        return references

    except httpx.HTTPStatusError as e:
        logger.warning("S2 HTTP error for references %s: %s", arxiv_id, e)
        return None
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("S2 connection issue for references %s: %s", arxiv_id, e)
        return None


async def get_citations(
    arxiv_id: str,
    limit: int = 200,
    api_key: Optional[str] = None,
) -> Optional[list[dict[str, Any]]]:
    """Get papers that CITE this paper (inbound citations).

    Returns a list of normalized paper dicts, or None if the API call fails.
    """
    arxiv_id = _strip_version(arxiv_id)
    client = _get_client()

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = await client.get(
            f"/paper/ARXIV:{arxiv_id}/citations",
            params={"fields": f"citingPaper.{_PAPER_FIELDS}", "limit": limit},
        )
        if resp.status_code == 404:
            logger.debug("Paper %s not found in Semantic Scholar", arxiv_id)
            return None
        resp.raise_for_status()

        data = resp.json().get("data", [])
        citations = []
        for entry in data:
            citing = entry.get("citingPaper")
            if citing and citing.get("title"):
                citations.append(_normalize_paper(citing))
        return citations

    except httpx.HTTPStatusError as e:
        logger.warning("S2 HTTP error for citations %s: %s", arxiv_id, e)
        return None
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("S2 connection issue for citations %s: %s", arxiv_id, e)
        return None


async def get_paper_metadata(
    arxiv_id: str,
    api_key: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Get enhanced metadata from Semantic Scholar (citation counts, venue).

    Returns a normalized paper dict, or None if unavailable.
    """
    arxiv_id = _strip_version(arxiv_id)
    client = _get_client()

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = await client.get(
            f"/paper/ARXIV:{arxiv_id}",
            params={"fields": _PAPER_FIELDS},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _normalize_paper(resp.json())

    except httpx.HTTPStatusError as e:
        logger.warning("S2 HTTP error for metadata %s: %s", arxiv_id, e)
        return None
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("S2 connection issue for metadata %s: %s", arxiv_id, e)
        return None