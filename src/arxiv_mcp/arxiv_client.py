"""Async arXiv API wrapper with caching and PDF extraction."""

import asyncio
import logging
import tempfile
import time
import httpx
from pathlib import Path
from typing import Any, Optional

import arxiv

from .cache import metadata_cache, text_cache
from .categories import resolve_categories
from .pdf_extractor import ExtractedPaper, extract_paper_text

logger = logging.getLogger("arxiv_mcp.arxiv_client")

# --- Constants ---
ARXIV_BASE_URL = "https://arxiv.org/abs"
ARXIV_USER_AGENT = "arxiv-scout/0.1.0 (+https://github.com/shawnnygoh/arxiv-scout)"
MAX_RETRIES = 3
RATE_LIMIT_SECONDS = 3.0
PAGE_SIZE = 50

# --- Shared arXiv Client (sync, used inside threads) ---
_client = arxiv.Client(
    page_size=PAGE_SIZE,
    delay_seconds=RATE_LIMIT_SECONDS,
    num_retries=MAX_RETRIES,
)

# --- Sort Criterion Mapping ---
SORT_MAP = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submitted_date": arxiv.SortCriterion.SubmittedDate,
    "last_updated": arxiv.SortCriterion.LastUpdatedDate,
}


def _extract_id(result: arxiv.Result) -> str:
    """Extract the short arXiv ID from a result's entry_id URL."""
    return result.entry_id.split("/abs/")[-1]


def _result_to_dict(result: arxiv.Result) -> dict[str, Any]:
    """Convert an arxiv.Result to a serializable dictionary."""
    return {
        "arxiv_id": _extract_id(result),
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "summary": result.summary,
        "published": result.published.isoformat() if result.published else None,
        "updated": result.updated.isoformat() if result.updated else None,
        "primary_category": result.primary_category,
        "categories": result.categories,
        "pdf_url": result.pdf_url,
        "abs_url": f"{ARXIV_BASE_URL}/{_extract_id(result)}",
        "comment": result.comment,
        "journal_ref": result.journal_ref,
        "doi": result.doi,
    }


def _sync_search(
    query: str,
    max_results: int,
    sort_by: str,
    date_from: Optional[str],
    date_to: Optional[str],
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    """Synchronous search — runs in a thread."""
    search = arxiv.Search(
        query=query,
        max_results=offset + max_results,
        sort_by=SORT_MAP.get(sort_by, arxiv.SortCriterion.Relevance),
    )

    all_results = []
    for r in _client.results(search):
        paper = _result_to_dict(r)

        # Client-side date filtering
        if date_from and paper["published"]:
            if paper["published"][:10] < date_from:
                continue
        if date_to and paper["published"]:
            if paper["published"][:10] > date_to:
                continue

        all_results.append(paper)
        # Cache each result as we go
        metadata_cache.set(paper["arxiv_id"], paper)

    total = len(all_results)
    paginated = all_results[offset : offset + max_results]
    return paginated, total


def _sync_get_paper(arxiv_id: str) -> dict[str, Any]:
    """Synchronous single paper fetch — runs in a thread."""
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    if not results:
        raise ValueError(f"No paper found with ID: {arxiv_id}")
    paper = _result_to_dict(results[0])
    metadata_cache.set(arxiv_id, paper)
    return paper


def _sync_get_papers_batch(arxiv_ids: list[str]) -> list[dict[str, Any]]:
    """Synchronous batch fetch — runs in a thread."""
    search = arxiv.Search(id_list=arxiv_ids)
    papers = []
    for r in _client.results(search):
        paper = _result_to_dict(r)
        metadata_cache.set(paper["arxiv_id"], paper)
        papers.append(paper)
    return papers


def _sync_download_and_extract(
    arxiv_id: str,
    max_characters: int,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> ExtractedPaper:
    """Synchronous PDF download + extraction — runs in a thread."""
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    if not results:
        raise ValueError(f"No paper found with ID: {arxiv_id}")
    pdf_url = results[0].pdf_url
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / f"{arxiv_id.replace('/', '_')}.pdf"
        with httpx.Client(follow_redirects=True, timeout=60.0,
                          headers={"User-Agent": ARXIV_USER_AGENT}) as client:
            for attempt in range(MAX_RETRIES):
                resp = client.get(pdf_url)
                if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                    continue
                resp.raise_for_status()
                pdf_path.write_bytes(resp.content)
                break
        logger.debug("Downloaded PDF for %s (%s)", arxiv_id, pdf_path)
        extracted = extract_paper_text(
            pdf_path=str(pdf_path),
            max_characters=max_characters,
            start_page=start_page,
            end_page=end_page,
            smart=True,
        )

    return extracted


async def search_papers(
    query: str,
    max_results: int = 10,
    categories: Optional[list[str]] = None,
    sort_by: str = "relevance",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Search arXiv for papers matching a query (async).

    Returns:
        Tuple of (list of paper dicts, estimated total results).
    """
    # Resolve semantic category names to arXiv codes
    resolved_cats = resolve_categories(categories)

    # Build query with optional category filter
    full_query = query
    if resolved_cats:
        cat_filter = " OR ".join(f"cat:{c}" for c in resolved_cats)
        full_query = f"({query}) AND ({cat_filter})"

    return await asyncio.to_thread(
        _sync_search,
        full_query,
        max_results,
        sort_by,
        date_from,
        date_to,
        offset,
    )


async def get_paper(arxiv_id: str) -> dict[str, Any]:
    """Retrieve metadata for a single paper by arXiv ID (async, cached)."""
    cached = metadata_cache.get(arxiv_id)
    if cached is not None:
        logger.debug("Cache hit for metadata: %s", arxiv_id)
        return cached

    return await asyncio.to_thread(_sync_get_paper, arxiv_id)


async def get_papers_batch(arxiv_ids: list[str]) -> list[dict[str, Any]]:
    """Retrieve metadata for multiple papers (async, with partial cache)."""
    results = []
    uncached_ids = []

    for aid in arxiv_ids:
        cached = metadata_cache.get(aid)
        if cached is not None:
            results.append(cached)
        else:
            uncached_ids.append(aid)

    if uncached_ids:
        fetched = await asyncio.to_thread(_sync_get_papers_batch, uncached_ids)
        results.extend(fetched)

    return results


async def download_and_extract(
    arxiv_id: str,
    max_characters: int = 50000,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> ExtractedPaper:
    """Download a paper's PDF and extract text (async, cached)."""
    cache_key = f"{arxiv_id}:{max_characters}:{start_page}:{end_page}"
    cached = text_cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for text: %s", arxiv_id)
        return cached

    extracted = await asyncio.to_thread(
        _sync_download_and_extract, arxiv_id, max_characters, start_page, end_page
    )
    text_cache.set(cache_key, extracted)
    return extracted


def handle_error(e: Exception, context: str) -> str:
    """Produce a consistent, actionable error message."""
    if isinstance(e, arxiv.UnexpectedEmptyPageError):
        return (
            f"Error ({context}): arXiv returned no results. "
            "The query may be too specific or the paper ID may not exist. "
            "Try broadening your search or double-checking the ID."
        )
    if isinstance(e, arxiv.HTTPError):
        if 400 <= e.status < 500:
            return (
                f"Error ({context}): arXiv rejected the request (HTTP {e.status}). "
                "Please verify the arXiv ID is correct (e.g., '2301.07041')."
            )
        return (
            f"Error ({context}): arXiv API returned HTTP {e.status}. "
            "This is usually temporary — please retry in a few seconds."
        )
    if isinstance(e, (StopIteration, ValueError)):
        return (
            f"Error ({context}): No paper found with the given ID. "
            "Please verify the arXiv ID is correct (e.g., '2301.07041')."
        )
    return f"Error ({context}): {type(e).__name__}: {e}"
