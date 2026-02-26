"""arXiv MCP Server — tools, prompts, and resources for arXiv paper access."""

import contextvars
import json
import logging
import os
from typing import Optional

import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

from arxiv_mcp import arxiv_client, formatting, semantic_scholar
from arxiv_mcp.cache import metadata_cache, text_cache
from arxiv_mcp.categories import CATEGORY_TAXONOMY, get_flat_taxonomy, resolve_categories

logger = logging.getLogger("arxiv_mcp.server")

VERSION = "0.1.0"

session_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_api_key", default=None
)


def get_semantic_scholar_api_key() -> str | None:
    """Get the Semantic Scholar API key for the current request.

    Priority:
    1. Per-session key from URL query params (set by SessionConfigMiddleware)
    2. Server-wide key from environment variable (fallback for local/DO deploys)
    """
    return session_api_key.get()


class SessionConfigMiddleware(BaseHTTPMiddleware):
    """Capture per-session config from URL query parameters.

    Smithery delivers user config as query params on the MCP endpoint URL:
        /mcp?SEMANTIC_SCHOLAR_API_KEY=xxx

    This middleware reads them into contextvars so each concurrent
    request/user gets isolated values. Falls back to env vars.
    """

    async def dispatch(self, request, call_next):
        api_key = request.query_params.get("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key:
            api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        token = session_api_key.set(api_key)
        try:
            response = await call_next(request)
            return response
        finally:
            session_api_key.reset(token)


mcp = FastMCP(
    name="arxiv-scout",
    host="0.0.0.0",
    instructions=(
        "An MCP server for searching, retrieving, and analyzing papers from arXiv.org. "
        "Supports full arXiv query syntax, semantic category names, PDF text extraction "
        "with page ranges, and citation network exploration via Semantic Scholar. "
        "Read the arxiv://help/query-syntax resource for advanced search operators."
    ),
    stateless_http=False,
)


def _normalize_arxiv_id(raw: str) -> str:
    """Extract a clean arXiv ID from a URL or raw string."""
    raw = raw.strip()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw.endswith(".pdf"):
        raw = raw[:-4]
    return raw


def _parse_list_arg(value: str) -> list[str]:
    """Parse a prompt argument into a list.

    MCP prompts pass all arguments as strings. This handles both JSON arrays
    (e.g. '["a", "b"]') and comma-separated values (e.g. "a, b").
    """
    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in value.split(",") if item.strip()]


# --- Tools ---


@mcp.tool(annotations=ToolAnnotations(
    title="Search arXiv Papers",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
))
async def arxiv_search_papers(
    ctx: Context,
    query: str,
    max_results: int = 10,
    categories: Optional[list[str]] = None,
    sort_by: str = "relevance",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    offset: int = 0,
    output_format: str = "markdown",
) -> str:
    """Search arXiv for papers matching a query.

    Supports arXiv query syntax: field prefixes (au:, ti:, abs:, cat:),
    boolean operators (AND, OR, ANDNOT), and grouping with parentheses.
    Categories can be arXiv codes (cs.LG) or natural language (machine learning).

    Args:
        query: Search query. Examples: "attention mechanism", "au:vaswani AND ti:attention",
               "abs:large language model AND cat:cs.CL"
        max_results: Number of results to return (1-50, default 10).
        categories: Optional category filter. Accepts arXiv codes ("cs.LG") or
                    natural language ("machine learning", "computer vision").
        sort_by: Sort order — "relevance", "submitted_date", or "last_updated".
        date_from: Filter papers published on or after this date (YYYY-MM-DD).
        date_to: Filter papers published on or before this date (YYYY-MM-DD).
        offset: Pagination offset (0-based). Use with max_results for paging.
        output_format: "markdown" for human-readable or "json" for structured data.
    """
    max_results = max(1, min(50, max_results))
    offset = max(0, offset)

    await ctx.report_progress(0, 2)
    logger.debug("Searching: query=%r, max=%d, cats=%s", query, max_results, categories)

    try:
        papers, total = await arxiv_client.search_papers(
            query=query,
            max_results=max_results,
            categories=categories,
            sort_by=sort_by,
            date_from=date_from,
            date_to=date_to,
            offset=offset,
        )
        await ctx.report_progress(2, 2)
    except Exception as e:
        return arxiv_client.handle_error(e, "search")

    if not papers:
        return f"No papers found for query: *{query}*. Try broadening your search or adjusting category/date filters."

    return formatting.format_search_results(papers, total, query, offset, output_format)


@mcp.tool(annotations=ToolAnnotations(
    title="Get arXiv Paper Details",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
))
async def arxiv_get_paper(
    ctx: Context,
    arxiv_id: str,
    output_format: str = "markdown",
) -> str:
    """Retrieve metadata for a single paper by arXiv ID or URL.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.07041", "2301.07041v2") or
                  full URL (e.g., "https://arxiv.org/abs/2301.07041").
        output_format: "markdown" for human-readable or "json" for structured data.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)

    try:
        paper = await arxiv_client.get_paper(arxiv_id)
    except Exception as e:
        return arxiv_client.handle_error(e, f"get_paper({arxiv_id})")

    return formatting.format_paper(paper, output_format)


@mcp.tool(annotations=ToolAnnotations(
    title="Batch Search arXiv Papers",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
))
async def arxiv_get_papers_batch(
    ctx: Context,
    arxiv_ids: list[str],
    output_format: str = "markdown",
) -> str:
    """Retrieve metadata for multiple papers in a single call.

    More efficient than calling arxiv_get_paper repeatedly — batches the API
    request and leverages the metadata cache for previously-seen papers.

    Args:
        arxiv_ids: List of arXiv IDs or URLs (max 20).
        output_format: "markdown" for human-readable or "json" for structured data.
    """
    if len(arxiv_ids) > 20:
        return "Error: Maximum 20 papers per batch request. Please split into smaller batches."

    clean_ids = [_normalize_arxiv_id(aid) for aid in arxiv_ids]

    try:
        papers = await arxiv_client.get_papers_batch(clean_ids)
    except Exception as e:
        return arxiv_client.handle_error(e, "batch_fetch")

    return formatting.format_batch(papers, output_format)


@mcp.tool(annotations=ToolAnnotations(
    title="Download arXiv Paper",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
))
async def arxiv_download_and_extract(
    ctx: Context,
    arxiv_id: str,
    max_characters: int = 50000,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    include_sections: bool = True,
    include_references: bool = False,
) -> str:
    """Download a paper's PDF and extract its full text.

    Uses smart extraction: strips repeated headers/footers, detects section
    boundaries (Abstract, Introduction, Methods, etc.), and optionally includes
    reference entries. Supports page-range extraction for focused reading.

    Results are cached for 24 hours — repeated calls for the same paper are instant.

    Args:
        arxiv_id: The arXiv paper ID or URL.
        max_characters: Maximum characters to return (1000-200000, default 50000).
        start_page: 1-based start page (inclusive). Omit for first page.
        end_page: 1-based end page (inclusive). Omit for last page.
        include_sections: If true, prepend a document structure outline.
        include_references: If true, append extracted reference entries from the PDF.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)
    max_characters = max(1000, min(200000, max_characters))

    await ctx.report_progress(0, 3)

    try:
        paper = await arxiv_client.get_paper(arxiv_id)
        await ctx.report_progress(1, 3)

        extracted = await arxiv_client.download_and_extract(
            arxiv_id, max_characters, start_page, end_page
        )
        await ctx.report_progress(3, 3)
    except Exception as e:
        return arxiv_client.handle_error(e, f"download_extract({arxiv_id})")

    if extracted.char_count == 0:
        total = extracted.page_count
        if start_page and start_page > total:
            return (
                f"Error: start_page ({start_page}) exceeds the paper's "
                f"page count ({total}). Use start_page between 1 and {total}."
            )
        if end_page and end_page > total:
            return (
                f"Warning: end_page ({end_page}) exceeds the paper's "
                f"page count ({total}). Returning pages up to {total}."
            )

    return formatting.format_extracted_paper(
        paper, extracted, include_sections, include_references
    )


@mcp.tool(annotations=ToolAnnotations(
    title="Get Paper References",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
))
async def arxiv_get_references(
    ctx: Context,
    arxiv_id: str,
    limit: int = 100,
    output_format: str = "markdown",
) -> str:
    """Get the papers cited BY this paper (outbound references / bibliography).

    Uses Semantic Scholar's citation graph for structured, accurate results.
    Falls back to PDF text extraction if the paper isn't indexed by S2.

    Args:
        arxiv_id: The arXiv paper ID or URL.
        limit: Maximum number of references to return (1-500, default 100).
        output_format: "markdown" for human-readable or "json" for structured data.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)
    limit = max(1, min(500, limit))

    await ctx.report_progress(0, 3)

    try:
        paper = await arxiv_client.get_paper(arxiv_id)
        await ctx.report_progress(1, 3)

        api_key = get_semantic_scholar_api_key()
        s2_refs = await semantic_scholar.get_references(arxiv_id, limit=limit, api_key=api_key)
        await ctx.report_progress(2, 3)

        if s2_refs is not None:
            await ctx.report_progress(3, 3)
            return formatting.format_references(
                paper, s2_refs, source="semantic_scholar", fmt=output_format
            )

        logger.info("S2 unavailable for %s, falling back to PDF extraction", arxiv_id)
        extracted = await arxiv_client.download_and_extract(
            arxiv_id, max_characters=200000
        )
        await ctx.report_progress(3, 3)

        pdf_refs = [
            {"index": r.index, "text": r.raw_text}
            for r in extracted.references
        ]
        return formatting.format_references(
            paper, pdf_refs, source="pdf_extraction", fmt=output_format
        )
    except Exception as e:
        return arxiv_client.handle_error(e, f"get_references({arxiv_id})")


@mcp.tool(annotations=ToolAnnotations(
    title="Get Paper Citations",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
))
async def arxiv_get_citations(
    ctx: Context,
    arxiv_id: str,
    limit: int = 100,
    output_format: str = "markdown",
) -> str:
    """Get papers that CITE this paper (inbound citations).

    Uses Semantic Scholar's citation graph. This data cannot be obtained from
    arXiv or the paper's PDF — only a citation index like Semantic Scholar
    tracks which papers reference a given work.

    Args:
        arxiv_id: The arXiv paper ID or URL.
        limit: Maximum number of citations to return (1-500, default 100).
        output_format: "markdown" for human-readable or "json" for structured data.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)
    limit = max(1, min(500, limit))

    await ctx.report_progress(0, 2)

    try:
        paper = await arxiv_client.get_paper(arxiv_id)
        await ctx.report_progress(1, 2)

        api_key = get_semantic_scholar_api_key()
        citations = await semantic_scholar.get_citations(arxiv_id, limit=limit, api_key=api_key)
        await ctx.report_progress(2, 2)

        if citations is None:
            return (
                f"Could not retrieve citations for '{paper['title']}'. "
                "The paper may not yet be indexed by Semantic Scholar. "
                "Recently published papers can take a few weeks to appear."
            )

        return formatting.format_citations(paper, citations, fmt=output_format)
    except Exception as e:
        return arxiv_client.handle_error(e, f"get_citations({arxiv_id})")


@mcp.tool(annotations=ToolAnnotations(
    title="List arXiv Categories",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
))
async def arxiv_list_categories(
    ctx: Context,
    group: Optional[str] = None,
    output_format: str = "markdown",
) -> str:
    """Browse the arXiv category taxonomy.

    Returns category codes and descriptions. Can filter by group (cs, math, stat, etc.)
    or return the full taxonomy. Also supports semantic lookup: pass a natural language
    term like "machine learning" to see which arXiv categories map to it.

    Args:
        group: Optional group filter — "cs", "math", "stat", "physics", "econ", "q-bio",
               "q-fin", "eess", or a natural language term like "machine learning".
        output_format: "markdown" for human-readable or "json" for structured data.
    """
    if group:
        group_lower = group.lower().strip()

        if group_lower in CATEGORY_TAXONOMY:
            cats = CATEGORY_TAXONOMY[group_lower]
            if output_format == "json":
                return json.dumps({"group": group_lower, "categories": cats}, indent=2)
            lines = [f"## arXiv Categories: {group_lower}\n"]
            for code, name in cats.items():
                lines.append(f"- **{code}**: {name}")
            return "\n".join(lines)

        resolved = resolve_categories([group])
        if resolved:
            flat = get_flat_taxonomy()
            result_cats = {c: flat.get(c, "Unknown") for c in resolved}
            if output_format == "json":
                return json.dumps({"query": group, "resolved_categories": result_cats}, indent=2)
            lines = [f"## Categories matching: *{group}*\n"]
            for code, name in result_cats.items():
                lines.append(f"- **{code}**: {name}")
            return "\n".join(lines)

    if output_format == "json":
        return json.dumps(CATEGORY_TAXONOMY, indent=2)

    lines = ["## arXiv Category Taxonomy\n"]
    for grp, cats in CATEGORY_TAXONOMY.items():
        lines.append(f"### {grp}")
        for code, name in cats.items():
            lines.append(f"- **{code}**: {name}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(annotations=ToolAnnotations(
    title="Get Cache Statistics",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
))
async def arxiv_cache_stats(
    ctx: Context, 
    verbose: bool = False,
) -> str:
    """View cache statistics for monitoring and debugging.

    Shows hit rates, sizes, and TTL configuration for both the metadata
    and text extraction caches.
    """
    return formatting.format_cache_stats(metadata_cache.stats, text_cache.stats)


# --- Prompts ---
#
# MCP prompts pass all arguments as strings, so list parameters like
# arxiv_ids must be typed as str and parsed internally via _parse_list_arg.


@mcp.prompt()
async def summarize_paper(arxiv_id: str) -> str:
    """Generate a structured summary of an arXiv paper.

    Downloads and extracts the full text, then provides a guided prompt
    for producing a summary covering problem, methodology, results,
    limitations, and significance.

    Args:
        arxiv_id: The arXiv paper ID or URL to summarize.
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id)

    try:
        paper = await arxiv_client.get_paper(arxiv_id)
        extracted = await arxiv_client.download_and_extract(arxiv_id, max_characters=100000)
    except Exception as e:
        return f"Error fetching paper: {e}. Please verify the arXiv ID and try again."

    section_outline = ""
    if extracted.sections:
        section_outline = "\n**Detected sections**: " + ", ".join(
            s.title for s in extracted.sections if s.title != "Preamble"
        )

    return f"""Please provide a structured summary of this research paper.

**Title**: {paper['title']}
**Authors**: {', '.join(paper['authors'])}
**arXiv ID**: {paper['arxiv_id']}
**Published**: {paper.get('published', 'N/A')}
**Categories**: {', '.join(paper.get('categories', []))}
{section_outline}

## Full Paper Text

{extracted.raw_text}

---

## Summary Instructions

Please structure your summary as follows:

1. **Problem Statement**: What problem does this paper address? Why does it matter?
2. **Key Contributions**: What are the main contributions (2-4 bullet points)?
3. **Methodology**: How do the authors approach the problem? What techniques/models/frameworks do they use?
4. **Main Results**: What are the key findings? Include specific numbers and benchmarks where available.
5. **Limitations**: What limitations do the authors acknowledge or that are apparent?
6. **Significance & Impact**: How does this advance the field? What are the implications?
7. **Related Work**: How does this compare to prior approaches?

Keep the summary concise but thorough (aim for ~500-800 words)."""


@mcp.prompt()
async def compare_papers(arxiv_ids: str) -> str:
    """Generate a side-by-side comparison of multiple arXiv papers.

    Fetches metadata and abstracts for all papers, then provides a guided
    prompt for comparative analysis.

    Args:
        arxiv_ids: Comma-separated arXiv IDs or a JSON array (2-5 papers).
                   Examples: "1706.03762, 1810.04805" or '["1706.03762", "1810.04805"]'
    """
    ids = _parse_list_arg(arxiv_ids)

    if len(ids) < 2:
        return "Please provide at least 2 arXiv IDs to compare."
    if len(ids) > 5:
        return "Please provide at most 5 papers for comparison. Use literature_review for larger sets."

    clean_ids = [_normalize_arxiv_id(aid) for aid in ids]

    try:
        papers = await arxiv_client.get_papers_batch(clean_ids)
    except Exception as e:
        return f"Error fetching papers: {e}. Please verify the arXiv IDs and try again."

    paper_sections = []
    for i, paper in enumerate(papers, 1):
        paper_sections.append(
            f"### Paper {i}: {paper['title']}\n"
            f"**Authors**: {', '.join(paper['authors'][:5])}\n"
            f"**Published**: {paper.get('published', 'N/A')}\n"
            f"**Categories**: {', '.join(paper.get('categories', []))}\n\n"
            f"**Abstract**: {paper['summary']}\n"
        )

    papers_text = "\n---\n\n".join(paper_sections)

    return f"""Please provide a detailed comparison of these {len(papers)} research papers.

{papers_text}

---

## Comparison Instructions

Please structure your comparison as follows:

1. **Overview**: Briefly describe what each paper is about (1-2 sentences each).
2. **Problem & Motivation**: How do the papers frame the problem differently?
3. **Methodology Comparison**: What approaches does each paper use? What are the key differences?
4. **Results Comparison**: How do the results compare? Are they evaluated on the same benchmarks?
5. **Strengths & Weaknesses**: What are each paper's key strengths and limitations?
6. **Connections**: How do these papers relate to each other? Does one build on another?
7. **Recommendation**: For someone entering this area, which paper(s) should they read first and why?

Use a comparison table where appropriate."""


@mcp.prompt()
async def literature_review(
    topic: str,
    max_papers: str = "10",
    categories: Optional[str] = None,
) -> str:
    """Generate a literature review by searching arXiv for a topic.

    Searches arXiv, retrieves the top papers, and provides a guided prompt
    for synthesizing a literature review.

    Args:
        topic: The research topic to review.
        max_papers: Number of papers to include (5-20, default 10).
        categories: Optional category filter — comma-separated arXiv codes or
                    natural language terms. Examples: "cs.CL, cs.AI" or "nlp".
    """
    try:
        max_papers_int = max(5, min(20, int(max_papers)))
    except ValueError:
        max_papers_int = 10

    parsed_cats = _parse_list_arg(categories) if categories else None

    try:
        papers, total = await arxiv_client.search_papers(
            query=topic,
            max_results=max_papers_int,
            categories=parsed_cats,
            sort_by="relevance",
        )
    except Exception as e:
        return f"Error searching for papers: {e}. Please try a different query."

    if not papers:
        return f"No papers found for topic: '{topic}'. Try broadening the search or using different terms."

    paper_sections = []
    for i, paper in enumerate(papers, 1):
        paper_sections.append(
            f"### [{i}] {paper['title']}\n"
            f"**Authors**: {', '.join(paper['authors'][:5])}\n"
            f"**Published**: {paper.get('published', 'N/A')}\n"
            f"**Categories**: {', '.join(paper.get('categories', []))}\n\n"
            f"**Abstract**: {paper['summary']}\n"
        )

    papers_text = "\n---\n\n".join(paper_sections)

    return f"""Please synthesize a literature review on "{topic}" based on these {len(papers)} papers from arXiv.

{papers_text}

---

## Literature Review Instructions

Please structure the review as follows:

1. **Introduction**: What is the topic and why is it important? (2-3 sentences)
2. **Landscape Overview**: What are the major threads of research in this area?
3. **Key Themes**: Group the papers by approach or sub-topic. For each theme:
   - What is the core idea?
   - Which papers contribute to this theme?
   - How do they build on or differ from each other?
4. **Methodological Trends**: What techniques are most common? Are there emerging approaches?
5. **Open Problems & Gaps**: What questions remain unanswered? Where do the papers disagree?
6. **Future Directions**: Based on the trends, what are the most promising research directions?
7. **Reading Recommendations**: Which 3-5 papers are most essential for someone entering this field?

Cite papers by their number (e.g., [1], [3, 7]). Aim for ~800-1200 words."""


# --- Resources ---


@mcp.resource("arxiv://categories")
async def resource_categories() -> str:
    """Complete arXiv category taxonomy with codes and descriptions."""
    lines = ["# arXiv Category Taxonomy\n"]
    for group, cats in CATEGORY_TAXONOMY.items():
        lines.append(f"## {group}")
        for code, name in cats.items():
            lines.append(f"- **{code}**: {name}")
        lines.append("")
    return "\n".join(lines)


@mcp.resource("arxiv://help/query-syntax")
async def resource_query_syntax() -> str:
    """Guide to arXiv search query syntax with examples."""
    return """# arXiv Query Syntax Reference

## Field Prefixes
| Prefix | Field | Example |
|--------|-------|---------|
| `ti:` | Title | `ti:attention mechanism` |
| `au:` | Author | `au:vaswani` |
| `abs:` | Abstract | `abs:large language model` |
| `cat:` | Category | `cat:cs.CL` |
| `co:` | Comment | `co:accepted ICML` |
| `jr:` | Journal Reference | `jr:Nature` |
| `all:` | All fields | `all:transformer` |

## Boolean Operators
- **AND**: Both conditions must match → `au:vaswani AND ti:attention`
- **OR**: Either condition matches → `cat:cs.CL OR cat:cs.AI`
- **ANDNOT**: Exclude matches → `cat:cs.AI ANDNOT ti:survey`

## Grouping
Use parentheses: `(ti:transformer OR ti:attention) AND au:vaswani`

## Examples
| Goal | Query |
|------|-------|
| Papers by Vaswani about attention | `au:vaswani AND ti:attention` |
| LLM papers in NLP category | `abs:large language model AND cat:cs.CL` |
| AI papers, excluding surveys | `cat:cs.AI ANDNOT ti:survey` |
| Multimodal learning papers | `ti:multimodal AND (cat:cs.CV OR cat:cs.CL)` |

## Semantic Category Search
This server supports natural language category names:
- `"machine learning"` → cs.LG, stat.ML
- `"computer vision"` → cs.CV
- `"nlp"` → cs.CL
- `"robotics"` → cs.RO

Pass these as the `categories` parameter in search.
"""


@mcp.resource("arxiv://server/info")
async def resource_server_info() -> str:
    """Information about this arXiv MCP server and its capabilities."""
    return f"""# arXiv MCP Server v{VERSION}

## Tools
| Tool | Description |
|------|-------------|
| `arxiv_search_papers` | Full-text search with query syntax, semantic categories, date filtering, pagination |
| `arxiv_get_paper` | Get metadata by arXiv ID or URL |
| `arxiv_get_papers_batch` | Fetch up to 20 papers in one call |
| `arxiv_download_and_extract` | Download PDF → smart text extraction with page ranges and section detection |
| `arxiv_get_references` | Outbound references via Semantic Scholar (PDF fallback) |
| `arxiv_get_citations` | Inbound citations via Semantic Scholar |
| `arxiv_list_categories` | Browse arXiv taxonomy with semantic lookup |
| `arxiv_cache_stats` | Monitor cache hit rates |

## Prompts
| Prompt | Description |
|--------|-------------|
| `summarize_paper` | Guided summary of a single paper |
| `compare_papers` | Side-by-side comparison of 2-5 papers |
| `literature_review` | Synthesized review of papers on a topic |

## Citation Data
References and citations use the Semantic Scholar Academic Graph API.
If a paper isn't indexed by S2, references fall back to PDF text extraction.
Pass SEMANTIC_SCHOLAR_API_KEY as a connection config for higher rate limits (optional).
"""


# --- Entry Point ---


def main():
    """Entry point for Arxiv Scout using Streamable HTTP."""
    import atexit
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info("Starting Arxiv Scout v%s on %s:%d (Streamable HTTP)", VERSION, host, port)

    app = mcp.streamable_http_app()

    CARD_PATH = Path(__file__).parent / "server-card.json"
    try:
        with open(CARD_PATH, "r") as f:
            CARD_DATA = json.load(f)
        logger.info("Loaded static server card from %s", CARD_PATH)
    except Exception as e:
        logger.error("Could not load server-card.json: %s", e)
        CARD_DATA = {"error": "Server card file missing"}

    async def server_card(request: Request) -> JSONResponse:
        return JSONResponse(CARD_DATA)

    async def mcp_config(request: Request) -> JSONResponse:
        return JSONResponse(CARD_DATA.get("configSchema", {}))
    
    async def health_check(request: Request):
        """Simple health check for GET requests."""
        return PlainTextResponse("ok")
    
    app.routes.insert(0, Route("/", health_check, methods=["GET"]))

    for route in app.routes:
        if getattr(route, "path", "") in ["/mcp", "/sse"]:
            logger.info("Aliasing /mcp endpoint to / for Smithery compatibility")
            app.routes.insert(0, Route("/", route.endpoint, methods=["POST"]))
            break

    app.routes.insert(0, Route("/.well-known/mcp/server-card.json", server_card, methods=["GET"]))
    app.routes.insert(0, Route("/.well-known/mcp-config", mcp_config, methods=["GET"]))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.add_middleware(SessionConfigMiddleware)

    def _cleanup():
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(semantic_scholar.close_client())
        except Exception:
            pass

    atexit.register(_cleanup)

    uvicorn.run(
        app,
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()