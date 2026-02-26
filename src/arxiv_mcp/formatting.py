"""Output formatting (Markdown and JSON) for all tool results."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pdf_extractor import ExtractedPaper


def format_paper_markdown(paper: dict[str, Any], verbose: bool = True) -> str:
    """Format a single paper dict as a Markdown block."""
    lines = [
        f"### {paper['title']}",
        f"**arXiv ID**: [{paper['arxiv_id']}]({paper['abs_url']})",
        f"**Authors**: {', '.join(paper['authors'][:10])}",
    ]
    if len(paper["authors"]) > 10:
        lines[-1] += f" (and {len(paper['authors']) - 10} more)"

    if paper.get("published"):
        try:
            pub_date = datetime.fromisoformat(paper["published"])
            lines.append(f"**Published**: {pub_date.strftime('%B %d, %Y')}")
        except (ValueError, TypeError):
            lines.append(f"**Published**: {paper['published']}")

    lines.append(f"**Categories**: {', '.join(paper.get('categories', []))}")

    if paper.get("doi"):
        lines.append(f"**DOI**: {paper['doi']}")
    if paper.get("journal_ref"):
        lines.append(f"**Journal**: {paper['journal_ref']}")
    if paper.get("comment"):
        lines.append(f"**Comment**: {paper['comment']}")

    lines.append(f"**PDF**: [Download]({paper['pdf_url']})")

    if verbose and paper.get("summary"):
        lines.append(f"\n**Abstract**: {paper['summary']}")

    return "\n".join(lines)


def format_search_results(
    papers: list[dict[str, Any]],
    total: int,
    query: str,
    offset: int,
    fmt: str = "markdown",
) -> str:
    """Format search results in the requested format."""
    if fmt == "json":
        return json.dumps(
            {
                "query": query,
                "total": total,
                "count": len(papers),
                "offset": offset,
                "has_more": total > offset + len(papers),
                "next_offset": (
                    offset + len(papers) if total > offset + len(papers) else None
                ),
                "papers": papers,
            },
            indent=2,
        )

    header = f"## arXiv Search Results for: *{query}*\n"
    header += f"Showing {offset + 1}–{offset + len(papers)} of {total} results\n"

    body = "\n\n---\n\n".join(
        format_paper_markdown(p, verbose=True) for p in papers
    )

    pagination = ""
    if offset + len(papers) < total:
        pagination = (
            f"\n\n---\n*More results available. "
            f"Use `offset={offset + len(papers)}` to see the next page.*"
        )

    return header + "\n" + body + pagination


def format_paper(paper: dict[str, Any], fmt: str = "markdown") -> str:
    """Format a single paper in the requested format."""
    if fmt == "json":
        return json.dumps(paper, indent=2)
    return format_paper_markdown(paper, verbose=True)


def format_batch(papers: list[dict[str, Any]], fmt: str = "markdown") -> str:
    """Format a batch of papers in the requested format."""
    if fmt == "json":
        return json.dumps({"count": len(papers), "papers": papers}, indent=2)

    lines = [f"## {len(papers)} Papers Retrieved\n"]
    for p in papers:
        lines.append(format_paper_markdown(p, verbose=False))
        lines.append("\n---\n")
    return "\n".join(lines)


def format_extracted_paper(
    paper_meta: dict[str, Any],
    extracted: ExtractedPaper,
    include_sections: bool = True,
    include_references: bool = False,
) -> str:
    """Format extracted paper text with optional section structure."""
    lines = [
        f"# {paper_meta['title']}",
        f"**arXiv ID**: {paper_meta['arxiv_id']}",
        f"**Authors**: {', '.join(paper_meta['authors'][:10])}",
        f"**Pages**: {extracted.page_count} | "
        f"**Characters**: {extracted.char_count:,}",
        "",
        "---",
        "",
    ]

    if include_sections and extracted.sections:
        lines.append("## Document Structure\n")
        for section in extracted.sections:
            indent = "  " * (section.level - 1)
            lines.append(f"{indent}- {section.title}")
        lines.append("\n---\n")

    lines.append(extracted.raw_text)

    if include_references and extracted.references:
        lines.append("\n---\n")
        lines.append("## Extracted References\n")
        for ref in extracted.references[:50]:  # Cap at 50 refs
            lines.append(f"[{ref.index}] {ref.raw_text}")

    return "\n".join(lines)


def format_references(
    paper_meta: dict[str, Any],
    references: list[dict[str, Any]],
    source: str,
    fmt: str = "markdown",
) -> str:
    """Format references (outbound) from Semantic Scholar or PDF fallback."""
    if fmt == "json":
        return json.dumps(
            {
                "arxiv_id": paper_meta["arxiv_id"],
                "title": paper_meta["title"],
                "source": source,
                "reference_count": len(references),
                "references": references,
            },
            indent=2,
        )

    if not references:
        return (
            f"No references found for '{paper_meta['title']}'. "
            "The paper may not be indexed by Semantic Scholar yet, "
            "and PDF extraction did not find structured references."
        )

    lines = [
        f"## References from: {paper_meta['title']}",
        f"**arXiv ID**: {paper_meta['arxiv_id']}",
        f"**Source**: {source.replace('_', ' ').title()}",
        f"**Total references**: {len(references)}\n",
    ]

    if source == "semantic_scholar":
        for i, ref in enumerate(references, 1):
            authors = ", ".join(ref.get("authors", [])[:3])
            if len(ref.get("authors", [])) > 3:
                authors += " et al."
            year = f" ({ref['year']})" if ref.get("year") else ""
            cites = f" [cited {ref['citation_count']}×]" if ref.get("citation_count") else ""
            arxiv_link = f" | arXiv:{ref['arxiv_id']}" if ref.get("arxiv_id") else ""
            lines.append(f"**[{i}]** {ref['title']}{year}")
            lines.append(f"  {authors}{cites}{arxiv_link}\n")
    else:
        for ref in references:
            lines.append(f"[{ref.get('index', '?')}] {ref.get('text', ref.get('raw_text', ''))}\n")

    return "\n".join(lines)


def format_citations(
    paper_meta: dict[str, Any],
    citations: list[dict[str, Any]],
    fmt: str = "markdown",
) -> str:
    """Format inbound citations from Semantic Scholar."""
    if fmt == "json":
        return json.dumps(
            {
                "arxiv_id": paper_meta["arxiv_id"],
                "title": paper_meta["title"],
                "citation_count": len(citations),
                "citations": citations,
            },
            indent=2,
        )

    if not citations:
        return (
            f"No citations found for '{paper_meta['title']}'. "
            "This paper may be too recent to have been cited, "
            "or it may not be indexed by Semantic Scholar yet."
        )

    lines = [
        f"## Papers Citing: {paper_meta['title']}",
        f"**arXiv ID**: {paper_meta['arxiv_id']}",
        f"**Total citations found**: {len(citations)}\n",
    ]

    for i, cit in enumerate(citations, 1):
        authors = ", ".join(cit.get("authors", [])[:3])
        if len(cit.get("authors", [])) > 3:
            authors += " et al."
        year = f" ({cit['year']})" if cit.get("year") else ""
        cites = f" [cited {cit['citation_count']}×]" if cit.get("citation_count") else ""
        arxiv_link = f" | arXiv:{cit['arxiv_id']}" if cit.get("arxiv_id") else ""
        lines.append(f"**[{i}]** {cit['title']}{year}")
        lines.append(f"  {authors}{cites}{arxiv_link}\n")

    return "\n".join(lines)


def format_cache_stats(
    metadata_stats: dict[str, Any],
    text_stats: dict[str, Any],
) -> str:
    """Format cache statistics as JSON."""
    return json.dumps(
        {
            "metadata_cache": metadata_stats,
            "text_cache": text_stats,
        },
        indent=2,
    )
