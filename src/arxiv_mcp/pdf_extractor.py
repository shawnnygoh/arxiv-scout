"""Smart PDF text extraction with section detection and page ranges."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

SECTION_PATTERNS = [
    # Numbered sections: "1. Introduction", "2.1 Related Work"
    re.compile(
        r"^(\d+\.?\d*\.?\d*)\s+"
        r"(Abstract|Introduction|Background|Related\s+Work|Preliminaries|"
        r"Problem\s+(Statement|Formulation|Definition)|Methodology|Method|Methods|"
        r"Approach|Model|Architecture|Framework|System|"
        r"Experiments?|Evaluation|Results?|Analysis|"
        r"Discussion|Ablation|Limitations?|"
        r"Conclusion|Conclusions|Summary|Future\s+Work|"
        r"Acknowledgm?ents?|References|Bibliography|Appendix)",
        re.IGNORECASE,
    ),
    # Unnumbered sections (all caps, standalone line)
    re.compile(
        r"^(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED\s+WORK|PRELIMINARIES|"
        r"METHODOLOGY|METHODS?|APPROACH|MODEL|ARCHITECTURE|"
        r"EXPERIMENTS?|EVALUATION|RESULTS?|ANALYSIS|"
        r"DISCUSSION|ABLATION|LIMITATIONS?|"
        r"CONCLUSIONS?|SUMMARY|FUTURE\s+WORK|"
        r"ACKNOWLEDGM?ENTS?|REFERENCES|BIBLIOGRAPHY|APPENDIX)\s*$",
        re.IGNORECASE,
    ),
]

REFERENCE_PATTERNS = [
    # [1] Author et al., "Title", Journal, Year.
    # Only match indices 1-999 to avoid years like [2018] or page numbers
    re.compile(r"^\[(\d{1,3})\]\s*(.+)$"),
    # 1. Author et al. (Year). Title.
    # Require uppercase start to avoid matching list items
    re.compile(r"^(\d{1,3})\.\s+([A-Z].+)$"),
]


@dataclass
class Section:
    """A detected section of the paper."""

    title: str
    content: str
    level: int = 1


@dataclass
class Reference:
    """A parsed reference entry."""

    index: str
    raw_text: str


@dataclass
class ExtractedPaper:
    """Result of smart PDF extraction."""

    title: str
    raw_text: str
    sections: list[Section] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    page_count: int = 0
    char_count: int = 0


def _extract_page_texts(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    doc.close()
    return pages


def _strip_headers_footers(pages: list[str], threshold: float = 0.6) -> list[str]:
    """Remove repeated header/footer lines that appear on many pages.

    Lines appearing on more than `threshold` fraction of pages are likely
    headers or footers (page numbers, journal names, running titles).
    """
    if len(pages) < 3:
        return pages

    # Count first and last 3 lines of each page
    line_counts: Counter = Counter()
    for page_text in pages:
        lines = page_text.strip().split("\n")
        candidates = lines[:3] + lines[-3:]
        for line in candidates:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                line_counts[stripped] += 1

    # Lines appearing on > threshold of pages are headers/footers
    min_count = int(len(pages) * threshold)
    noise_lines = {line for line, count in line_counts.items() if count >= min_count}

    # Also remove standalone page numbers
    noise_lines |= {str(i) for i in range(1, len(pages) + 5)}

    cleaned = []
    for page_text in pages:
        lines = page_text.split("\n")
        filtered = [line for line in lines if line.strip() not in noise_lines]
        cleaned.append("\n".join(filtered))

    return cleaned


def _detect_sections(text: str) -> list[Section]:
    """Detect section boundaries in the extracted text."""
    lines = text.split("\n")
    sections: list[Section] = []
    current_title = "Preamble"
    current_lines: list[str] = []
    current_level = 1

    for line in lines:
        stripped = line.strip()
        is_heading = False

        for pattern in SECTION_PATTERNS:
            match = pattern.match(stripped)
            if match:
                # Save the previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append(Section(
                            title=current_title,
                            content=content,
                            level=current_level,
                        ))

                # Start new section
                current_title = stripped
                current_lines = []
                if match.group(1) and "." in str(match.group(1)):
                    current_level = str(match.group(1)).count(".") + 1
                else:
                    current_level = 1
                is_heading = True
                break

        if not is_heading:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(Section(
                title=current_title,
                content=content,
                level=current_level,
            ))

    return sections


def _extract_references(sections: list[Section]) -> list[Reference]:
    """Extract structured references from the References section.

    Strategy (in order of preference):
    1. Numbered references: [1], [2], ... or 1., 2., ...
    2. Author-year format: lines starting with author initials
    3. Raw fallback: paragraph-split entries
    """
    refs_section = None
    for section in sections:
        if re.match(r"^(\d+\.?\s*)?(REFERENCES|BIBLIOGRAPHY)", section.title, re.IGNORECASE):
            refs_section = section
            break

    if not refs_section:
        return []

    numbered_refs = _extract_numbered_references(refs_section.content)
    if numbered_refs:
        return numbered_refs

    author_year_refs = _extract_author_year_references(refs_section.content)
    if author_year_refs:
        return author_year_refs

    return _extract_raw_references(refs_section.content)


def _extract_numbered_references(content: str) -> list[Reference]:
    """Extract references with numbered indices like [1] or 1.

    Validates that the first detected index is <= 5 and that we find at
    least 3 references to avoid false positives.
    """
    lines = content.split("\n")
    current_ref: Optional[list[str]] = None
    current_index = ""
    references = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for pattern in REFERENCE_PATTERNS:
            match = pattern.match(stripped)
            if match:
                idx_num = int(match.group(1))
                # Reject if index is suspiciously large for a first match
                if not references and current_ref is None and idx_num > 5:
                    continue

                # Save previous reference
                if current_ref is not None:
                    references.append(Reference(
                        index=current_index,
                        raw_text=" ".join(current_ref),
                    ))
                current_index = match.group(1)
                current_ref = [match.group(2)]
                matched = True
                break

        if not matched and current_ref is not None:
            current_ref.append(stripped)

    if current_ref is not None:
        references.append(Reference(
            index=current_index,
            raw_text=" ".join(current_ref),
        ))

    if len(references) < 3:
        return []

    return references


def _extract_author_year_references(content: str) -> list[Reference]:
    """Extract references in author-year format (no numbered indices).

    Detects entries by lines starting with author initials or organization
    names. Groups continuation lines with their parent entry.
    """
    author_start = re.compile(
        r"^(?:"
        r"[A-Z](?:\.-?[A-Z])*\.\s+"
        r"|[A-Z][A-Za-z]+(?:-[A-Za-z]+)*\.\s+"
        r"|[A-Z]{2,}(?:-[A-Za-z]+)*\.\s+"
        r"|[A-Z][a-z]+,\s+"
        r")"
    )

    lines = content.split("\n")
    entries: list[list[str]] = []
    current_entry: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_entry:
                entries.append(current_entry)
                current_entry = []
            continue

        if author_start.match(stripped) and current_entry:
            entries.append(current_entry)
            current_entry = [stripped]
        else:
            current_entry.append(stripped)

    if current_entry:
        entries.append(current_entry)

    references = []
    for i, entry in enumerate(entries):
        text = " ".join(entry)
        if text.startswith("--- Page") or len(text) < 30:
            continue
        references.append(Reference(
            index=str(i + 1),
            raw_text=text,
        ))

    if len(references) < 5:
        return []

    return references


def _extract_raw_references(content: str) -> list[Reference]:
    """Last-resort fallback: split references by blank lines or page breaks."""
    chunks = re.split(r"\n\s*\n|--- Page \d+ ---", content)

    references = []
    for i, chunk in enumerate(chunks):
        text = chunk.strip()
        if len(text) < 30:
            continue
        references.append(Reference(
            index=str(i + 1),
            raw_text=text,
        ))

    return references


def extract_paper_text(
    pdf_path: str,
    max_characters: int = 50000,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    smart: bool = True,
) -> ExtractedPaper:
    """Extract text from a PDF with optional smart processing and page ranges.

    Args:
        pdf_path: Path to the PDF file.
        max_characters: Maximum characters for the raw text output.
        start_page: 1-based start page (inclusive). None = first page.
        end_page: 1-based end page (inclusive). None = last page.
        smart: If True, apply header/footer stripping and section detection.

    Returns:
        ExtractedPaper with raw text, detected sections, and references.
    """
    all_pages = _extract_page_texts(pdf_path)
    total_page_count = len(all_pages)

    # Apply page range (1-based, inclusive)
    p_start = max(0, (start_page or 1) - 1)
    p_end = min(total_page_count, end_page or total_page_count)
    pages = all_pages[p_start:p_end]
    page_count = len(pages)

    if smart and page_count >= 3:
        pages = _strip_headers_footers(pages)

    # Build raw text with page markers (using original page numbers)
    raw_parts = []
    for i, page_text in enumerate(pages):
        if page_text.strip():
            original_page_num = p_start + i + 1
            raw_parts.append(f"--- Page {original_page_num} ---\n{page_text}")

    raw_text = "\n\n".join(raw_parts)
    full_char_count = len(raw_text)

    sections = _detect_sections(raw_text) if smart and page_count >= 2 else []
    references = _extract_references(sections) if smart else []

    # Truncate if needed
    if len(raw_text) > max_characters:
        raw_text = raw_text[:max_characters] + (
            f"\n\n[... Truncated at {max_characters:,} characters. "
            f"Total: {full_char_count:,} characters. "
            "Increase max_characters for full content.]"
        )

    # Try to extract title from preamble
    title = ""
    if sections and sections[0].title == "Preamble":
        first_lines = sections[0].content.split("\n")
        for line in first_lines[:5]:
            if len(line.strip()) > 10 and not line.strip().startswith("arXiv"):
                title = line.strip()
                break

    return ExtractedPaper(
        title=title,
        raw_text=raw_text,
        sections=sections,
        references=references,
        page_count=total_page_count,
        char_count=full_char_count,
    )
