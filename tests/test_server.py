import asyncio
import json
import os
import shutil
import tempfile
import time

import pymupdf
import pytest


@pytest.fixture
def sample_paper():
    """A mock arXiv paper metadata dict."""
    return {
        "arxiv_id": "2301.07041",
        "title": "Attention Mechanisms in Transformers",
        "authors": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "summary": "We present a novel attention mechanism for transformers.",
        "published": "2023-01-17T00:00:00",
        "categories": ["cs.CL", "cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2301.07041",
        "abs_url": "https://arxiv.org/abs/2301.07041",
        "doi": "10.1234/test",
        "journal_ref": "NeurIPS 2023",
        "comment": "12 pages, 5 figures",
    }


@pytest.fixture
def sample_s2_refs():
    """Mock Semantic Scholar reference data."""
    return [
        {
            "s2_id": "abc123",
            "arxiv_id": "1706.03762",
            "doi": "10.5555/3295222.3295349",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
            "year": 2017,
            "citation_count": 95000,
            "venue": "NeurIPS",
            "abstract": "The dominant sequence transduction models...",
            "url": "https://www.semanticscholar.org/paper/abc123",
        },
        {
            "s2_id": "def456",
            "arxiv_id": "1810.04805",
            "doi": None,
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": 2019,
            "citation_count": 75000,
            "venue": "NAACL",
            "abstract": None,
            "url": None,
        },
    ]


@pytest.fixture
def test_pdf_path():
    """Create a multi-page test PDF and return its path. Cleaned up after test."""
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, "test.pdf")

    doc = pymupdf.open()
    pages_content = [
        "Running Header\n\nMy Great Paper Title\narXiv:2301.07041\n\nABSTRACT\nThis paper presents a new approach.\n\nRunning Header\n1",
        "Running Header\n\n1 Introduction\nAttention is important in NLP.\nWe build on prior work.\n\nRunning Header\n2",
        "Running Header\n\n2 Methods\nWe propose a novel architecture.\nOur model uses multi-head attention.\n\nRunning Header\n3",
        "Running Header\n\n3 Results\nOur model achieves 95% accuracy on GLUE.\nThis is state-of-the-art.\n\nRunning Header\n4",
        "Running Header\n\n4 Conclusion\nWe demonstrated strong results.\n\nREFERENCES\n[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.\n[2] Devlin et al. BERT. NAACL 2019.\n[3] Brown et al. GPT-3. NeurIPS 2020.\n[4] Radford et al. GPT-2. 2019.\n\nRunning Header\n5",
    ]

    for content in pages_content:
        page = doc.new_page()
        tw = pymupdf.TextWriter(page.rect)
        tw.append((72, 72), content, fontsize=10)
        tw.write_text(page)

    doc.save(pdf_path)
    doc.close()

    yield pdf_path

    shutil.rmtree(tmpdir)


class TestTTLCache:
    """Tests for the TTL-based LRU cache."""

    def test_basic_set_get(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        c.set("key", "value")
        assert c.get("key") == "value"

    def test_get_missing_key(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        assert c.get("nonexistent") is None

    def test_ttl_expiry(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=0.01)
        c.set("key", "value")
        time.sleep(0.02)
        assert c.get("key") is None

    def test_lru_eviction(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=3, ttl_seconds=60)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.get("a")  # Touch 'a', making 'b' the LRU
        c.set("d", 4)  # Should evict 'b'
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3
        assert c.get("d") == 4

    def test_overwrite_existing_key(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        c.set("key", "v1")
        c.set("key", "v2")
        assert c.get("key") == "v2"
        assert c.stats["size"] == 1  # No duplicate

    def test_invalidate(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        c.set("key", "value")
        c.invalidate("key")
        assert c.get("key") is None

    def test_invalidate_missing_key(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        c.invalidate("nonexistent")  # Should not raise

    def test_clear(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.stats["size"] == 0
        assert c.get("a") is None

    def test_stats_tracking(self):
        from arxiv_mcp.cache import TTLCache
        c = TTLCache(max_size=10, ttl_seconds=60)
        c.set("a", 1)
        c.get("a")      # hit
        c.get("b")      # miss
        c.get("c")      # miss
        stats = c.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 1

    def test_global_caches_exist(self):
        from arxiv_mcp.cache import metadata_cache, text_cache
        assert metadata_cache is not None
        assert text_cache is not None
        assert metadata_cache.stats["max_size"] == 500
        assert text_cache.stats["max_size"] == 100


class TestCategories:
    """Tests for category taxonomy and semantic resolution."""

    def test_exact_code_passthrough(self):
        from arxiv_mcp.categories import resolve_categories
        assert resolve_categories(["cs.AI"]) == ["cs.AI"]
        assert resolve_categories(["cs.CV", "cs.CL"]) == ["cs.CV", "cs.CL"]

    def test_semantic_alias_resolution(self):
        from arxiv_mcp.categories import resolve_categories
        result = resolve_categories(["machine learning"])
        assert "cs.LG" in result
        assert "stat.ML" in result

    def test_semantic_aliases_case_insensitive(self):
        from arxiv_mcp.categories import resolve_categories
        result = resolve_categories(["Machine Learning"])
        assert "cs.LG" in result

    def test_nlp_alias(self):
        from arxiv_mcp.categories import resolve_categories
        assert resolve_categories(["nlp"]) == ["cs.CL"]

    def test_mixed_codes_and_aliases(self):
        from arxiv_mcp.categories import resolve_categories
        result = resolve_categories(["cs.CV", "nlp", "machine learning"])
        assert "cs.CV" in result
        assert "cs.CL" in result
        assert "cs.LG" in result
        assert "stat.ML" in result

    def test_deduplication(self):
        from arxiv_mcp.categories import resolve_categories
        result = resolve_categories(["cs.LG", "machine learning"])
        assert result.count("cs.LG") == 1

    def test_unknown_passthrough(self):
        from arxiv_mcp.categories import resolve_categories
        assert resolve_categories(["unknown.XYZ"]) == ["unknown.XYZ"]

    def test_none_input(self):
        from arxiv_mcp.categories import resolve_categories
        assert resolve_categories(None) is None

    def test_empty_list(self):
        from arxiv_mcp.categories import resolve_categories
        assert resolve_categories([]) is None

    def test_flat_taxonomy_completeness(self):
        from arxiv_mcp.categories import get_flat_taxonomy, CATEGORY_TAXONOMY
        flat = get_flat_taxonomy()
        # Count total categories across all groups
        total = sum(len(cats) for cats in CATEGORY_TAXONOMY.values())
        assert len(flat) == total
        assert "cs.AI" in flat
        assert "stat.ML" in flat
        assert "math.AG" in flat

    def test_taxonomy_groups(self):
        from arxiv_mcp.categories import CATEGORY_TAXONOMY
        expected_groups = {"cs", "stat", "math", "physics", "econ", "q-bio", "q-fin", "eess"}
        assert set(CATEGORY_TAXONOMY.keys()) == expected_groups


class TestSemanticScholar:
    """Tests for Semantic Scholar client."""

    def test_strip_version_basic(self):
        from arxiv_mcp.semantic_scholar import _strip_version
        assert _strip_version("2412.19437v1") == "2412.19437"
        assert _strip_version("2412.19437v3") == "2412.19437"
        assert _strip_version("2412.19437v12") == "2412.19437"

    def test_strip_version_no_version(self):
        from arxiv_mcp.semantic_scholar import _strip_version
        assert _strip_version("2412.19437") == "2412.19437"

    def test_strip_version_old_format(self):
        from arxiv_mcp.semantic_scholar import _strip_version
        assert _strip_version("hep-th/9901001v2") == "hep-th/9901001"

    def test_normalize_paper_full(self):
        from arxiv_mcp.semantic_scholar import _normalize_paper
        raw = {
            "paperId": "abc123",
            "externalIds": {"ArXiv": "2301.07041", "DOI": "10.1234/test"},
            "title": "Test Paper",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2023,
            "citationCount": 42,
            "abstract": "Abstract text.",
            "url": "https://semanticscholar.org/paper/abc123",
            "venue": "NeurIPS",
        }
        result = _normalize_paper(raw)
        assert result["s2_id"] == "abc123"
        assert result["arxiv_id"] == "2301.07041"
        assert result["doi"] == "10.1234/test"
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["year"] == 2023
        assert result["citation_count"] == 42
        assert result["venue"] == "NeurIPS"

    def test_normalize_paper_minimal(self):
        from arxiv_mcp.semantic_scholar import _normalize_paper
        result = _normalize_paper({"paperId": "xyz"})
        assert result["s2_id"] == "xyz"
        assert result["title"] == "Unknown Title"
        assert result["authors"] == []
        assert result["arxiv_id"] is None
        assert result["year"] is None
        assert result["venue"] is None

    def test_normalize_paper_empty_venue(self):
        from arxiv_mcp.semantic_scholar import _normalize_paper
        result = _normalize_paper({"paperId": "x", "venue": ""})
        assert result["venue"] is None  # Empty string normalized to None

    def test_client_initialization(self):
        from arxiv_mcp.semantic_scholar import _get_client
        client = _get_client()
        assert client is not None
        assert not client.is_closed

    def test_async_function_signatures(self):
        import inspect
        from arxiv_mcp.semantic_scholar import get_references, get_citations, get_paper_metadata
        assert inspect.iscoroutinefunction(get_references)
        assert inspect.iscoroutinefunction(get_citations)
        assert inspect.iscoroutinefunction(get_paper_metadata)

    @pytest.mark.integration
    def test_real_s2_references(self):
        """Integration test: fetch real references for 'Attention Is All You Need'."""
        from arxiv_mcp.semantic_scholar import get_references
        refs = asyncio.run(
            get_references("1706.03762")
        )
        assert refs is not None
        assert len(refs) > 10
        # Should find some well-known references
        titles = [r["title"].lower() for r in refs]
        assert any("sequence" in t or "neural" in t for t in titles)

    @pytest.mark.integration
    def test_real_s2_citations(self):
        """Integration test: fetch real citations for 'Attention Is All You Need'."""
        from arxiv_mcp.semantic_scholar import get_citations
        cites = asyncio.run(
            get_citations("1706.03762", limit=10)
        )
        assert cites is not None
        assert len(cites) > 0

    @pytest.mark.integration
    def test_real_s2_versioned_id(self):
        """Integration test: versioned ID should work after stripping."""
        from arxiv_mcp.semantic_scholar import get_references
        refs = asyncio.run(
            get_references("1706.03762v1")
        )
        # Should not return None due to 404 — version should be stripped
        assert refs is not None


class TestPDFExtractor:
    """Tests for PDF text extraction and reference parsing."""

    def test_section_detection_numbered(self):
        from arxiv_mcp.pdf_extractor import _detect_sections

        text = "Preamble text\n\n1 Introduction\nIntro content.\n\n2 Methods\nMethod content.\n\n3 Conclusion\nConclusion content."
        sections = _detect_sections(text)
        titles = [s.title for s in sections]
        assert "Preamble" in titles
        assert "1 Introduction" in titles
        assert "2 Methods" in titles
        assert "3 Conclusion" in titles

    def test_section_detection_allcaps(self):
        from arxiv_mcp.pdf_extractor import _detect_sections

        text = "Title\n\nABSTRACT\nAbstract text.\n\nINTRODUCTION\nIntro text.\n\nMETHODS\nMethods text.\n\nREFERENCES\nRef text."
        sections = _detect_sections(text)
        titles = [s.title for s in sections]
        assert "ABSTRACT" in titles
        assert "INTRODUCTION" in titles
        assert "REFERENCES" in titles

    def test_section_level_detection(self):
        from arxiv_mcp.pdf_extractor import _detect_sections

        text = "1 Introduction\nContent.\n\n2.1 Related Work\nSub content."
        sections = _detect_sections(text)
        # Section 1 should be level 1, Section 2.1 should be level 2
        intro = [s for s in sections if "Introduction" in s.title]
        assert len(intro) == 1

    def test_numbered_references_standard(self):
        from arxiv_mcp.pdf_extractor import _extract_numbered_references

        content = """[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.
[2] Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. 2019.
[3] Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020.
[4] Radford et al. GPT-2. 2019.
[5] Raffel et al. T5. JMLR 2020."""
        refs = _extract_numbered_references(content)
        assert len(refs) == 5
        assert refs[0].index == "1"
        assert "Vaswani" in refs[0].raw_text

    def test_numbered_references_multiline(self):
        from arxiv_mcp.pdf_extractor import _extract_numbered_references

        content = """[1] Vaswani, A., Shazeer, N., Parmar, N.
Attention Is All You Need. NeurIPS 2017.
[2] Devlin, J., Chang, M.
BERT. NAACL 2019.
[3] Brown, T., Mann, B.
GPT-3. NeurIPS 2020."""
        refs = _extract_numbered_references(content)
        assert len(refs) == 3
        # Multi-line entries should be joined
        assert "Attention Is All You Need" in refs[0].raw_text

    def test_numbered_references_rejects_years(self):
        """The DeepSeek bug: [2018] and [2378] are years/page numbers, not ref indices."""
        from arxiv_mcp.pdf_extractor import _extract_numbered_references

        content = """[2018] URL http://arxiv.org/abs/1803.05457.
Some continuation text about a paper.
[2378] Association for Computational Linguistics, 2019.
More continuation text here."""
        refs = _extract_numbered_references(content)
        assert len(refs) == 0, f"Should reject [2018]/[2378], got {len(refs)} refs"

    def test_numbered_references_dot_style(self):
        from arxiv_mcp.pdf_extractor import _extract_numbered_references

        content = """1. Vaswani et al. Attention Is All You Need. 2017.
2. Devlin et al. BERT. 2019.
3. Brown et al. GPT-3. 2020.
4. Radford et al. GPT-2. 2019."""
        refs = _extract_numbered_references(content)
        assert len(refs) == 4
        assert refs[0].index == "1"

    def test_numbered_references_minimum_count(self):
        """Need at least 3 refs to be confident — reject tiny matches."""
        from arxiv_mcp.pdf_extractor import _extract_numbered_references

        content = "[1] Single lonely reference."
        refs = _extract_numbered_references(content)
        assert len(refs) == 0

    def test_author_year_references(self):
        from arxiv_mcp.pdf_extractor import _extract_author_year_references

        content = """A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones. Attention is all you need. NeurIPS 2017.
J. Devlin, M. Chang, K. Lee, K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.
T. Brown, B. Mann, N. Ryder, M. Subbiah. Language Models are Few-Shot Learners. NeurIPS 2020.
A. Radford, J. Wu, R. Child, D. Luan. Language Models are Unsupervised Multitask Learners. 2019.
C. Raffel, N. Shazeer, A. Roberts. Exploring the Limits of Transfer Learning. JMLR 2020.
K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016."""
        refs = _extract_author_year_references(content)
        assert len(refs) >= 5
        assert "Vaswani" in refs[0].raw_text

    def test_author_year_references_with_org_names(self):
        from arxiv_mcp.pdf_extractor import _extract_author_year_references

        content = """A. Vaswani, N. Shazeer. Attention is all you need. 2017.
DeepSeek-AI. DeepSeek LLM: scaling open-source models. 2024.
OpenAI. GPT-4 Technical Report. 2023.
B. Brown, T. Mann. GPT-3 paper. 2020.
NVIDIA. TransformerEngine documentation. 2024.
J. Devlin, M. Chang. BERT paper. 2019."""
        refs = _extract_author_year_references(content)
        assert len(refs) >= 5

    def test_raw_fallback_references(self):
        from arxiv_mcp.pdf_extractor import _extract_raw_references

        content = """First reference entry that is long enough to be a reference.

Second reference entry that is also long enough to be valid.

Third reference entry with sufficient length to matter.

Fourth reference that completes our set of bibliography items."""
        refs = _extract_raw_references(content)
        assert len(refs) >= 3

    def test_extraction_cascade(self):
        """_extract_references should try numbered → author-year → raw."""
        from arxiv_mcp.pdf_extractor import _extract_references, Section

        # Numbered refs: should use numbered extractor
        numbered_section = Section(
            title="REFERENCES",
            content="[1] Ref one text here.\n[2] Ref two text here.\n[3] Ref three text.\n[4] Ref four.",
            level=1,
        )
        refs = _extract_references([numbered_section])
        assert len(refs) == 4
        assert refs[0].index == "1"

    def test_page_range_extraction(self, test_pdf_path):
        from arxiv_mcp.pdf_extractor import extract_paper_text

        # Full extraction
        full = extract_paper_text(test_pdf_path, max_characters=100000, smart=False)
        assert full.page_count == 5

        # Pages 2-3 only
        partial = extract_paper_text(
            test_pdf_path, max_characters=100000, start_page=2, end_page=3, smart=False
        )
        assert partial.page_count == 5  # Total pages in PDF unchanged
        # Should have content from pages 2-3 but not 1, 4, 5
        assert "Page 2" in partial.raw_text
        assert "Page 3" in partial.raw_text
        assert "Page 1" not in partial.raw_text
        assert "Page 4" not in partial.raw_text

    def test_single_page_extraction(self, test_pdf_path):
        from arxiv_mcp.pdf_extractor import extract_paper_text

        result = extract_paper_text(
            test_pdf_path, max_characters=100000, start_page=3, end_page=3, smart=False
        )
        assert "Page 3" in result.raw_text
        assert "Page 2" not in result.raw_text
        assert "Page 4" not in result.raw_text

    def test_max_characters_truncation(self, test_pdf_path):
        from arxiv_mcp.pdf_extractor import extract_paper_text

        result = extract_paper_text(test_pdf_path, max_characters=100, smart=False)
        assert "Truncated" in result.raw_text
        assert result.char_count > 100

    def test_header_footer_stripping(self):
        from arxiv_mcp.pdf_extractor import _strip_headers_footers

        pages = [
            "Running Header\nContent page 1\nPage Footer\n1",
            "Running Header\nContent page 2\nPage Footer\n2",
            "Running Header\nContent page 3\nPage Footer\n3",
            "Running Header\nContent page 4\nPage Footer\n4",
        ]
        cleaned = _strip_headers_footers(pages, threshold=0.6)
        for page in cleaned:
            assert "Running Header" not in page
            assert "Page Footer" not in page

    def test_header_stripping_short_document(self):
        """Documents with < 3 pages should not be stripped."""
        from arxiv_mcp.pdf_extractor import _strip_headers_footers

        pages = ["Header\nContent 1", "Header\nContent 2"]
        cleaned = _strip_headers_footers(pages)
        assert cleaned == pages  # Unchanged


class TestFormatting:
    """Tests for output formatting functions."""

    def test_format_paper_markdown(self, sample_paper):
        from arxiv_mcp.formatting import format_paper
        result = format_paper(sample_paper, "markdown")
        assert "### Attention Mechanisms" in result
        assert "2301.07041" in result
        assert "Alice Smith" in result
        assert "cs.CL" in result
        assert "NeurIPS 2023" in result

    def test_format_paper_json(self, sample_paper):
        from arxiv_mcp.formatting import format_paper
        result = format_paper(sample_paper, "json")
        parsed = json.loads(result)
        assert parsed["arxiv_id"] == "2301.07041"
        assert parsed["title"] == "Attention Mechanisms in Transformers"

    def test_format_search_results_markdown(self, sample_paper):
        from arxiv_mcp.formatting import format_search_results
        result = format_search_results([sample_paper], 1, "test query", 0, "markdown")
        assert "arXiv Search Results" in result
        assert "test query" in result
        assert "1–1 of 1" in result

    def test_format_search_results_pagination(self, sample_paper):
        from arxiv_mcp.formatting import format_search_results
        result = format_search_results([sample_paper], 50, "test", 0, "markdown")
        assert "offset=1" in result  # Should suggest next page

    def test_format_search_results_json(self, sample_paper):
        from arxiv_mcp.formatting import format_search_results
        result = format_search_results([sample_paper], 10, "q", 0, "json")
        parsed = json.loads(result)
        assert parsed["total"] == 10
        assert parsed["has_more"] is True
        assert parsed["next_offset"] == 1

    def test_format_batch(self, sample_paper):
        from arxiv_mcp.formatting import format_batch
        result = format_batch([sample_paper, sample_paper], "markdown")
        assert "2 Papers Retrieved" in result

    def test_format_references_s2(self, sample_paper, sample_s2_refs):
        from arxiv_mcp.formatting import format_references
        result = format_references(sample_paper, sample_s2_refs, "semantic_scholar", "markdown")
        assert "Semantic Scholar" in result
        assert "Attention Is All You Need" in result
        assert "et al." in result  # 4 authors truncated
        assert "cited 95000" in result

    def test_format_references_s2_json(self, sample_paper, sample_s2_refs):
        from arxiv_mcp.formatting import format_references
        result = format_references(sample_paper, sample_s2_refs, "semantic_scholar", "json")
        parsed = json.loads(result)
        assert parsed["source"] == "semantic_scholar"
        assert parsed["reference_count"] == 2

    def test_format_references_pdf_fallback(self, sample_paper):
        from arxiv_mcp.formatting import format_references
        pdf_refs = [
            {"index": "1", "text": "Author A. Some paper. 2020."},
            {"index": "2", "text": "Author B. Another paper. 2021."},
        ]
        result = format_references(sample_paper, pdf_refs, "pdf_extraction", "markdown")
        assert "Pdf Extraction" in result
        assert "Author A." in result

    def test_format_references_empty(self, sample_paper):
        from arxiv_mcp.formatting import format_references
        result = format_references(sample_paper, [], "semantic_scholar", "markdown")
        assert "No references found" in result

    def test_format_citations(self, sample_paper, sample_s2_refs):
        from arxiv_mcp.formatting import format_citations
        result = format_citations(sample_paper, sample_s2_refs, "markdown")
        assert "Papers Citing" in result
        assert "Attention Is All You Need" in result

    def test_format_citations_json(self, sample_paper, sample_s2_refs):
        from arxiv_mcp.formatting import format_citations
        result = format_citations(sample_paper, sample_s2_refs, "json")
        parsed = json.loads(result)
        assert parsed["citation_count"] == 2

    def test_format_citations_empty(self, sample_paper):
        from arxiv_mcp.formatting import format_citations
        result = format_citations(sample_paper, [], "markdown")
        assert "No citations found" in result

    def test_format_extracted_paper(self, sample_paper):
        from arxiv_mcp.pdf_extractor import ExtractedPaper, Section, Reference
        from arxiv_mcp.formatting import format_extracted_paper

        extracted = ExtractedPaper(
            title="Paper Title",
            raw_text="Full text content here...",
            sections=[Section(title="1 Introduction", content="Intro", level=1)],
            references=[Reference(index="1", raw_text="Ref text")],
            page_count=10,
            char_count=5000,
        )
        result = format_extracted_paper(sample_paper, extracted, True, True)
        assert "Attention Mechanisms" in result
        assert "Document Structure" in result
        assert "1 Introduction" in result
        assert "Extracted References" in result

    def test_format_cache_stats(self):
        from arxiv_mcp.formatting import format_cache_stats
        result = format_cache_stats(
            {"hits": 5, "misses": 2, "size": 3},
            {"hits": 1, "misses": 0, "size": 1},
        )
        parsed = json.loads(result)
        assert parsed["metadata_cache"]["hits"] == 5
        assert parsed["text_cache"]["size"] == 1

    def test_format_paper_many_authors(self):
        """Papers with > 10 authors should be truncated."""
        from arxiv_mcp.formatting import format_paper_markdown
        paper = {
            "arxiv_id": "test",
            "title": "Many Authors Paper",
            "authors": [f"Author {i}" for i in range(20)],
            "summary": "Abstract",
            "published": "2024-01-01",
            "categories": ["cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/test",
            "abs_url": "https://arxiv.org/abs/test",
            "doi": None,
            "journal_ref": None,
            "comment": None,
        }
        result = format_paper_markdown(paper)
        assert "and 10 more" in result


class TestServerUtils:
    """Tests for server-level utility functions."""

    def test_normalize_arxiv_id_plain(self):
        from arxiv_mcp.server import _normalize_arxiv_id
        assert _normalize_arxiv_id("2301.07041") == "2301.07041"

    def test_normalize_arxiv_id_versioned(self):
        from arxiv_mcp.server import _normalize_arxiv_id
        assert _normalize_arxiv_id("2301.07041v2") == "2301.07041v2"

    def test_normalize_arxiv_id_abs_url(self):
        from arxiv_mcp.server import _normalize_arxiv_id
        assert _normalize_arxiv_id("https://arxiv.org/abs/2301.07041") == "2301.07041"

    def test_normalize_arxiv_id_pdf_url(self):
        from arxiv_mcp.server import _normalize_arxiv_id
        assert _normalize_arxiv_id("https://arxiv.org/pdf/2301.07041.pdf") == "2301.07041"

    def test_normalize_arxiv_id_http(self):
        from arxiv_mcp.server import _normalize_arxiv_id
        assert _normalize_arxiv_id("http://arxiv.org/abs/2301.07041v3") == "2301.07041v3"

    def test_normalize_arxiv_id_whitespace(self):
        from arxiv_mcp.server import _normalize_arxiv_id
        assert _normalize_arxiv_id("  2301.07041  ") == "2301.07041"


class TestMCPRegistration:
    """Tests that all MCP tools, prompts, and resources are correctly registered."""

    def test_tool_count(self):
        from arxiv_mcp.server import mcp
        tools = mcp._tool_manager._tools
        assert len(tools) == 8

    def test_expected_tools(self):
        from arxiv_mcp.server import mcp
        tools = set(mcp._tool_manager._tools.keys())
        expected = {
            "arxiv_search_papers", "arxiv_get_paper", "arxiv_get_papers_batch",
            "arxiv_download_and_extract", "arxiv_get_references", "arxiv_get_citations",
            "arxiv_list_categories", "arxiv_cache_stats",
        }
        assert tools == expected

    def test_prompt_count(self):
        from arxiv_mcp.server import mcp
        prompts = mcp._prompt_manager._prompts
        assert len(prompts) == 3

    def test_expected_prompts(self):
        from arxiv_mcp.server import mcp
        prompts = set(mcp._prompt_manager._prompts.keys())
        expected = {"summarize_paper", "compare_papers", "literature_review"}
        assert prompts == expected

    def test_resource_count(self):
        from arxiv_mcp.server import mcp
        resources = mcp._resource_manager._resources
        assert len(resources) == 3

    def test_expected_resources(self):
        from arxiv_mcp.server import mcp
        resources = set(mcp._resource_manager._resources.keys())
        expected = {"arxiv://categories", "arxiv://help/query-syntax", "arxiv://server/info"}
        assert resources == expected

    def test_schemas_are_flat(self):
        """No tool should have a nested 'params' or 'ctx' in its schema."""
        from arxiv_mcp.server import mcp
        for name, tool in mcp._tool_manager._tools.items():
            props = tool.parameters.get("properties", {})
            assert "params" not in props, f"{name}: nested 'params' wrapper"
            assert "ctx" not in props, f"{name}: 'ctx' leaked into schema"

    def test_search_tool_params(self):
        from arxiv_mcp.server import mcp
        tool = mcp._tool_manager._tools["arxiv_search_papers"]
        props = set(tool.parameters.get("properties", {}).keys())
        expected = {"query", "max_results", "categories", "sort_by", "date_from", "date_to", "offset", "output_format"}
        assert props == expected

    def test_download_tool_has_page_range(self):
        from arxiv_mcp.server import mcp
        tool = mcp._tool_manager._tools["arxiv_download_and_extract"]
        props = tool.parameters.get("properties", {})
        assert "start_page" in props
        assert "end_page" in props
        assert "max_characters" in props

    def test_citations_tool_params(self):
        from arxiv_mcp.server import mcp
        tool = mcp._tool_manager._tools["arxiv_get_citations"]
        props = set(tool.parameters.get("properties", {}).keys())
        assert props == {"arxiv_id", "limit", "output_format"}

    def test_references_tool_params(self):
        from arxiv_mcp.server import mcp
        tool = mcp._tool_manager._tools["arxiv_get_references"]
        props = set(tool.parameters.get("properties", {}).keys())
        assert props == {"arxiv_id", "limit", "output_format"}


@pytest.mark.integration
class TestIntegration:
    """Integration tests that hit real APIs. Run with: pytest -m integration"""

    def test_real_search(self):
        from arxiv_mcp.arxiv_client import search_papers
        papers, total = asyncio.run(
            search_papers("attention mechanism", max_results=3)
        )
        assert len(papers) > 0
        assert total > 0
        assert "title" in papers[0]
        assert "arxiv_id" in papers[0]

    def test_real_get_paper(self):
        from arxiv_mcp.arxiv_client import get_paper
        paper = asyncio.run(
            get_paper("1706.03762")
        )
        assert "attention" in paper["title"].lower()
        assert len(paper["authors"]) > 0

    def test_real_download_extract(self):
        from arxiv_mcp.arxiv_client import download_and_extract
        extracted = asyncio.run(
            download_and_extract("1706.03762", max_characters=10000)
        )
        assert extracted.page_count > 0
        assert extracted.char_count > 0
        assert len(extracted.raw_text) > 100

    def test_real_s2_references_attention_v1(self):
        from arxiv_mcp.semantic_scholar import get_references
        refs = asyncio.run(
            get_references("1706.03762v1")
        )
        # After fix, should strip v1 and succeed
        assert refs is not None
        assert len(refs) > 10

    def test_real_s2_citations_attention(self):
        from arxiv_mcp.semantic_scholar import get_citations
        cites = asyncio.run(
            get_citations("1706.03762", limit=5)
        )
        assert cites is not None
        assert len(cites) >= 1