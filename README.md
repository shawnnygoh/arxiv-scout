# arXiv Scout

[![smithery badge](https://smithery.ai/badge/shawnnygoh/arxiv-scout)](https://smithery.ai/servers/shawnnygoh/arxiv-scout)

A [Model Context Protocol](https://modelcontextprotocol.io) server for searching, retrieving, and analyzing papers from [arXiv.org](https://arxiv.org) with citation data from [Semantic Scholar](https://www.semanticscholar.org/).

No API key required.

## Tools

| Tool | Description |
|------|-------------|
| `arxiv_search_papers` | Search with full query syntax, semantic categories, date filtering, pagination |
| `arxiv_get_paper` | Get metadata by arXiv ID or URL |
| `arxiv_get_papers_batch` | Fetch up to 20 papers in one call |
| `arxiv_download_and_extract` | PDF → text with section detection, header stripping, page ranges |
| `arxiv_get_references` | Outbound references via Semantic Scholar (PDF fallback) |
| `arxiv_get_citations` | Inbound citations via Semantic Scholar |
| `arxiv_list_categories` | Browse taxonomy with semantic lookup (`"machine learning"` → `cs.LG`) |
| `arxiv_cache_stats` | Cache hit rates and sizes |

## Prompts

| Prompt | Description |
|--------|-------------|
| `summarize_paper` | Guided structured summary (problem, methods, results, limitations) |
| `compare_papers` | Side-by-side comparison of 2–5 papers |
| `literature_review` | Search a topic and synthesize a review |

## Resources

| URI | Description |
|-----|-------------|
| `arxiv://categories` | Complete arXiv category taxonomy |
| `arxiv://help/query-syntax` | Query syntax reference |
| `arxiv://server/info` | Server capabilities |

## Installation

### Smithery (hosted)

No local installation needed:

```bash
npx -y @smithery/cli@latest run @shawnnygoh/arxiv-scout
```

### Local

```bash
uv pip install arxiv-scout
```

Or from source:

```bash
git clone https://github.com/shawnnygoh/arxiv-scout.git
cd arxiv-scout
uv sync
```

## Usage

### Claude Desktop

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-scout"]
    }
  }
}
```

### VS Code

```json
{
  "servers": {
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-scout"]
    }
  }
}
```

### Cursor

```json
{
  "mcpServers": {
    "arxiv": {
      "url": "https://server.smithery.ai/@shawnnygoh/arxiv-scout/mcp"
    }
  }
}
```

### Claude Code

```bash
claude mcp add --transport http arxiv https://server.smithery.ai/@shawnnygoh/arxiv-scout/mcp
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `SEMANTIC_SCHOLAR_API_KEY` | *(none)* | Optional — higher S2 rate limits |

## Development

```bash
git clone https://github.com/shawnnygoh/arxiv-scout.git
cd arxiv-scout
uv sync --dev

# Unit tests (no network)
uv run pytest

# Include integration tests (hits real APIs)
uv run pytest -m integration

# Test with MCP Inspector
uv run arxiv-scout &
npx @modelcontextprotocol/inspector
# Transport: Streamable HTTP → http://localhost:8000/mcp
```

## License

MIT