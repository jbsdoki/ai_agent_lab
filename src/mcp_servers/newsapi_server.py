"""MCP server exposing NewsAPI tools."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.newsapi_client import get_top_headlines, search_news

mcp = FastMCP("newsapi")

DEFAULT_COUNTRY = "us"
DEFAULT_PAGE_SIZE = 5


def normalize_page_size(page_size: int | None) -> int:
    """Use default when the model passes null instead of omitting page_size."""
    return DEFAULT_PAGE_SIZE if page_size is None else page_size


def normalize_country(country: str | None) -> str:
    """Use default when the model passes null instead of omitting country."""
    return DEFAULT_COUNTRY if country is None else country


@mcp.tool()
def news_search(query: str, page_size: int | None = 5) -> str:
    """Search recent news articles for a keyword or topic."""
    return search_news(query, normalize_page_size(page_size))


@mcp.tool()
def news_headlines(
    country: str | None = "us",
    category: str | None = None,
    query: str | None = None,
    page_size: int | None = 5,
) -> str:
    """Get top headlines, optionally filtered by country, category, or query."""
    return get_top_headlines(
        normalize_country(country),
        category,
        query,
        normalize_page_size(page_size),
    )


if __name__ == "__main__":
    mcp.run()
