"""MCP server exposing SEC EDGAR tools."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.sec_client import get_recent_filings, lookup_cik

mcp = FastMCP("sec")

DEFAULT_FILING_LIMIT = 5


def normalize_form_type(form_type: str | None) -> str | None:
    if form_type is None:
        return None
    cleaned = form_type.strip()
    return cleaned or None


def normalize_limit(limit: int | None) -> int:
    return DEFAULT_FILING_LIMIT if limit is None else limit


@mcp.tool()
def sec_lookup_cik(symbol: str) -> str:
    """Resolve a US stock ticker symbol to its SEC CIK."""
    return lookup_cik(symbol)


@mcp.tool()
def sec_recent_filings(
    symbol: str,
    form_type: str | None = None,
    limit: int | None = 5,
) -> str:
    """List recent SEC filings for a ticker, optionally filtered by form type."""
    return get_recent_filings(
        symbol,
        normalize_form_type(form_type),
        normalize_limit(limit),
    )


if __name__ == "__main__":
    mcp.run()
