"""MCP server exposing allowlisted web fetch tools."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.web_client import fetch_url as fetch_allowlisted_url

mcp = FastMCP("web")


@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch and extract readable text from an allowlisted HTTPS URL."""
    return fetch_allowlisted_url(url)


if __name__ == "__main__":
    mcp.run()
