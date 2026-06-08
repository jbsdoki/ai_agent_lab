"""MCP server exposing yfinance stock tools."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.yfinance_client import get_stock_history, get_stock_quote

mcp = FastMCP("yfinance")


@mcp.tool()
def stock_quote(symbol: str) -> str:
    """Get the current quote and key stats for a stock ticker symbol."""
    return get_stock_quote(symbol)


@mcp.tool()
def stock_history(symbol: str, period: str = "1mo") -> str:
    """Get recent daily OHLCV history for a stock ticker symbol."""
    return get_stock_history(symbol, period)


if __name__ == "__main__":
    mcp.run()
