"""MCP server exposing classified database file tools."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.database_client import (
    list_accessible_files as list_accessible_database_files,
    read_classified_file as read_classified_database_file,
)

mcp = FastMCP("database")


@mcp.tool()
def list_accessible_files() -> str:
    """List classified database files the current session user may read."""
    return list_accessible_database_files()


@mcp.tool()
def read_classified_file(relative_path: str) -> str:
    """Read a classified database file after clearance and operator grant checks."""
    return read_classified_database_file(relative_path)


if __name__ == "__main__":
    mcp.run()
