"""MCP server exposing long-term user memory tools."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.memory_client import (
    delete_memory as delete_user_memory,
    list_memories as list_user_memories,
    save_memory as save_user_memory,
    search_memories as search_user_memories,
)

mcp = FastMCP("memory")


@mcp.tool()
def save_memory(content: str, tags: list[str] | None = None) -> str:
    """Save a fact or preference for the current session user."""
    return save_user_memory(content, tags)


@mcp.tool()
def search_memories(query: str) -> str:
    """Search saved memories by content or tag substring."""
    return search_user_memories(query)


@mcp.tool()
def list_memories() -> str:
    """List all saved memories for the current session user."""
    return list_user_memories()


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """Delete a saved memory by id."""
    return delete_user_memory(memory_id)


if __name__ == "__main__":
    mcp.run()
