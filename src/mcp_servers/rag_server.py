"""MCP server exposing document search tools over the local RAG index."""

from mcp.server.fastmcp import FastMCP

from src.data_retrieval.rag_client import (
    list_indexed_sources as list_rag_indexed_sources,
    search_documents as search_rag_documents,
)

mcp = FastMCP("rag")


@mcp.tool()
def search_documents(query: str, max_results: int = 5) -> str:
    """Search the standard-safe document corpus for relevant passages."""
    return search_rag_documents(query, max_results)


@mcp.tool()
def list_indexed_sources() -> str:
    """List documents currently indexed in the RAG vector store."""
    return list_rag_indexed_sources()


if __name__ == "__main__":
    mcp.run()
