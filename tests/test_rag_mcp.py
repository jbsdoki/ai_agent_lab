"""Smoke tests for the RAG MCP server."""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import SESSION_LOG_PATH_ENV
from src.data_retrieval.rag_client import (
    CHROMA_PERSIST_DIR_ENV,
    ingest_corpus,
    search_documents,
)

LOGS_DIR = PROJECT_ROOT / "logs"
RAG_MCP_TOOLS = ["search_documents", "list_indexed_sources"]


def build_mcp_client(project_root: Path, chroma_dir: Path, log_path: Path) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "rag": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.rag_server"],
                "cwd": str(project_root),
                "env": {
                    **os.environ,
                    CHROMA_PERSIST_DIR_ENV: str(chroma_dir),
                    SESSION_LOG_PATH_ENV: str(log_path),
                },
            }
        }
    )


async def fetch_tool_names(client: MultiServerMCPClient) -> list[str]:
    tools = await client.get_tools()
    return [tool.name for tool in tools]


def run_client_search_test(chroma_dir: Path, embeddings: FakeEmbeddings) -> tuple[bool, list[str]]:
    os.environ[CHROMA_PERSIST_DIR_ENV] = str(chroma_dir)
    ingest_corpus(embeddings=embeddings, force=True)
    payload = json.loads(search_documents("RAG-FAQ-BETA", max_results=10, embeddings=embeddings))
    failures: list[str] = []
    if not any("RAG-FAQ-BETA" in hit["content"] for hit in payload.get("results", [])):
        failures.append("search_documents returned no FAQ hits")
    return not failures, failures


async def run_test() -> tuple[str, bool]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    chroma_dir = LOGS_DIR / f"rag_mcp_{timestamp}"
    log_path = LOGS_DIR / f"rag_mcp_{timestamp}.logs"
    embeddings = FakeEmbeddings(size=32)

    client = build_mcp_client(PROJECT_ROOT, chroma_dir, log_path)
    tool_names = await fetch_tool_names(client)
    missing = [name for name in RAG_MCP_TOOLS if name not in tool_names]
    client_ok, client_failures = run_client_search_test(chroma_dir, embeddings)

    lines = [
        "RAG MCP smoke test",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Tools: {tool_names}",
        f"Registration: {'PASS' if not missing else 'FAIL'} missing={missing}",
        f"Client search: {'PASS' if client_ok else 'FAIL'}",
    ]
    if client_failures:
        lines.extend(f"  - {entry}" for entry in client_failures)
    overall = not missing and client_ok
    lines.append(f"Result: {'PASS' if overall else 'FAIL'}")
    return "\n".join(lines), overall


if __name__ == "__main__":
    output, passed = asyncio.run(run_test())
    print(output)
    if not passed:
        raise SystemExit(1)
