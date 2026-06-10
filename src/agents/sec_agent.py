"""Local Ollama agent with SEC EDGAR MCP tools."""

import asyncio
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    build_llm,
    get_project_root,
    interactive_loop,
)

SYSTEM_PROMPT = (
    "You are an SEC filings assistant. Use the available SEC tools to answer "
    "questions about EDGAR filings such as 10-K, 10-Q, and 8-K. Always call a "
    "tool for filing metadata instead of guessing dates or accession numbers. "
    "When the user names a company, infer the ticker if you can (for example "
    "Apple -> AAPL)."
)


def build_mcp_client(project_root):
    return MultiServerMCPClient(
        {
            "sec": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.sec_server"],
                "cwd": str(project_root),
            }
        }
    )


async def build_sec_agent(project_root):
    client = build_mcp_client(project_root)
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


async def main() -> None:
    project_root = get_project_root()
    agent = await build_sec_agent(project_root)
    await interactive_loop(
        agent,
        "SEC agent ready. Ask about filings (e.g. 'Recent 10-K filings for AAPL').",
        session_name="sec_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
