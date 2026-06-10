"""Local Ollama agent with NewsAPI MCP tools."""

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
    "You are a news assistant. Use the available news tools to answer questions "
    "about current events, headlines, and topics. Always call a tool for live news "
    "data instead of guessing headlines."
)


def build_mcp_client(project_root):
    return MultiServerMCPClient(
        {
            "newsapi": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.newsapi_server"],
                "cwd": str(project_root),
            }
        }
    )


async def build_news_agent(project_root):
    client = build_mcp_client(project_root)
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


async def main() -> None:
    project_root = get_project_root()
    agent = await build_news_agent(project_root)
    await interactive_loop(
        agent,
        "News agent ready. Ask about headlines or topics (e.g. 'Latest AI news').",
        session_name="news_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
