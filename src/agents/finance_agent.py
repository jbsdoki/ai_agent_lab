"""Local Ollama agent with yfinance MCP tools."""

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
    "You are a finance assistant. Use the available stock tools to answer questions "
    "about ticker symbols. When the user asks about a company, infer the ticker if "
    "you can (for example Apple -> AAPL). Always call a tool for live market data "
    "instead of guessing prices."
)


def build_mcp_client(project_root):
    return MultiServerMCPClient(
        {
            "yfinance": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.yfinance_server"],
                "cwd": str(project_root),
            }
        }
    )


async def main() -> None:
    project_root = get_project_root()
    client = build_mcp_client(project_root)
    tools = await client.get_tools()
    llm = build_llm()
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    await interactive_loop(
        agent,
        "Finance agent ready. Ask about stocks (e.g. 'What is AAPL trading at?').",
        session_name="finance_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
