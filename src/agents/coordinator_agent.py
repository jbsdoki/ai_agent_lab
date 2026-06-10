"""Coordinator agent with finance and news subagents.

The coordinator receives your questions and delegates work to specialist
subagents. Each subagent has its own MCP tools (yfinance or NewsAPI).
"""

import asyncio
import sys

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    build_llm,
    extract_reply,
    get_project_root,
    interactive_loop,
    log_subagent_invocation,
)

# Tells the coordinator when to delegate and how to combine subagent answers.
SYSTEM_PROMPT = (
    "You are a coordinator assistant. Delegate stock and market questions to the "
    "finance subagent and news or headline questions to the news subagent. "
    "For mixed questions, call both subagents and combine their answers clearly."
)


def build_yfinance_mcp_config(project_root, python_executable: str) -> dict:
    """Return MCP connection settings for spawning the yfinance server process."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.yfinance_server"],
        "cwd": str(project_root),
    }


def build_newsapi_mcp_config(project_root, python_executable: str) -> dict:
    """Return MCP connection settings for spawning the NewsAPI server process."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.newsapi_server"],
        "cwd": str(project_root),
    }


async def build_finance_subagent(project_root):
    """Create the finance specialist agent with yfinance MCP tools."""
    client = MultiServerMCPClient(
        {"yfinance": build_yfinance_mcp_config(project_root, sys.executable)}
    )
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="finance_subagent",
        system_prompt=(
            "You are a finance specialist. Use stock tools for ticker questions. "
            "Infer tickers from company names when needed."
        ),
    )


async def build_news_subagent(project_root):
    """Create the news specialist agent with NewsAPI MCP tools."""
    client = MultiServerMCPClient(
        {"newsapi": build_newsapi_mcp_config(project_root, sys.executable)}
    )
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="news_subagent",
        system_prompt=(
            "You are a news specialist. Use news tools for headlines and topic searches."
        ),
    )


def build_subagent_tool(agent, tool_name: str, description: str) -> StructuredTool:
    """Wrap a subagent as a callable tool the coordinator can invoke."""

    async def invoke_subagent(query: str) -> str:
        # Run the specialist agent on the delegated query.
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        # Record subagent tool calls and replies in the session log.
        log_subagent_invocation(tool_name, query, result)
        return extract_reply(result)

    return StructuredTool.from_function(
        coroutine=invoke_subagent,
        name=tool_name,
        description=description,
    )


async def build_coordinator_agent(project_root):
    """Build the top-level coordinator and attach both subagents as tools."""
    finance_agent = await build_finance_subagent(project_root)
    news_agent = await build_news_subagent(project_root)

    finance_tool = build_subagent_tool(
        finance_agent,
        "finance_subagent",
        "Ask the finance specialist about stocks, prices, and market data.",
    )
    news_tool = build_subagent_tool(
        news_agent,
        "news_subagent",
        "Ask the news specialist about headlines, topics, and current events.",
    )

    llm = build_llm()
    return create_agent(
        model=llm,
        tools=[finance_tool, news_tool],
        name="coordinator_agent",
        system_prompt=SYSTEM_PROMPT,
    )


async def main() -> None:
    """Entry point: build the coordinator and start the interactive chat loop."""
    project_root = get_project_root()
    agent = await build_coordinator_agent(project_root)
    await interactive_loop(
        agent,
        "Coordinator ready. Ask about stocks, news, or both "
        "(e.g. 'What is AAPL trading at and any recent Apple news?').",
        session_name="coordinator_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
