"""Coordinator agent with finance, news, web, and SEC subagents.

The coordinator uses intent routing to enable only relevant subagents each turn.
Single-intent prompts dispatch directly to one specialist without a coordinator LLM call.
"""

import asyncio
import sys

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    append_session_log,
    build_llm,
    extract_reply,
    get_project_root,
    log_router_decision,
    log_subagent_invocation,
    run_prompt,
    start_session_log,
)
from src.agents.intent_router import (
    FINANCE_SUBAGENT,
    NEWS_SUBAGENT,
    SEC_SUBAGENT,
    WEB_SUBAGENT,
    detect_subagent_intents,
    get_default_subagents,
    is_subagent_enabled,
)

SUBAGENT_DESCRIPTIONS = {
    FINANCE_SUBAGENT: (
        "Ask the finance specialist about stocks, prices, and market data."
    ),
    NEWS_SUBAGENT: (
        "Ask the news specialist about headlines, topics, and current events."
    ),
    WEB_SUBAGENT: (
        "Ask the web specialist to fetch and summarize allowlisted HTTPS pages."
    ),
    SEC_SUBAGENT: (
        "Ask the SEC specialist about EDGAR filings such as 10-K, 10-Q, and 8-K."
    ),
}


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


def build_web_mcp_config(project_root, python_executable: str) -> dict:
    """Return MCP connection settings for spawning the web server process."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.web_server"],
        "cwd": str(project_root),
    }


def build_sec_mcp_config(project_root, python_executable: str) -> dict:
    """Return MCP connection settings for spawning the SEC server process."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.sec_server"],
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


async def build_web_subagent(project_root):
    """Create the web specialist agent with allowlisted fetch MCP tools."""
    client = MultiServerMCPClient(
        {"web": build_web_mcp_config(project_root, sys.executable)}
    )
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="web_subagent",
        system_prompt=(
            "You are a web page specialist. Use fetch_url for allowlisted HTTPS pages "
            "and summarize extracted text clearly."
        ),
    )


async def build_sec_subagent(project_root):
    """Create the SEC specialist agent with EDGAR MCP tools."""
    client = MultiServerMCPClient(
        {"sec": build_sec_mcp_config(project_root, sys.executable)}
    )
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="sec_subagent",
        system_prompt=(
            "You are an SEC filings specialist. Use SEC tools for 10-K, 10-Q, and 8-K "
            "questions. Do not guess filing dates or accession numbers."
        ),
    )


def build_subagent_tool(agent, tool_name: str, description: str) -> StructuredTool:
    """Wrap a subagent as a callable tool the coordinator can invoke."""

    async def invoke_subagent(query: str) -> str:
        if not is_subagent_enabled(query, tool_name):
            return (
                f"{tool_name} is not enabled for this delegated query based on intent "
                "routing. Rephrase the query or use another available specialist."
            )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        log_subagent_invocation(tool_name, query, result)
        return extract_reply(result)

    return StructuredTool.from_function(
        coroutine=invoke_subagent,
        name=tool_name,
        description=description,
    )


def build_system_prompt(enabled_tool_names: list[str]) -> str:
    specialists = ", ".join(enabled_tool_names)
    return (
        "You are a coordinator assistant for this turn. Delegate only to the "
        f"specialists available as tools: {specialists}. "
        "Do not mention or attempt unavailable domains or specialists. "
        "For mixed questions, call the relevant subagent tools and combine their "
        "answers clearly."
    )


def build_coordinator_for_tools(enabled_tools: list[StructuredTool]):
    """Create a coordinator agent that can use only the provided subagent tools."""
    enabled_names = [tool.name for tool in enabled_tools]
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=enabled_tools,
        name="coordinator_agent",
        system_prompt=build_system_prompt(enabled_names),
    )


def resolve_turn_intents(user_prompt: str, cached_subagents: dict) -> set[str]:
    intents = detect_subagent_intents(user_prompt)
    available = {name for name in intents if name in cached_subagents}
    if not available:
        available = get_default_subagents() & set(cached_subagents.keys())
    return available


def select_turn_subagents(
    cached_subagents: dict,
    enabled_intents: set[str],
) -> tuple[list[str], list, list[StructuredTool]]:
    ordered_names = sorted(enabled_intents)
    agents = [cached_subagents[name]["agent"] for name in ordered_names]
    tools = [cached_subagents[name]["tool"] for name in ordered_names]
    return ordered_names, agents, tools


async def build_cached_subagents(project_root) -> dict:
    """Build all specialist agents once and wrap them as coordinator tools."""
    finance_agent = await build_finance_subagent(project_root)
    news_agent = await build_news_subagent(project_root)
    web_agent = await build_web_subagent(project_root)
    sec_agent = await build_sec_subagent(project_root)

    agent_map = {
        FINANCE_SUBAGENT: finance_agent,
        NEWS_SUBAGENT: news_agent,
        WEB_SUBAGENT: web_agent,
        SEC_SUBAGENT: sec_agent,
    }

    cached = {}
    for name, agent in agent_map.items():
        cached[name] = {
            "agent": agent,
            "tool": build_subagent_tool(agent, name, SUBAGENT_DESCRIPTIONS[name]),
        }
    return cached


async def run_coordinator_turn(cached_subagents: dict, user_prompt: str) -> str:
    """Route one user prompt through the intent router and dispatch specialists."""
    enabled_intents = resolve_turn_intents(user_prompt, cached_subagents)
    log_router_decision(user_prompt, enabled_intents)

    ordered_names, agents, tools = select_turn_subagents(
        cached_subagents,
        enabled_intents,
    )

    if len(tools) == 1:
        return await run_prompt(
            agents[0],
            user_prompt,
            log_label=ordered_names[0].upper(),
        )

    coordinator = build_coordinator_for_tools(tools)
    return await run_prompt(
        coordinator,
        user_prompt,
        log_label="COORDINATOR_AGENT",
    )


async def build_coordinator_agent(project_root):
    """Build a coordinator with all subagents enabled (legacy non-routed entry point)."""
    cached_subagents = await build_cached_subagents(project_root)
    all_tools = [cached_subagents[name]["tool"] for name in sorted(cached_subagents)]
    return build_coordinator_for_tools(all_tools)


async def interactive_coordinator_loop(
    cached_subagents: dict,
    welcome_message: str,
    session_name: str = "coordinator_agent",
) -> None:
    """Run a REPL chat loop with per-turn intent routing."""
    log_path = start_session_log(session_name)
    print(welcome_message)
    print(f"Session log: {log_path}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            append_session_log("=== SESSION END ===")
            break

        reply = await run_coordinator_turn(cached_subagents, user_prompt)
        print(f"\nAgent: {reply}\n")


async def main() -> None:
    """Entry point: build cached subagents and start the routed chat loop."""
    project_root = get_project_root()
    cached_subagents = await build_cached_subagents(project_root)
    await interactive_coordinator_loop(
        cached_subagents,
        "Coordinator ready. Ask about stocks, news, SEC filings, or allowlisted web pages.",
        session_name="coordinator_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
