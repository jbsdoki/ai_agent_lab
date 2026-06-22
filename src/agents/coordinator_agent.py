"""Coordinator agent with finance, news, web, SEC, and RAG subagents.

The coordinator uses intent routing to enable only relevant subagents each turn.
Single-intent prompts dispatch directly to one specialist without a coordinator LLM call.
Memory MCP tools are available on direct dispatch and multi-subagent coordinator turns.
RAG dispatch uses document search tools only (no memory tools).
"""

import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    SESSION_LOG_PATH_ENV,
    append_session_log,
    build_llm,
    create_conversation_history,
    extract_reply,
    get_project_root,
    log_router_decision,
    log_subagent_invocation,
    prompt_session_username,
    run_prompt_with_history,
    start_session_log,
)
from src.agents.intent_router import (
    FINANCE_SUBAGENT,
    NEWS_SUBAGENT,
    RAG_SUBAGENT,
    SEC_SUBAGENT,
    WEB_SUBAGENT,
    detect_subagent_intents,
    get_default_subagents,
    is_subagent_enabled,
)
from src.data_retrieval.memory_client import (
    MEMORY_STORE_PATH_ENV,
    MEMORY_USER_ENV,
    build_memory_recall_prompt,
    initialize_memory_store,
)
from src.data_retrieval.rag_client import CHROMA_PERSIST_DIR_ENV

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
    RAG_SUBAGENT: (
        "Ask the document search specialist to find passages in the standard-safe corpus."
    ),
}

SUBAGENT_SYSTEM_PROMPTS = {
    FINANCE_SUBAGENT: (
        "You are a finance specialist. Use stock tools for ticker questions. "
        "Infer tickers from company names when needed."
    ),
    NEWS_SUBAGENT: (
        "You are a news specialist. Use news tools for headlines and topic searches."
    ),
    WEB_SUBAGENT: (
        "You are a web page specialist. Use fetch_url for allowlisted HTTPS pages "
        "and summarize extracted text clearly."
    ),
    SEC_SUBAGENT: (
        "You are an SEC filings specialist. Use SEC tools for 10-K, 10-Q, and 8-K "
        "questions. Do not guess filing dates or accession numbers."
    ),
    RAG_SUBAGENT: (
        "You are a document search specialist for the standard-safe corpus. "
        "Use search_documents for passage lookups. Cite source_path from results. "
        "Do not invent passages. Classified secret and top_secret files are not "
        "in this index; direct users to the database agent for classified reads."
    ),
}

MEMORY_SYSTEM_INSTRUCTIONS = (
    "Use save_memory when the user explicitly asks you to remember a fact or preference. "
    "Use search_memories before answering questions about prior preferences or stored facts. "
    "Do not store classified document contents in memory."
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


def build_rag_mcp_config(
    project_root,
    python_executable: str,
    session_log_path,
    chroma_persist_path,
) -> dict:
    """Return MCP settings for the RAG server subprocess."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.rag_server"],
        "cwd": str(project_root),
        "env": {
            **os.environ,
            CHROMA_PERSIST_DIR_ENV: str(chroma_persist_path),
            SESSION_LOG_PATH_ENV: str(session_log_path),
        },
    }


def build_memory_mcp_config(
    project_root,
    python_executable: str,
    username: str,
    session_log_path,
    memory_store_path,
) -> dict:
    """Return MCP settings for the memory server subprocess."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.memory_server"],
        "cwd": str(project_root),
        "env": {
            **os.environ,
            MEMORY_USER_ENV: username,
            MEMORY_STORE_PATH_ENV: str(memory_store_path),
            SESSION_LOG_PATH_ENV: str(session_log_path),
        },
    }


async def build_memory_tools(
    project_root,
    username: str,
    session_log_path,
    memory_store_path,
) -> list:
    client = MultiServerMCPClient(
        {
            "memory": build_memory_mcp_config(
                project_root,
                sys.executable,
                username,
                session_log_path,
                memory_store_path,
            )
        }
    )
    return await client.get_tools()


def append_memory_context(base_prompt: str, recall_prompt: str) -> str:
    sections = [base_prompt, MEMORY_SYSTEM_INSTRUCTIONS]
    if recall_prompt:
        sections.append(recall_prompt)
    return "\n\n".join(sections)


def build_finance_subagent(tools, recall_prompt: str = ""):
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="finance_subagent",
        system_prompt=append_memory_context(
            SUBAGENT_SYSTEM_PROMPTS[FINANCE_SUBAGENT], recall_prompt
        ),
    )


def build_news_subagent(tools, recall_prompt: str = ""):
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="news_subagent",
        system_prompt=append_memory_context(
            SUBAGENT_SYSTEM_PROMPTS[NEWS_SUBAGENT], recall_prompt
        ),
    )


def build_web_subagent(tools, recall_prompt: str = ""):
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="web_subagent",
        system_prompt=append_memory_context(
            SUBAGENT_SYSTEM_PROMPTS[WEB_SUBAGENT], recall_prompt
        ),
    )


def build_sec_subagent(tools, recall_prompt: str = ""):
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="sec_subagent",
        system_prompt=append_memory_context(
            SUBAGENT_SYSTEM_PROMPTS[SEC_SUBAGENT], recall_prompt
        ),
    )


def build_rag_subagent(tools):
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=tools,
        name="rag_subagent",
        system_prompt=SUBAGENT_SYSTEM_PROMPTS[RAG_SUBAGENT],
    )


async def load_finance_mcp_tools(project_root) -> list:
    client = MultiServerMCPClient(
        {"yfinance": build_yfinance_mcp_config(project_root, sys.executable)}
    )
    return await client.get_tools()


async def load_news_mcp_tools(project_root) -> list:
    client = MultiServerMCPClient(
        {"newsapi": build_newsapi_mcp_config(project_root, sys.executable)}
    )
    return await client.get_tools()


async def load_web_mcp_tools(project_root) -> list:
    client = MultiServerMCPClient(
        {"web": build_web_mcp_config(project_root, sys.executable)}
    )
    return await client.get_tools()


async def load_sec_mcp_tools(project_root) -> list:
    client = MultiServerMCPClient(
        {"sec": build_sec_mcp_config(project_root, sys.executable)}
    )
    return await client.get_tools()


async def load_rag_mcp_tools(
    project_root,
    session_log_path,
    chroma_persist_path,
) -> list:
    client = MultiServerMCPClient(
        {
            "rag": build_rag_mcp_config(
                project_root,
                sys.executable,
                session_log_path,
                chroma_persist_path,
            )
        }
    )
    return await client.get_tools()


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


def build_coordinator_system_prompt(
    enabled_tool_names: list[str],
    recall_prompt: str = "",
) -> str:
    specialists = ", ".join(enabled_tool_names)
    base_prompt = (
        "You are a coordinator assistant for this turn. Delegate only to the "
        f"specialists available as tools: {specialists}. "
        "Do not mention or attempt unavailable domains or specialists. "
        "For mixed questions, call the relevant subagent tools and combine their "
        "answers clearly."
    )
    return append_memory_context(base_prompt, recall_prompt)


def build_coordinator_for_tools(
    enabled_tools: list[StructuredTool],
    memory_tools: list,
    recall_prompt: str = "",
):
    """Create a coordinator agent with subagent and memory tools."""
    enabled_names = [tool.name for tool in enabled_tools]
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=[*enabled_tools, *memory_tools],
        name="coordinator_agent",
        system_prompt=build_coordinator_system_prompt(enabled_names, recall_prompt),
    )


def build_direct_dispatch_agent(
    subagent_name: str,
    mcp_tools: list,
    memory_tools: list,
    recall_prompt: str = "",
):
    builders = {
        FINANCE_SUBAGENT: build_finance_subagent,
        NEWS_SUBAGENT: build_news_subagent,
        WEB_SUBAGENT: build_web_subagent,
        SEC_SUBAGENT: build_sec_subagent,
    }
    builder = builders[subagent_name]
    return builder(
        [*memory_tools, *mcp_tools],
        recall_prompt,
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


async def build_cached_subagents(
    project_root,
    session_log_path,
    chroma_persist_path,
    recall_prompt: str = "",
) -> dict:
    """Build all specialist agents once and wrap them as coordinator tools."""
    finance_tools = await load_finance_mcp_tools(project_root)
    news_tools = await load_news_mcp_tools(project_root)
    web_tools = await load_web_mcp_tools(project_root)
    sec_tools = await load_sec_mcp_tools(project_root)
    rag_tools = await load_rag_mcp_tools(
        project_root,
        session_log_path,
        chroma_persist_path,
    )

    agent_specs = {
        FINANCE_SUBAGENT: (finance_tools, True),
        NEWS_SUBAGENT: (news_tools, True),
        WEB_SUBAGENT: (web_tools, True),
        SEC_SUBAGENT: (sec_tools, True),
        RAG_SUBAGENT: (rag_tools, False),
    }

    cached = {}
    for name, (mcp_tools, uses_memory) in agent_specs.items():
        if uses_memory:
            agent = build_direct_dispatch_agent(
                name,
                mcp_tools,
                memory_tools=[],
                recall_prompt=recall_prompt,
            )
        else:
            agent = build_rag_subagent(mcp_tools)
        cached[name] = {
            "agent": agent,
            "tool": build_subagent_tool(agent, name, SUBAGENT_DESCRIPTIONS[name]),
            "mcp_tools": mcp_tools,
            "uses_memory": uses_memory,
            "system_prompt": SUBAGENT_SYSTEM_PROMPTS[name],
        }
    return cached


async def build_coordinator_session(
    project_root,
    username: str,
    session_log_path,
    memory_store_path,
) -> dict:
    recall_prompt = build_memory_recall_prompt(username)
    chroma_persist_path = project_root / "data" / "chroma"
    os.environ.setdefault(CHROMA_PERSIST_DIR_ENV, str(chroma_persist_path))
    memory_tools = await build_memory_tools(
        project_root,
        username,
        session_log_path,
        memory_store_path,
    )
    cached_subagents = await build_cached_subagents(
        project_root,
        session_log_path,
        chroma_persist_path,
        recall_prompt,
    )
    return {
        "username": username,
        "log_path": session_log_path,
        "memory_store_path": memory_store_path,
        "chroma_persist_path": chroma_persist_path,
        "memory_tools": memory_tools,
        "recall_prompt": recall_prompt,
        "cached_subagents": cached_subagents,
    }


async def run_coordinator_turn(
    session: dict,
    user_prompt: str,
    conversation_messages: list,
) -> str:
    """Route one user prompt through the intent router and dispatch specialists."""
    cached_subagents = session["cached_subagents"]
    enabled_intents = resolve_turn_intents(user_prompt, cached_subagents)
    log_router_decision(user_prompt, enabled_intents)

    ordered_names, _agents, tools = select_turn_subagents(
        cached_subagents,
        enabled_intents,
    )

    if len(tools) == 1:
        subagent_name = ordered_names[0]
        subagent_entry = cached_subagents[subagent_name]
        if subagent_entry.get("uses_memory", True):
            agent = build_direct_dispatch_agent(
                subagent_name,
                subagent_entry["mcp_tools"],
                session["memory_tools"],
                session["recall_prompt"],
            )
        else:
            agent = build_rag_subagent(subagent_entry["mcp_tools"])
        return await run_prompt_with_history(
            agent,
            user_prompt,
            conversation_messages,
            log_label=subagent_name.upper(),
        )

    coordinator = build_coordinator_for_tools(
        tools,
        session["memory_tools"],
        session["recall_prompt"],
    )
    return await run_prompt_with_history(
        coordinator,
        user_prompt,
        conversation_messages,
        log_label="COORDINATOR_AGENT",
    )


async def build_coordinator_agent(project_root, username: str, session_log_path, memory_store_path):
    """Build a coordinator with all subagents enabled (legacy non-routed entry point)."""
    session = await build_coordinator_session(
        project_root, username, session_log_path, memory_store_path
    )
    all_tools = [
        session["cached_subagents"][name]["tool"]
        for name in sorted(session["cached_subagents"])
    ]
    return build_coordinator_for_tools(
        all_tools,
        session["memory_tools"],
        session["recall_prompt"],
    )


async def interactive_coordinator_loop(
    session: dict,
    welcome_message: str,
    session_name: str = "coordinator_agent",
) -> None:
    """Run a REPL chat loop with per-turn intent routing and session history."""
    log_path = session["log_path"]
    conversation_messages = create_conversation_history()
    print(welcome_message)
    print(f"Session log: {log_path}")
    print(f"Memory store: {session['memory_store_path']}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            append_session_log("=== SESSION END ===")
            break

        reply = await run_coordinator_turn(
            session,
            user_prompt,
            conversation_messages,
        )
        print(f"\nAgent: {reply}\n")


async def main() -> None:
    """Entry point: build cached subagents and start the routed chat loop."""
    project_root = get_project_root()
    username = prompt_session_username()
    log_path = start_session_log("coordinator_agent")
    memory_store_path = initialize_memory_store(username)
    os.environ[MEMORY_STORE_PATH_ENV] = str(memory_store_path)
    os.environ[MEMORY_USER_ENV] = username

    session = await build_coordinator_session(
        project_root,
        username,
        log_path,
        memory_store_path,
    )
    await interactive_coordinator_loop(
        session,
        "Coordinator ready. Ask about stocks, news, SEC filings, allowlisted web pages, "
        "or document search in the standard corpus (run ingest_rag_corpus first).",
        session_name="coordinator_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
