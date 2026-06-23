"""Local Ollama agent with RAG document search MCP tools."""

import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    SESSION_LOG_PATH_ENV,
    append_session_log,
    build_llm,
    create_conversation_history,
    get_project_root,
    run_prompt_with_history,
    start_session_log,
)
from src.data_retrieval.rag_client import CHROMA_PERSIST_DIR_ENV

SYSTEM_PROMPT = (
    "You are a document search assistant for the lab's standard-safe corpus. "
    "Use search_documents to find relevant passages before answering content questions. "
    "Always cite source_path from tool results. Do not invent passages. "
    "This corpus does not include classified secret or top_secret files; "
    "direct users to the database agent for classified file reads."
)


def build_rag_mcp_config(project_root, python_executable: str, session_log_path) -> dict:
    chroma_persist_path = project_root / "data" / "chroma"
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


def build_mcp_client(project_root, session_log_path):
    return MultiServerMCPClient(
        {
            "rag": build_rag_mcp_config(
                project_root,
                sys.executable,
                session_log_path,
            )
        }
    )


async def rag_interactive_loop(agent, welcome_message: str, log_path) -> None:
    conversation_messages = create_conversation_history()
    print(welcome_message)
    print(f"Session log: {log_path}")
    from src.agents.trace_utils import get_active_trace_path

    trace_path = get_active_trace_path()
    if trace_path is not None:
        print(f"Trace log: {trace_path}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            append_session_log("=== SESSION END ===")
            break

        reply = await run_prompt_with_history(
            agent,
            user_prompt,
            conversation_messages,
            log_label="RAG_AGENT",
        )
        print(f"\nAgent: {reply}\n")


async def main() -> None:
    project_root = get_project_root()
    log_path = start_session_log("rag_agent")
    os.environ.setdefault(CHROMA_PERSIST_DIR_ENV, str(project_root / "data" / "chroma"))
    os.environ[SESSION_LOG_PATH_ENV] = str(log_path)

    client = build_mcp_client(project_root, log_path)
    tools = await client.get_tools()
    llm = build_llm()
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    await rag_interactive_loop(
        agent,
        "RAG agent ready. Ask to search standard docs (run ingest_rag_corpus first).",
        log_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
