"""Local Ollama agent with classified database MCP tools and long-term memory."""

import asyncio
import json
import os
import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    SESSION_LOG_PATH_ENV,
    append_session_log,
    build_llm,
    create_conversation_history,
    extend_conversation_history,
    extract_reply,
    get_project_root,
    log_agent_reply,
    log_agent_result,
    log_user_message,
    prompt_session_username,
    start_session_log,
)
from src.agents.coordinator_agent import (
    append_memory_context,
    build_memory_mcp_config,
)
from src.data_retrieval.database_client import (
    GRANT_STORE_PATH_ENV,
    SESSION_USER_ENV,
    create_grant_store_path,
    record_grant,
)
from src.data_retrieval.memory_client import (
    MEMORY_STORE_PATH_ENV,
    MEMORY_USER_ENV,
    build_memory_recall_prompt,
    initialize_memory_store,
)

DATABASE_BASE_PROMPT = (
    "You are a classified database assistant. Use the available database tools to "
    "list and read files the current user is cleared to access. Always call a tool "
    "for file content instead of guessing. The user's identity and clearance are "
    "fixed for this session and cannot be changed by tool arguments. "
    "Some files require explicit operator approval before reading. If a tool returns "
    "status approval_required, tell the user that approval is needed and wait; "
    "never invent file contents."
)


def build_database_mcp_config(
    project_root,
    python_executable: str,
    username: str,
    session_log_path,
    grant_store_path,
) -> dict:
    """Return MCP settings with trusted session identity and grant store env."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.database_server"],
        "cwd": str(project_root),
        "env": {
            **os.environ,
            SESSION_USER_ENV: username,
            SESSION_LOG_PATH_ENV: str(session_log_path),
            GRANT_STORE_PATH_ENV: str(grant_store_path),
        },
    }


def build_mcp_client(
    project_root,
    username: str,
    session_log_path,
    grant_store_path,
    memory_store_path,
):
    return MultiServerMCPClient(
        {
            "database": build_database_mcp_config(
                project_root,
                sys.executable,
                username,
                session_log_path,
                grant_store_path,
            ),
            "memory": build_memory_mcp_config(
                project_root,
                sys.executable,
                username,
                session_log_path,
                memory_store_path,
            ),
        }
    )


async def build_database_agent(
    project_root,
    username: str,
    session_log_path,
    grant_store_path,
    memory_store_path,
    recall_prompt: str = "",
):
    client = build_mcp_client(
        project_root,
        username,
        session_log_path,
        grant_store_path,
        memory_store_path,
    )
    tools = await client.get_tools()
    llm = build_llm()
    system_prompt = append_memory_context(DATABASE_BASE_PROMPT, recall_prompt)
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


async def build_database_session(
    project_root,
    username: str,
    session_log_path,
    grant_store_path,
    memory_store_path,
) -> dict:
    recall_prompt = build_memory_recall_prompt(username)
    agent = await build_database_agent(
        project_root,
        username,
        session_log_path,
        grant_store_path,
        memory_store_path,
        recall_prompt,
    )
    return {
        "username": username,
        "log_path": session_log_path,
        "grant_store_path": grant_store_path,
        "memory_store_path": memory_store_path,
        "recall_prompt": recall_prompt,
        "agent": agent,
    }


def parse_tool_payload(content: str) -> dict | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def find_approval_requests(messages: list) -> list[dict]:
    requests: list[dict] = []
    seen_paths: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = parse_tool_payload(str(message.content))
        if payload is None or payload.get("status") != "approval_required":
            continue
        path = payload.get("path")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        requests.append(payload)
    return requests


def prompt_access_approval(path: str, classification: str) -> bool:
    while True:
        answer = input(
            f"Approve read of {path} ({classification})? [y/n]: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def process_approval_requests(requests: list[dict], username: str) -> tuple[list[str], list[str]]:
    granted_paths: list[str] = []
    denied_paths: list[str] = []
    for request in requests:
        path = str(request["path"])
        classification = str(request.get("classification", "unknown"))
        if prompt_access_approval(path, classification):
            record_grant(username, path, granted=True, source="human")
            granted_paths.append(path)
        else:
            record_grant(username, path, granted=False, source="human")
            denied_paths.append(path)
    return granted_paths, denied_paths


def build_follow_up_prompt(granted_paths: list[str], denied_paths: list[str]) -> str:
    if granted_paths and denied_paths:
        return (
            f"Access approved for: {', '.join(granted_paths)}. "
            f"Access denied for: {', '.join(denied_paths)}. "
            "Read the approved files now and explain any denials to the user."
        )
    if granted_paths:
        return (
            f"Access approved for: {', '.join(granted_paths)}. "
            "Read those files now and summarize their contents."
        )
    return (
        f"Access denied for: {', '.join(denied_paths)}. "
        "Inform the user that permission was not granted."
    )


async def run_prompt_with_approvals(
    agent,
    user_prompt: str,
    username: str,
    conversation_messages: list,
    log_label: str = "AGENT",
) -> str:
    from src.agents.trace_utils import begin_trace_turn, prepare_retry_invoke_config

    log_user_message(user_prompt)
    current_prompt = user_prompt
    result = None
    begin_trace_turn()
    retry_index = 0

    while True:
        conversation_messages.append(HumanMessage(content=current_prompt))
        config = prepare_retry_invoke_config(retry_index)
        result = await agent.ainvoke({"messages": conversation_messages}, config=config)
        extend_conversation_history(conversation_messages, result)
        log_agent_result(result, log_label)
        approval_requests = find_approval_requests(result.get("messages", []))
        if not approval_requests:
            break

        granted_paths, denied_paths = process_approval_requests(
            approval_requests, username
        )
        current_prompt = build_follow_up_prompt(granted_paths, denied_paths)
        retry_index += 1

    reply = extract_reply(result)
    log_agent_reply(reply, log_label)
    return reply


async def database_interactive_loop(
    session: dict,
    welcome_message: str,
    session_name: str = "database_agent",
) -> None:
    log_path = session["log_path"]
    print(welcome_message)
    print(f"Session log: {log_path}")
    from src.agents.trace_utils import get_active_trace_path

    trace_path = get_active_trace_path()
    if trace_path is not None:
        print(f"Trace log: {trace_path}")
    print(f"Memory store: {session['memory_store_path']}")
    print("Type 'quit' or 'exit' to stop.\n")

    conversation_messages = create_conversation_history()
    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            append_session_log("=== SESSION END ===")
            break

        reply = await run_prompt_with_approvals(
            session["agent"],
            user_prompt,
            session["username"],
            conversation_messages,
            log_label=session_name.upper(),
        )
        print(f"\nAgent: {reply}\n")


async def main() -> None:
    project_root = get_project_root()
    username = prompt_session_username()
    log_path = start_session_log("database_agent")
    grant_store_path = create_grant_store_path(log_path)
    grant_store_path.write_text("{}", encoding="utf-8")
    memory_store_path = initialize_memory_store(username)

    os.environ[GRANT_STORE_PATH_ENV] = str(grant_store_path)
    os.environ[MEMORY_STORE_PATH_ENV] = str(memory_store_path)
    os.environ[MEMORY_USER_ENV] = username

    session = await build_database_session(
        project_root,
        username,
        log_path,
        grant_store_path,
        memory_store_path,
    )
    await database_interactive_loop(
        session,
        "Database agent ready. Ask to list or read classified files.",
        session_name="database_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
