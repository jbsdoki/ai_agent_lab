"""Local Ollama agent with classified database MCP tools."""

import asyncio
import json
import os
import sys

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    SESSION_LOG_PATH_ENV,
    append_session_log,
    build_llm,
    extract_reply,
    get_project_root,
    log_agent_reply,
    log_agent_result,
    log_user_message,
    prompt_session_username,
    start_session_log,
)
from src.data_retrieval.database_client import (
    GRANT_STORE_PATH_ENV,
    SESSION_USER_ENV,
    create_grant_store_path,
    record_grant,
)

SYSTEM_PROMPT = (
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


def build_mcp_client(project_root, username: str, session_log_path, grant_store_path):
    return MultiServerMCPClient(
        {
            "database": build_database_mcp_config(
                project_root,
                sys.executable,
                username,
                session_log_path,
                grant_store_path,
            )
        }
    )


async def build_database_agent(
    project_root,
    username: str,
    session_log_path,
    grant_store_path,
):
    client = build_mcp_client(
        project_root, username, session_log_path, grant_store_path
    )
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


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
    log_label: str = "AGENT",
) -> str:
    log_user_message(user_prompt)
    current_prompt = user_prompt
    result = None

    while True:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": current_prompt}]}
        )
        log_agent_result(result, log_label)
        approval_requests = find_approval_requests(result.get("messages", []))
        if not approval_requests:
            break

        granted_paths, denied_paths = process_approval_requests(
            approval_requests, username
        )
        current_prompt = build_follow_up_prompt(granted_paths, denied_paths)

    reply = extract_reply(result)
    log_agent_reply(reply, log_label)
    return reply


async def database_interactive_loop(
    agent,
    username: str,
    welcome_message: str,
    log_path,
    session_name: str = "database_agent",
) -> None:
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

        reply = await run_prompt_with_approvals(
            agent,
            user_prompt,
            username,
            log_label=session_name.upper(),
        )
        print(f"\nAgent: {reply}\n")


async def main() -> None:
    project_root = get_project_root()
    username = prompt_session_username()
    log_path = start_session_log("database_agent")
    grant_store_path = create_grant_store_path(log_path)
    os.environ[GRANT_STORE_PATH_ENV] = str(grant_store_path)
    grant_store_path.write_text("{}", encoding="utf-8")

    agent = await build_database_agent(
        project_root, username, log_path, grant_store_path
    )
    await database_interactive_loop(
        agent,
        username,
        "Database agent ready. Ask to list or read classified files.",
        log_path,
        session_name="database_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
