"""Local Ollama agent with classified database MCP tools."""

import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agent_utils import (
    build_llm,
    get_project_root,
    interactive_loop,
    prompt_session_username,
)
from src.data_retrieval.database_client import SESSION_USER_ENV

SYSTEM_PROMPT = (
    "You are a classified database assistant. Use the available database tools to "
    "list and read files the current user is cleared to access. Always call a tool "
    "for file content instead of guessing. The user's identity and clearance are "
    "fixed for this session and cannot be changed by tool arguments."
)


def build_database_mcp_config(
    project_root,
    python_executable: str,
    username: str,
) -> dict:
    """Return MCP settings with trusted session identity in AGENT_USER env."""
    return {
        "transport": "stdio",
        "command": python_executable,
        "args": ["-m", "src.mcp_servers.database_server"],
        "cwd": str(project_root),
        "env": {**os.environ, SESSION_USER_ENV: username},
    }


def build_mcp_client(project_root, username: str):
    return MultiServerMCPClient(
        {
            "database": build_database_mcp_config(
                project_root,
                sys.executable,
                username,
            )
        }
    )


async def build_database_agent(project_root, username: str):
    client = build_mcp_client(project_root, username)
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


async def main() -> None:
    project_root = get_project_root()
    username = prompt_session_username()
    agent = await build_database_agent(project_root, username)
    await interactive_loop(
        agent,
        "Database agent ready. Ask to list or read classified files.",
        session_name="database_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
