"""Local Ollama agent with allowlisted web fetch MCP tools."""

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
    "You are a web page assistant. Use the fetch_url tool to retrieve page content "
    "from allowlisted HTTPS sites. Always call the tool instead of guessing page "
    "content. Summarize the extracted text clearly and mention when content was "
    "truncated. Only request URLs on allowlisted domains."
)


def build_mcp_client(project_root):
    return MultiServerMCPClient(
        {
            "web": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.web_server"],
                "cwd": str(project_root),
            }
        }
    )


async def build_web_agent(project_root):
    client = build_mcp_client(project_root)
    tools = await client.get_tools()
    llm = build_llm()
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


async def main() -> None:
    project_root = get_project_root()
    agent = await build_web_agent(project_root)
    await interactive_loop(
        agent,
        "Web agent ready. Ask about an allowlisted URL (e.g. a page on apple.com).",
        session_name="web_agent",
    )


if __name__ == "__main__":
    asyncio.run(main())
