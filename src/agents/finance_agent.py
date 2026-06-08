"""Local Ollama agent with yfinance MCP tools."""

import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:latest"
SYSTEM_PROMPT = (
    "You are a finance assistant. Use the available stock tools to answer questions "
    "about ticker symbols. When the user asks about a company, infer the ticker if "
    "you can (for example Apple -> AAPL). Always call a tool for live market data "
    "instead of guessing prices."
)


def get_project_root() -> Path:
    return PROJECT_ROOT


def build_mcp_client(project_root: Path) -> MultiServerMCPClient:
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


def build_llm(model: str, base_url: str) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def extract_reply(result: dict) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "No response from agent."


async def load_tools(client: MultiServerMCPClient):
    return await client.get_tools()


async def run_prompt(agent, user_prompt: str) -> str:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    return extract_reply(result)


async def interactive_loop(agent) -> None:
    print("Finance agent ready. Ask about stocks (e.g. 'What is AAPL trading at?').")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            break

        reply = await run_prompt(agent, user_prompt)
        print(f"\nAgent: {reply}\n")


async def main() -> None:
    project_root = get_project_root()
    client = build_mcp_client(project_root)
    tools = await load_tools(client)
    llm = build_llm(MODEL, OLLAMA_HOST)
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    await interactive_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
