"""Smoke test for the NewsAPI MCP server tool registration."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


def get_project_root() -> Path:
    return PROJECT_ROOT


def build_mcp_client(project_root: Path) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "newsapi": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.newsapi_server"],
                "cwd": str(project_root),
            }
        }
    )


async def fetch_tool_names(client: MultiServerMCPClient) -> list[str]:
    tools = await client.get_tools()
    return [tool.name for tool in tools]


def format_result(tool_names: list[str]) -> str:
    lines = [
        "NewsAPI MCP smoke test",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Project root: {PROJECT_ROOT}",
        f"Tool count: {len(tool_names)}",
        "Tools:",
    ]
    lines.extend(f"  - {name}" for name in tool_names)
    lines.append("")
    lines.append("Result: PASS" if tool_names else "Result: FAIL (no tools returned)")
    return "\n".join(lines)


def build_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"test_newsapi_mcp_{timestamp}.logs"


def write_log_file(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


async def run_test() -> tuple[str, Path]:
    project_root = get_project_root()
    client = build_mcp_client(project_root)
    tool_names = await fetch_tool_names(client)
    result_text = format_result(tool_names)
    log_path = build_log_path(LOGS_DIR)
    write_log_file(log_path, result_text)
    return result_text, log_path


if __name__ == "__main__":
    output, log_file = asyncio.run(run_test())
    print(output)
    print(f"\nLog written to: {log_file}")
