"""Smoke tests for the memory MCP server and client tools.

Run:
  python tests/test_memory_mcp.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import SESSION_LOG_PATH_ENV
from src.data_retrieval.memory_client import (
    MEMORY_STORE_PATH_ENV,
    MEMORY_USER_ENV,
    delete_memory,
    list_memories,
    save_memory,
    search_memories,
)

LOGS_DIR = PROJECT_ROOT / "logs"
MEMORY_MCP_TOOLS = [
    "save_memory",
    "search_memories",
    "list_memories",
    "delete_memory",
]


def get_project_root() -> Path:
    return PROJECT_ROOT


def build_store_path(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.memory.json"


def build_log_path(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.logs"


def build_mcp_client(project_root: Path, username: str, store_path: Path) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "memory": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.memory_server"],
                "cwd": str(project_root),
                "env": {
                    **os.environ,
                    MEMORY_USER_ENV: username,
                    MEMORY_STORE_PATH_ENV: str(store_path),
                },
            }
        }
    )


async def fetch_tool_names(client: MultiServerMCPClient) -> list[str]:
    tools = await client.get_tools()
    return [tool.name for tool in tools]


def verify_registered_tools(tool_names: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in MEMORY_MCP_TOOLS if name not in tool_names]
    return not missing, missing


def parse_response(response_text: str) -> dict:
    return json.loads(response_text)


def run_client_tool_tests(store_path: Path, log_path: Path) -> tuple[bool, list[str]]:
    os.environ[MEMORY_USER_ENV] = "alice"
    os.environ[MEMORY_STORE_PATH_ENV] = str(store_path)
    os.environ[SESSION_LOG_PATH_ENV] = str(log_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps({"user": "alice", "memories": []}, indent=2),
        encoding="utf-8",
    )

    failures: list[str] = []

    saved = parse_response(save_memory("Favorite ticker is AAPL.", ["finance"]))
    if saved.get("id") != "mem-001":
        failures.append("save_memory did not return mem-001")

    search_payload = parse_response(search_memories("ticker"))
    if search_payload.get("count") != 1:
        failures.append("search_memories did not find saved memory")

    list_payload = parse_response(list_memories())
    if list_payload.get("count") != 1:
        failures.append("list_memories did not return one memory")

    delete_payload = parse_response(delete_memory("mem-001"))
    if not delete_payload.get("deleted"):
        failures.append("delete_memory did not delete mem-001")

    empty_payload = parse_response(list_memories())
    if empty_payload.get("count") != 0:
        failures.append("list_memories was not empty after delete")

    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if "[MEMORY]" not in log_text:
        failures.append("session log missing [MEMORY] entries")

    return not failures, failures


def format_result(
    tool_names: list[str],
    registration_passed: bool,
    missing_tools: list[str],
    client_passed: bool,
    client_failures: list[str],
) -> str:
    lines = [
        "Memory MCP smoke test",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Project root: {PROJECT_ROOT}",
        f"Expected tools: {MEMORY_MCP_TOOLS}",
        f"Tool count: {len(tool_names)}",
        "Tools:",
    ]
    lines.extend(f"  - {name}" for name in tool_names)
    lines.append("")
    lines.append(
        "Registration: PASS"
        if registration_passed
        else f"Registration: FAIL (missing {missing_tools})"
    )
    lines.append(
        "Client tools: PASS" if client_passed else "Client tools: FAIL"
    )
    if client_failures:
        lines.append("Client failures:")
        lines.extend(f"  - {entry}" for entry in client_failures)
    overall = registration_passed and client_passed
    lines.append("")
    lines.append("Result: PASS" if overall else "Result: FAIL")
    return "\n".join(lines)


async def run_test() -> tuple[str, Path, bool]:
    project_root = get_project_root()
    store_path = build_store_path("memory_mcp")
    log_path = build_log_path("test_memory_mcp")

    client = build_mcp_client(project_root, "alice", store_path)
    tool_names = await fetch_tool_names(client)
    registration_passed, missing_tools = verify_registered_tools(tool_names)
    client_passed, client_failures = run_client_tool_tests(store_path, log_path)

    result_text = format_result(
        tool_names,
        registration_passed,
        missing_tools,
        client_passed,
        client_failures,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result_text + "\n", encoding="utf-8")
    overall_pass = registration_passed and client_passed
    return result_text, log_path, overall_pass


if __name__ == "__main__":
    output, log_file, passed = asyncio.run(run_test())
    print(output)
    print(f"\nLog written to: {log_file}")
    if not passed:
        raise SystemExit(1)
