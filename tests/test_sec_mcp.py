"""Smoke test for the SEC MCP server tool registration.

Optional live lookup runs when SEC_USER_AGENT is set in .env.

Run:
  python tests/test_sec_mcp.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"

SEC_MCP_TOOLS = ["sec_lookup_cik", "sec_recent_filings"]
LIVE_LOOKUP_SYMBOL = "AAPL"


def get_project_root() -> Path:
    return PROJECT_ROOT


def build_mcp_client(project_root: Path) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "sec": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "src.mcp_servers.sec_server"],
                "cwd": str(project_root),
            }
        }
    )


async def fetch_tool_names(client: MultiServerMCPClient) -> list[str]:
    tools = await client.get_tools()
    return [tool.name for tool in tools]


def has_sec_user_agent() -> bool:
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.getenv("SEC_USER_AGENT"))


def run_live_lookup() -> tuple[str, bool]:
    from src.data_retrieval.sec_client import lookup_cik

    response_text = lookup_cik(LIVE_LOOKUP_SYMBOL)
    payload = json.loads(response_text)
    passed = "cik" in payload and payload.get("symbol") == LIVE_LOOKUP_SYMBOL
    return response_text, passed


def verify_registered_tools(tool_names: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in SEC_MCP_TOOLS if name not in tool_names]
    return not missing, missing


def format_result(
    tool_names: list[str],
    registration_passed: bool,
    missing_tools: list[str],
    live_lookup_attempted: bool,
    live_lookup_passed: bool | None,
    live_lookup_response: str | None,
) -> str:
    lines = [
        "SEC MCP smoke test",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Project root: {PROJECT_ROOT}",
        f"Expected tools: {SEC_MCP_TOOLS}",
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

    if live_lookup_attempted:
        lines.append(
            f"Live lookup ({LIVE_LOOKUP_SYMBOL}): PASS"
            if live_lookup_passed
            else f"Live lookup ({LIVE_LOOKUP_SYMBOL}): FAIL"
        )
        if live_lookup_response:
            lines.append("Live lookup response:")
            lines.extend(f"  {line}" for line in live_lookup_response.splitlines())
    else:
        lines.append("Live lookup: SKIPPED (SEC_USER_AGENT not set)")

    overall_pass = registration_passed and (
        live_lookup_passed is None or live_lookup_passed
    )
    lines.append("")
    lines.append("Result: PASS" if overall_pass else "Result: FAIL")
    return "\n".join(lines)


def build_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"test_sec_mcp_{timestamp}.logs"


def write_log_file(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


async def run_test() -> tuple[str, Path, bool]:
    project_root = get_project_root()
    client = build_mcp_client(project_root)
    tool_names = await fetch_tool_names(client)
    registration_passed, missing_tools = verify_registered_tools(tool_names)

    live_lookup_attempted = has_sec_user_agent()
    live_lookup_passed = None
    live_lookup_response = None
    if live_lookup_attempted:
        live_lookup_response, live_lookup_passed = run_live_lookup()

    result_text = format_result(
        tool_names,
        registration_passed,
        missing_tools,
        live_lookup_attempted,
        live_lookup_passed,
        live_lookup_response,
    )
    log_path = build_log_path(LOGS_DIR)
    write_log_file(log_path, result_text)

    overall_pass = registration_passed and (
        live_lookup_passed is None or live_lookup_passed
    )
    return result_text, log_path, overall_pass


if __name__ == "__main__":
    output, log_file, passed = asyncio.run(run_test())
    print(output)
    print(f"\nLog written to: {log_file}")
    if not passed:
        raise SystemExit(1)
