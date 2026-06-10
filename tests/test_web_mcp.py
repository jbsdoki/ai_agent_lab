"""Smoke test for the web MCP server tool registration.

Also checks allowlist rejection locally and optionally fetches an allowlisted URL.

Run:
  python tests/test_web_mcp.py
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"

WEB_MCP_TOOLS = ["fetch_url"]
LIVE_FETCH_URL = "https://news.ycombinator.com"
BLOCKED_FETCH_URL = "https://example.com/not-allowlisted"


def get_project_root() -> Path:
    return PROJECT_ROOT


def build_mcp_client(project_root: Path) -> MultiServerMCPClient:
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


async def fetch_tool_names(client: MultiServerMCPClient) -> list[str]:
    tools = await client.get_tools()
    return [tool.name for tool in tools]


def verify_registered_tools(tool_names: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in WEB_MCP_TOOLS if name not in tool_names]
    return not missing, missing


def run_allowlist_rejection_test() -> tuple[str, bool]:
    from src.data_retrieval.web_client import fetch_url

    response_text = fetch_url(BLOCKED_FETCH_URL)
    payload = json.loads(response_text)
    passed = "error" in payload and "allowlisted" in payload["error"].lower()
    return response_text, passed


def run_live_fetch_test() -> tuple[str, bool | None]:
    from src.data_retrieval.web_client import fetch_url

    try:
        response_text = fetch_url(LIVE_FETCH_URL)
    except requests.RequestException as exc:
        return str(exc), None

    payload = json.loads(response_text)
    if "error" in payload:
        return response_text, False

    passed = (
        payload.get("url") == LIVE_FETCH_URL
        and bool(payload.get("title") or payload.get("text"))
    )
    return response_text, passed


def format_result(
    tool_names: list[str],
    registration_passed: bool,
    missing_tools: list[str],
    allowlist_passed: bool,
    allowlist_response: str,
    live_fetch_attempted: bool,
    live_fetch_passed: bool | None,
    live_fetch_response: str | None,
) -> str:
    lines = [
        "Web MCP smoke test",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Project root: {PROJECT_ROOT}",
        f"Expected tools: {WEB_MCP_TOOLS}",
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
        "Allowlist rejection: PASS"
        if allowlist_passed
        else "Allowlist rejection: FAIL"
    )
    lines.append("Allowlist rejection response:")
    lines.extend(f"  {line}" for line in allowlist_response.splitlines())

    if live_fetch_passed is None:
        lines.append(f"Live fetch ({LIVE_FETCH_URL}): SKIPPED (network error)")
        if live_fetch_response:
            lines.append(f"  {live_fetch_response}")
    else:
        lines.append(
            f"Live fetch ({LIVE_FETCH_URL}): PASS"
            if live_fetch_passed
            else f"Live fetch ({LIVE_FETCH_URL}): FAIL"
        )
        if live_fetch_response:
            lines.append("Live fetch response preview:")
            preview = live_fetch_response.splitlines()[:8]
            lines.extend(f"  {line}" for line in preview)

    overall_pass = registration_passed and allowlist_passed and (
        live_fetch_passed is None or live_fetch_passed
    )
    lines.append("")
    lines.append("Result: PASS" if overall_pass else "Result: FAIL")
    return "\n".join(lines)


def build_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"test_web_mcp_{timestamp}.logs"


def write_log_file(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


async def run_test() -> tuple[str, Path, bool]:
    project_root = get_project_root()
    client = build_mcp_client(project_root)
    tool_names = await fetch_tool_names(client)
    registration_passed, missing_tools = verify_registered_tools(tool_names)

    allowlist_response, allowlist_passed = run_allowlist_rejection_test()

    live_fetch_response, live_fetch_passed = run_live_fetch_test()

    result_text = format_result(
        tool_names,
        registration_passed,
        missing_tools,
        allowlist_passed,
        allowlist_response,
        live_fetch_attempted=True,
        live_fetch_passed=live_fetch_passed,
        live_fetch_response=live_fetch_response,
    )
    log_path = build_log_path(LOGS_DIR)
    write_log_file(log_path, result_text)

    overall_pass = registration_passed and allowlist_passed and (
        live_fetch_passed is None or live_fetch_passed
    )
    return result_text, log_path, overall_pass


if __name__ == "__main__":
    output, log_file, passed = asyncio.run(run_test())
    print(output)
    print(f"\nLog written to: {log_file}")
    if not passed:
        raise SystemExit(1)
