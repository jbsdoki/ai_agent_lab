"""Integration test: routed coordinator delegates to subagents and MCP tools are called.

Requires:
  - Ollama running at http://localhost:11434
  - NEWSAPI_API_KEY set in .env (for news scenarios)
  - SEC_USER_AGENT set in .env (for sec scenarios)
  - conda env ai_agent_lab with project dependencies installed

Run:
  python tests/test_coordinator_delegation.py
"""

import ast
import asyncio
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import (
    append_session_log,
    get_active_session_log,
    start_session_log,
)
from src.agents.coordinator_agent import build_cached_subagents, run_coordinator_turn

LOGS_DIR = PROJECT_ROOT / "logs"
SESSION_NAME = "test_coordinator_delegation"

FINANCE_MCP_TOOLS = ["stock_quote", "stock_history"]
NEWS_MCP_TOOLS = ["news_search", "news_headlines"]
SEC_MCP_TOOLS = ["sec_lookup_cik", "sec_recent_filings"]
WEB_MCP_TOOLS = ["fetch_url"]

SUBAGENT_LOG_LABELS = [
    "FINANCE_SUBAGENT",
    "NEWS_SUBAGENT",
    "WEB_SUBAGENT",
    "SEC_SUBAGENT",
]

TEST_SCENARIOS = [
    {
        "name": "finance_only",
        "prompt": "What is the current stock price for AAPL?",
        "expect_dispatch": "direct",
        "expect_router_subagents": ["finance_subagent"],
        "expect_coordinator_tools": [],
        "expect_subagent_invokes": [],
        "expect_mcp_tool_groups": [FINANCE_MCP_TOOLS],
        "expect_forbidden_mcp_tools": SEC_MCP_TOOLS + WEB_MCP_TOOLS,
    },
    {
        "name": "finance_no_sec",
        "prompt": "What is the current stock quote for MSFT?",
        "expect_dispatch": "direct",
        "expect_router_subagents": ["finance_subagent"],
        "expect_coordinator_tools": [],
        "expect_subagent_invokes": [],
        "expect_mcp_tool_groups": [FINANCE_MCP_TOOLS],
        "expect_forbidden_mcp_tools": SEC_MCP_TOOLS + WEB_MCP_TOOLS,
    },
    {
        "name": "news_only",
        "prompt": "Search for recent news headlines about Apple.",
        "expect_dispatch": "direct",
        "expect_router_subagents": ["news_subagent"],
        "expect_coordinator_tools": [],
        "expect_subagent_invokes": [],
        "expect_mcp_tool_groups": [NEWS_MCP_TOOLS],
        "expect_forbidden_mcp_tools": SEC_MCP_TOOLS + WEB_MCP_TOOLS,
    },
    {
        "name": "sec_only",
        "prompt": "Show recent 10-K filings for AAPL.",
        "expect_dispatch": "direct",
        "expect_router_subagents": ["sec_subagent"],
        "expect_coordinator_tools": [],
        "expect_subagent_invokes": [],
        "expect_mcp_tool_groups": [SEC_MCP_TOOLS],
        "expect_forbidden_mcp_tools": FINANCE_MCP_TOOLS + WEB_MCP_TOOLS,
        "requires_sec_user_agent": True,
    },
    {
        "name": "web_only",
        "prompt": "Open this link https://www.apple.com and summarize the page text.",
        "expect_dispatch": "direct",
        "expect_router_subagents": ["web_subagent"],
        "expect_coordinator_tools": [],
        "expect_subagent_invokes": [],
        "expect_mcp_tool_groups": [WEB_MCP_TOOLS],
        "expect_forbidden_mcp_tools": FINANCE_MCP_TOOLS + SEC_MCP_TOOLS,
    },
    {
        "name": "both",
        "prompt": (
            "What is AAPL trading at right now, and what are recent Apple news headlines?"
        ),
        "expect_dispatch": "coordinator",
        "expect_router_subagents": ["finance_subagent", "news_subagent"],
        "expect_coordinator_tools": ["finance_subagent", "news_subagent"],
        "expect_subagent_invokes": ["finance_subagent", "news_subagent"],
        "expect_mcp_tool_groups": [FINANCE_MCP_TOOLS, NEWS_MCP_TOOLS],
        "expect_forbidden_mcp_tools": SEC_MCP_TOOLS + WEB_MCP_TOOLS,
    },
]


def get_project_root() -> Path:
    return PROJECT_ROOT


def has_sec_user_agent() -> bool:
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.getenv("SEC_USER_AGENT"))


def read_log_text(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


def extract_scenario_log(log_text: str, scenario_name: str) -> str:
    """Return only log lines recorded during one test scenario."""
    lines = log_text.splitlines()
    scenario_lines = []
    capture = False

    for line in lines:
        if line == f"=== TEST SCENARIO: {scenario_name} ===":
            capture = True
            continue
        if capture and line.startswith("=== TEST SCENARIO:"):
            break
        if capture:
            scenario_lines.append(line)

    return "\n".join(scenario_lines)


def find_router_enabled_subagents(log_text: str) -> list[str]:
    for line in log_text.splitlines():
        if not line.startswith("[ROUTER] enabled_subagents="):
            continue
        match = re.search(r"enabled_subagents=(\[[^\]]*\])", line)
        if match:
            return ast.literal_eval(match.group(1))
    return []


def find_coordinator_tool_calls(log_text: str) -> list[str]:
    tools = []
    for line in log_text.splitlines():
        if "[COORDINATOR_AGENT TOOL CALL]" in line:
            tool_name = line.split("] ", 1)[1].split(" ", 1)[0]
            tools.append(tool_name)
    return tools


def find_subagent_invocations(log_text: str) -> list[str]:
    invocations = []
    for line in log_text.splitlines():
        if line.startswith("[SUBAGENT INVOKE] "):
            subagent_name = line.split("[SUBAGENT INVOKE] ", 1)[1].split(" ", 1)[0]
            invocations.append(subagent_name)
    return invocations


def find_mcp_tool_calls(log_text: str) -> list[str]:
    """Find MCP tools invoked by specialist subagents in the session log."""
    tools = []
    for line in log_text.splitlines():
        if " TOOL CALL]" not in line:
            continue
        if not any(label in line for label in SUBAGENT_LOG_LABELS):
            continue
        tool_name = line.split(" TOOL CALL] ", 1)[1].split(" ", 1)[0]
        tools.append(tool_name)
    return tools


def includes_all(found: list[str], expected: list[str]) -> bool:
    return all(item in found for item in expected)


def excludes_all(found: list[str], forbidden: list[str]) -> bool:
    return not any(item in found for item in forbidden)


def includes_any_from_groups(found: list[str], tool_groups: list[list[str]]) -> bool:
    """Pass when at least one tool from each expected group appears in the log."""
    return all(any(tool in found for tool in group) for group in tool_groups)


def router_matches(found: list[str], expected: list[str]) -> bool:
    return sorted(found) == sorted(expected)


def check_dispatch_mode(
    scenario: dict,
    coordinator_tools: list[str],
    subagent_invokes: list[str],
) -> bool:
    dispatch_mode = scenario["expect_dispatch"]
    if dispatch_mode == "direct":
        return not coordinator_tools and not subagent_invokes
    return includes_all(
        coordinator_tools, scenario["expect_coordinator_tools"]
    ) and includes_all(subagent_invokes, scenario["expect_subagent_invokes"])


def build_skipped_result(scenario: dict, reason: str) -> dict:
    return {
        "name": scenario["name"],
        "prompt": scenario["prompt"],
        "passed": True,
        "skipped": True,
        "skip_reason": reason,
        "checks": {},
        "found_router_subagents": [],
        "found_coordinator_tools": [],
        "found_subagent_invokes": [],
        "found_mcp_tools": [],
    }


def evaluate_scenario(
    scenario: dict,
    log_text: str,
    execution_error: str | None = None,
) -> dict:
    router_subagents = find_router_enabled_subagents(log_text)
    coordinator_tools = find_coordinator_tool_calls(log_text)
    subagent_invokes = find_subagent_invocations(log_text)
    mcp_tools = find_mcp_tool_calls(log_text)

    checks = {
        "router_intents": router_matches(
            router_subagents, scenario["expect_router_subagents"]
        ),
        "dispatch_mode": check_dispatch_mode(
            scenario, coordinator_tools, subagent_invokes
        ),
        "coordinator_delegation": includes_all(
            coordinator_tools, scenario["expect_coordinator_tools"]
        ),
        "subagent_invocation": includes_all(
            subagent_invokes, scenario["expect_subagent_invokes"]
        ),
        "mcp_tool_calls": includes_any_from_groups(
            mcp_tools, scenario["expect_mcp_tool_groups"]
        ),
        "forbidden_mcp_tools": excludes_all(
            mcp_tools, scenario.get("expect_forbidden_mcp_tools", [])
        ),
        "scenario_execution": execution_error is None,
    }

    if scenario.get("requires_sec_user_agent") and not has_sec_user_agent():
        checks["mcp_tool_calls"] = True

    passed = all(checks.values())

    return {
        "name": scenario["name"],
        "prompt": scenario["prompt"],
        "passed": passed,
        "checks": checks,
        "found_router_subagents": router_subagents,
        "found_coordinator_tools": coordinator_tools,
        "found_subagent_invokes": subagent_invokes,
        "found_mcp_tools": mcp_tools,
        "execution_error": execution_error,
    }


def format_checklist_result(result: dict) -> str:
    lines = [
        f"Scenario: {result['name']}",
        f"Prompt: {result['prompt']}",
    ]
    if result.get("skipped"):
        lines.append(f"Skipped: {result['skip_reason']}")
        lines.append("Scenario result: PASS (skipped)")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            f"Router subagents found: {result['found_router_subagents']}",
            f"Coordinator tools found: {result['found_coordinator_tools']}",
            f"Subagent invocations found: {result['found_subagent_invokes']}",
            f"MCP tools found: {result['found_mcp_tools']}",
        ]
    )
    if result.get("execution_error"):
        lines.append(f"Execution error: {result['execution_error']}")
    lines.append("Checks:")
    for check_name, passed in result["checks"].items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"  - {check_name}: {status}")
    lines.append(f"Scenario result: {'PASS' if result['passed'] else 'FAIL'}")
    lines.append("")
    return "\n".join(lines)


def build_report_log_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"test_coordinator_delegation_{timestamp}.logs"


def write_report_log(report_path: Path, report_text: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")


def build_report_header(session_log_path: Path) -> str:
    return "\n".join(
        [
            "Coordinator delegation integration test",
            f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"Project root: {PROJECT_ROOT}",
            f"Session log reviewed: {session_log_path}",
            "",
        ]
    )


async def run_scenario(cached_subagents: dict, scenario: dict) -> dict:
    execution_error = None
    try:
        await run_coordinator_turn(cached_subagents, scenario["prompt"])
    except Exception as exc:
        execution_error = str(exc)
        append_session_log(f"[TEST ERROR] {scenario['name']}: {execution_error}")

    session_log = get_active_session_log()
    full_log = read_log_text(session_log) if session_log else ""
    scenario_log = extract_scenario_log(full_log, scenario["name"])
    return evaluate_scenario(scenario, scenario_log, execution_error)


async def run_all_scenarios() -> tuple[list[dict], Path, Path]:
    project_root = get_project_root()
    session_log_path = start_session_log(SESSION_NAME)
    cached_subagents = await build_cached_subagents(project_root)

    results = []
    for scenario in TEST_SCENARIOS:
        if scenario.get("requires_sec_user_agent") and not has_sec_user_agent():
            append_session_log(
                f"=== TEST SCENARIO: {scenario['name']} (SKIPPED) ==="
            )
            results.append(
                build_skipped_result(scenario, "SEC_USER_AGENT not set")
            )
            continue

        append_session_log(f"=== TEST SCENARIO: {scenario['name']} ===")
        results.append(await run_scenario(cached_subagents, scenario))

    append_session_log("=== SESSION END ===")

    report_lines = [build_report_header(session_log_path)]
    for result in results:
        report_lines.append(format_checklist_result(result))

    overall_pass = all(result["passed"] for result in results)
    report_lines.append(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")

    report_text = "\n".join(report_lines)
    report_log_path = build_report_log_path()
    write_report_log(report_log_path, report_text)

    return results, session_log_path, report_log_path


def print_summary(
    results: list[dict], session_log_path: Path, report_log_path: Path
) -> None:
    for result in results:
        print(format_checklist_result(result))
    print(f"Session log: {session_log_path}")
    print(f"Report log: {report_log_path}")
    print(f"Overall result: {'PASS' if all(r['passed'] for r in results) else 'FAIL'}")


async def main() -> int:
    results, session_log_path, report_log_path = await run_all_scenarios()
    print_summary(results, session_log_path, report_log_path)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
