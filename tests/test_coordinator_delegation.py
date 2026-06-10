"""Integration test: coordinator delegates to subagents and MCP tools are called.

Requires:
  - Ollama running at http://localhost:11434
  - NEWSAPI_API_KEY set in .env (for news scenarios)
  - conda env ai_agent_lab with project dependencies installed

Run:
  python tests/test_coordinator_delegation.py
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import (
    append_session_log,
    get_active_session_log,
    run_prompt,
    start_session_log,
)
from src.agents.coordinator_agent import build_coordinator_agent

LOGS_DIR = PROJECT_ROOT / "logs"
SESSION_NAME = "test_coordinator_delegation"

FINANCE_MCP_TOOLS = ["stock_quote", "stock_history"]
NEWS_MCP_TOOLS = ["news_search", "news_headlines"]

TEST_SCENARIOS = [
    {
        "name": "finance_only",
        "prompt": (
            "Delegate to the finance subagent only. "
            "What is the current stock price for AAPL?"
        ),
        "expect_coordinator_tools": ["finance_subagent"],
        "expect_subagent_invokes": ["finance_subagent"],
        "expect_mcp_tool_groups": [FINANCE_MCP_TOOLS],
    },
    {
        "name": "news_only",
        "prompt": (
            "Delegate to the news subagent only. "
            "Search for recent news headlines about Apple."
        ),
        "expect_coordinator_tools": ["news_subagent"],
        "expect_subagent_invokes": ["news_subagent"],
        "expect_mcp_tool_groups": [NEWS_MCP_TOOLS],
    },
    {
        "name": "both",
        "prompt": (
            "What is AAPL trading at right now, and what are recent Apple news headlines?"
        ),
        "expect_coordinator_tools": ["finance_subagent", "news_subagent"],
        "expect_subagent_invokes": ["finance_subagent", "news_subagent"],
        "expect_mcp_tool_groups": [FINANCE_MCP_TOOLS, NEWS_MCP_TOOLS],
    },
]


def get_project_root() -> Path:
    return PROJECT_ROOT


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
    """Find MCP tools invoked by finance/news subagents in the session log."""
    tools = []
    for line in log_text.splitlines():
        if " TOOL CALL]" not in line:
            continue
        if "FINANCE_SUBAGENT" not in line and "NEWS_SUBAGENT" not in line:
            continue
        tool_name = line.split(" TOOL CALL] ", 1)[1].split(" ", 1)[0]
        tools.append(tool_name)
    return tools


def includes_all(found: list[str], expected: list[str]) -> bool:
    return all(item in found for item in expected)


def includes_any_from_groups(found: list[str], tool_groups: list[list[str]]) -> bool:
    """Pass when at least one tool from each expected group appears in the log."""
    return all(any(tool in found for tool in group) for group in tool_groups)


def evaluate_scenario(scenario: dict, log_text: str) -> dict:
    coordinator_tools = find_coordinator_tool_calls(log_text)
    subagent_invokes = find_subagent_invocations(log_text)
    mcp_tools = find_mcp_tool_calls(log_text)

    checks = {
        "coordinator_delegation": includes_all(
            coordinator_tools, scenario["expect_coordinator_tools"]
        ),
        "subagent_invocation": includes_all(
            subagent_invokes, scenario["expect_subagent_invokes"]
        ),
        "mcp_tool_calls": includes_any_from_groups(
            mcp_tools, scenario["expect_mcp_tool_groups"]
        ),
    }
    passed = all(checks.values())

    return {
        "name": scenario["name"],
        "prompt": scenario["prompt"],
        "passed": passed,
        "checks": checks,
        "found_coordinator_tools": coordinator_tools,
        "found_subagent_invokes": subagent_invokes,
        "found_mcp_tools": mcp_tools,
    }


def format_checklist_result(result: dict) -> str:
    lines = [
        f"Scenario: {result['name']}",
        f"Prompt: {result['prompt']}",
        f"Coordinator tools found: {result['found_coordinator_tools']}",
        f"Subagent invocations found: {result['found_subagent_invokes']}",
        f"MCP tools found: {result['found_mcp_tools']}",
        "Checks:",
    ]
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


async def run_scenario(agent, scenario: dict) -> dict:
    await run_prompt(agent, scenario["prompt"], log_label="COORDINATOR_AGENT")
    session_log = get_active_session_log()
    full_log = read_log_text(session_log) if session_log else ""
    scenario_log = extract_scenario_log(full_log, scenario["name"])
    return evaluate_scenario(scenario, scenario_log)


async def run_all_scenarios() -> tuple[list[dict], Path, Path]:
    project_root = get_project_root()
    session_log_path = start_session_log(SESSION_NAME)
    agent = await build_coordinator_agent(project_root)

    results = []
    for scenario in TEST_SCENARIOS:
        append_session_log(f"=== TEST SCENARIO: {scenario['name']} ===")
        results.append(await run_scenario(agent, scenario))

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
