"""Manual and integration checks for the classified database agent.

Automated checks (no Ollama):
  python tests/test_database_agent_manual.py

Live agent review (requires Ollama):
  python tests/test_database_agent_manual.py --live

Post-session log review after a manual run:
  python tests/test_database_agent_manual.py --check-log logs/database_agent_YYYYMMDD_HHMMSS.logs --user carol

Run:
  python tests/test_database_agent_manual.py
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import SESSION_LOG_PATH_ENV
from src.agents.database_agent import build_database_mcp_config
from src.data_retrieval.database_client import (
    GRANT_STORE_PATH_ENV,
    SESSION_USER_ENV,
    create_grant_store_path,
)

LOGS_DIR = PROJECT_ROOT / "logs"

MANUAL_SCENARIOS = [
    {
        "user": "carol",
        "clearance": "standard",
        "prompts": [
            "List the classified files I can access.",
            "Read standard/public_briefing.txt",
            "Read secret/project_notes.txt",
        ],
        "expect_access_allowed": ["standard/public_briefing.txt"],
        "expect_access_denied": ["secret/project_notes.txt"],
    },
    {
        "user": "alice",
        "clearance": "top_secret",
        "prompts": [
            "List all classified files I can access.",
            "Read top_secret/classified_plan.txt",
        ],
        "expect_access_allowed": ["top_secret/classified_plan.txt"],
        "expect_access_denied": [],
        "expect_grants": ["top_secret/classified_plan.txt"],
    },
]


def get_project_root() -> Path:
    return PROJECT_ROOT


def test_mcp_config_sets_session_env() -> None:
    log_path = LOGS_DIR / "test_database_mcp_session.logs"
    grant_path = LOGS_DIR / "test_database_mcp_session.grants.json"
    config = build_database_mcp_config(
        get_project_root(), sys.executable, "carol", log_path, grant_path
    )
    assert config["env"][SESSION_USER_ENV] == "carol"
    assert config["env"][SESSION_LOG_PATH_ENV] == str(log_path)
    assert config["env"][GRANT_STORE_PATH_ENV] == str(grant_path)


def test_prompt_session_username_accepts_known_user() -> None:
    from src.agents.agent_utils import prompt_session_username

    with patch("builtins.input", side_effect=["unknown", "carol"]):
        username = prompt_session_username()
    assert username == "carol"


def find_access_lines(log_text: str) -> list[str]:
    return [line for line in log_text.splitlines() if line.startswith("[ACCESS]")]


def access_line_allows_path(line: str, path: str, allowed: bool) -> bool:
    return (
        f"resource={path!r}" in line
        and f"allowed={allowed}" in line
    )


def find_grant_lines(log_text: str) -> list[str]:
    return [line for line in log_text.splitlines() if line.startswith("[GRANT]")]


def grant_line_matches(line: str, path: str, granted: bool) -> bool:
    return f"resource={path!r}" in line and f"granted={granted}" in line


def evaluate_manual_log(log_text: str, scenario: dict) -> dict:
    access_lines = find_access_lines(log_text)
    grant_lines = find_grant_lines(log_text)
    checks = {
        "has_access_logs": bool(access_lines),
        "allowed_reads": all(
            any(
                access_line_allows_path(line, path, True)
                for line in access_lines
            )
            for path in scenario["expect_access_allowed"]
        ),
        "denied_reads": all(
            any(
                access_line_allows_path(line, path, False)
                for line in access_lines
            )
            for path in scenario["expect_access_denied"]
        ),
        "grant_logs": all(
            any(grant_line_matches(line, path, True) for line in grant_lines)
            for path in scenario.get("expect_grants", [])
        ),
    }
    if not scenario.get("expect_grants"):
        checks["grant_logs"] = True
    return {
        "user": scenario["user"],
        "checks": checks,
        "passed": all(checks.values()),
        "access_lines": access_lines,
    }


def format_manual_checklist() -> str:
    lines = [
        "Classified database agent manual review",
        "",
        "Prerequisites: Ollama running, llama3.2 pulled",
        "",
        "For each user, run:",
        "  python -m src.agents.database_agent",
        "",
        "Scenarios:",
    ]
    for scenario in MANUAL_SCENARIOS:
        lines.append(f"- User: {scenario['user']} ({scenario['clearance']})")
        for prompt in scenario["prompts"]:
            lines.append(f"    Prompt: {prompt!r}")
        lines.append(
            f"    Expect allowed reads: {scenario['expect_access_allowed']}"
        )
        lines.append(
            f"    Expect denied reads: {scenario['expect_access_denied']}"
        )
        lines.append("")
    lines.append("Verify logs/database_agent_*.logs contains [ACCESS] and [GRANT] lines.")
    return "\n".join(lines)


def build_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"test_database_agent_manual_{timestamp}.logs"


def write_log_file(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def run_automated_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_mcp_config_sets_session_env,
        test_prompt_session_username_accepts_known_user,
    ]
    passed: list[str] = []
    failed: list[str] = []
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed.append(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
    return passed, failed


async def run_live_scenario(scenario: dict) -> dict:
    import os

    from src.agents.agent_utils import (
        append_session_log,
        get_active_session_log,
        start_session_log,
    )
    from src.agents.database_agent import build_database_agent, run_prompt_with_approvals

    session_name = f"test_database_agent_{scenario['user']}"
    log_path = start_session_log(session_name)
    grant_store_path = create_grant_store_path(log_path)
    grant_store_path.write_text("{}", encoding="utf-8")
    os.environ[GRANT_STORE_PATH_ENV] = str(grant_store_path)
    append_session_log(f"=== MANUAL SCENARIO: {scenario['user']} ===")

    agent = await build_database_agent(
        get_project_root(),
        scenario["user"],
        log_path,
        grant_store_path,
    )
    approval_count = len(scenario.get("expect_grants", []))
    approval_inputs = ["y"] * max(approval_count, 1)
    with patch("builtins.input", side_effect=approval_inputs):
        for prompt in scenario["prompts"]:
            append_session_log(f"=== PROMPT: {prompt!r} ===")
            await run_prompt_with_approvals(
                agent,
                prompt,
                scenario["user"],
                log_label="DATABASE_AGENT",
            )

    session_log = get_active_session_log()
    log_text = session_log.read_text(encoding="utf-8") if session_log else ""
    result = evaluate_manual_log(log_text, scenario)
    result["session_log"] = str(session_log) if session_log else ""
    return result


async def run_live_scenarios() -> list[dict]:
    results = []
    for scenario in MANUAL_SCENARIOS:
        results.append(await run_live_scenario(scenario))
    return results


def format_automated_result(passed: list[str], failed: list[str]) -> str:
    lines = [
        "Database agent automated checks",
        f"Passed: {len(passed)}",
        f"Failed: {len(failed)}",
    ]
    if passed:
        lines.append("Passing tests:")
        lines.extend(f"  - {name}" for name in passed)
    if failed:
        lines.append("Failing tests:")
        lines.extend(f"  - {entry}" for entry in failed)
    lines.append("")
    lines.append("Result: PASS" if not failed else "Result: FAIL")
    return "\n".join(lines)


def format_live_results(results: list[dict]) -> str:
    lines = ["Database agent live scenario results", ""]
    for result in results:
        lines.append(f"User: {result['user']}")
        lines.append(f"Session log: {result.get('session_log', '')}")
        for check_name, passed in result["checks"].items():
            lines.append(f"  - {check_name}: {'PASS' if passed else 'FAIL'}")
        lines.append(f"Scenario result: {'PASS' if result['passed'] else 'FAIL'}")
        lines.append("")
    overall = all(result["passed"] for result in results)
    lines.append(f"Overall result: {'PASS' if overall else 'FAIL'}")
    return "\n".join(lines)


def check_log_file(log_path: Path, username: str) -> int:
    scenario = next((item for item in MANUAL_SCENARIOS if item["user"] == username), None)
    if scenario is None:
        print(f"Unknown scenario user: {username}")
        return 1

    log_text = log_path.read_text(encoding="utf-8")
    result = evaluate_manual_log(log_text, scenario)
    print(format_live_results([result]))
    return 0 if result["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Database agent manual test harness")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live Ollama scenarios for alice and carol",
    )
    parser.add_argument(
        "--check-log",
        type=Path,
        help="Evaluate an existing database_agent session log",
    )
    parser.add_argument(
        "--user",
        choices=[scenario["user"] for scenario in MANUAL_SCENARIOS],
        help="User scenario to use with --check-log",
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="Print the manual review checklist only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.checklist:
        print(format_manual_checklist())
        return 0

    if args.check_log:
        if not args.user:
            print("--check-log requires --user")
            return 1
        return check_log_file(args.check_log, args.user)

    passed, failed = run_automated_tests()
    output = format_automated_result(passed, failed)
    output = output + "\n\n" + format_manual_checklist()
    log_path = build_log_path(LOGS_DIR)
    write_log_file(log_path, output)

    print(output)
    print(f"\nLog written to: {log_path}")

    if failed:
        return 1

    if args.live:
        results = asyncio.run(run_live_scenarios())
        live_output = format_live_results(results)
        print(live_output)
        return 0 if all(result["passed"] for result in results) else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
