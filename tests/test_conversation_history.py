"""Unit tests for within-session conversation history helpers.

Run:
  python tests/test_conversation_history.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.agent_utils import (
    create_conversation_history,
    extend_conversation_history,
)

LOGS_DIR = PROJECT_ROOT / "logs"


def test_create_conversation_history_starts_empty() -> None:
    history = create_conversation_history()
    assert history == []


def test_extend_conversation_history_replaces_messages() -> None:
    history = create_conversation_history()
    history.append(HumanMessage(content="Hello"))
    result = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there."),
        ]
    }
    extend_conversation_history(history, result)
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)


def test_extend_conversation_history_ignores_empty_result() -> None:
    history = create_conversation_history()
    history.append(HumanMessage(content="Hello"))
    extend_conversation_history(history, {"messages": []})
    assert len(history) == 1


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_create_conversation_history_starts_empty,
        test_extend_conversation_history_replaces_messages,
        test_extend_conversation_history_ignores_empty_result,
    ]
    passed: list[str] = []
    failed: list[str] = []
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed.append(name)
        except AssertionError as exc:
            failed.append(f"{name}: {exc}")
    return passed, failed


def format_result(passed: list[str], failed: list[str]) -> str:
    lines = [
        "Conversation history unit tests",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
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


if __name__ == "__main__":
    passed_tests, failed_tests = run_all_tests()
    output = format_result(passed_tests, failed_tests)
    print(output)
    if failed_tests:
        raise SystemExit(1)
