"""Unit tests for intent_router (no Ollama required).

Run:
  python tests/test_intent_router.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.intent_router import (
    FILES_SUBAGENT,
    FINANCE_SUBAGENT,
    NEWS_SUBAGENT,
    SEC_SUBAGENT,
    WEB_SUBAGENT,
    contains_url,
    detect_subagent_intents,
    format_router_log_line,
    get_default_subagents,
    is_subagent_enabled,
    match_keywords,
)

LOGS_DIR = PROJECT_ROOT / "logs"


def assert_intents(prompt: str, expected: set[str]) -> None:
    actual = detect_subagent_intents(prompt)
    assert actual == expected, (
        f"prompt={prompt!r}\n  expected={sorted(expected)}\n  actual={sorted(actual)}"
    )


def test_finance_intent() -> None:
    assert_intents("What is AAPL trading at?", {FINANCE_SUBAGENT})
    assert_intents("MSFT ticker and market cap", {FINANCE_SUBAGENT})


def test_news_intent() -> None:
    assert_intents("Latest tech headlines", {NEWS_SUBAGENT})
    assert_intents("Breaking news about Apple", {NEWS_SUBAGENT})


def test_sec_intent() -> None:
    assert_intents("Find the 10-K filing for Apple", {SEC_SUBAGENT})
    assert_intents("Look up CIK in EDGAR", {SEC_SUBAGENT})


def test_web_intent_from_url() -> None:
    assert_intents(
        "Summarize https://example.com/report",
        {WEB_SUBAGENT},
    )


def test_web_intent_from_keywords() -> None:
    assert_intents("Please fetch url content from a website", {WEB_SUBAGENT})


def test_files_intent() -> None:
    assert_intents("Show me the readme in this repo", {FILES_SUBAGENT})


def test_default_when_no_match() -> None:
    assert_intents("Hello there", get_default_subagents())
    assert_intents("   ", get_default_subagents())


def test_aapl_price_does_not_enable_sec_or_web() -> None:
    intents = detect_subagent_intents("AAPL price")
    assert SEC_SUBAGENT not in intents
    assert WEB_SUBAGENT not in intents


def test_combined_intents() -> None:
    assert_intents(
        "Stock price and recent news for Apple",
        {FINANCE_SUBAGENT, NEWS_SUBAGENT},
    )
    assert_intents(
        "10-K filing and stock quote for AAPL",
        {FINANCE_SUBAGENT, SEC_SUBAGENT},
    )


def test_case_insensitive_matching() -> None:
    assert_intents("STOCK NEWS HEADLINES", {FINANCE_SUBAGENT, NEWS_SUBAGENT})


def test_match_keywords() -> None:
    assert match_keywords("Hello STOCK world", ("stock",))
    assert not match_keywords("Hello world", ("stock",))


def test_contains_url() -> None:
    assert contains_url("See https://example.com/path for details")
    assert contains_url("HTTP://EXAMPLE.COM")
    assert not contains_url("no link here")


def test_is_subagent_enabled() -> None:
    prompt = "Latest 10-Q filing"
    assert is_subagent_enabled(prompt, SEC_SUBAGENT)
    assert not is_subagent_enabled(prompt, WEB_SUBAGENT)


def test_format_router_log_line() -> None:
    line = format_router_log_line("test prompt", {NEWS_SUBAGENT, FINANCE_SUBAGENT})
    assert line.startswith("[ROUTER] enabled_subagents=")
    assert "'finance_subagent', 'news_subagent'" in line
    assert "prompt='test prompt'" in line


def build_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"test_intent_router_{timestamp}.logs"


def write_log_file(log_path: Path, lines: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_finance_intent,
        test_news_intent,
        test_sec_intent,
        test_web_intent_from_url,
        test_web_intent_from_keywords,
        test_files_intent,
        test_default_when_no_match,
        test_aapl_price_does_not_enable_sec_or_web,
        test_combined_intents,
        test_case_insensitive_matching,
        test_match_keywords,
        test_contains_url,
        test_is_subagent_enabled,
        test_format_router_log_line,
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
        "Intent router unit tests",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Project root: {PROJECT_ROOT}",
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
    log_path = build_log_path(LOGS_DIR)
    write_log_file(log_path, output.splitlines())

    print(output)
    print(f"\nLog written to: {log_path}")

    if failed_tests:
        raise SystemExit(1)
