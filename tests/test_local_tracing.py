"""Unit tests for opt-in local JSONL tracing (no Ollama required).

Run:
  python tests/test_local_tracing.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.trace_utils import (
    LocalTraceHandler,
    REDACTED_CLASSIFIED,
    build_invoke_config,
    is_local_tracing_enabled,
    load_tracing_env,
    prepare_turn_invoke_config,
    redact_tool_output,
    reset_tracing_state_for_tests,
    start_trace_session,
)

LOGS_DIR = PROJECT_ROOT / "logs"


def build_trace_path(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.trace.jsonl"


def read_trace_events(trace_path: Path) -> list[dict]:
    if not trace_path.exists():
        return []
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_redact_classified_tool_output() -> None:
    secret_text = "KEYWORD: NIGHTSHADE-NOTES " + ("x" * 500)
    preview = redact_tool_output("read_classified_file", secret_text)
    assert preview == REDACTED_CLASSIFIED
    assert "NIGHTSHADE" not in preview


def test_redact_non_classified_tool_truncates() -> None:
    preview = redact_tool_output("stock_quote", "a" * 400)
    assert len(preview) == 300


def test_tracing_disabled_by_default() -> None:
    reset_tracing_state_for_tests()
    os.environ.pop("LOCAL_TRACING", None)
    load_tracing_env(PROJECT_ROOT)
    assert not is_local_tracing_enabled()
    config = build_invoke_config("finance_agent", "turn-1", build_trace_path("disabled"))
    assert config == {}


def test_tracing_enabled_adds_callbacks() -> None:
    reset_tracing_state_for_tests()
    os.environ["LOCAL_TRACING"] = "true"
    load_tracing_env(PROJECT_ROOT)
    trace_path = build_trace_path("enabled")
    config = build_invoke_config("finance_agent", "turn-1", trace_path)
    assert "callbacks" in config
    assert len(config["callbacks"]) == 1
    assert isinstance(config["callbacks"][0], LocalTraceHandler)


def test_handler_writes_tool_events_with_latency() -> None:
    reset_tracing_state_for_tests()
    trace_path = build_trace_path("handler")
    handler = LocalTraceHandler(
        trace_path=trace_path,
        session_name="finance_agent",
        run_id="turn-1",
    )
    run_id = uuid4()
    handler.on_tool_start({"name": "stock_quote"}, '{"ticker": "AAPL"}', run_id=run_id)
    time.sleep(0.01)
    handler.on_tool_end('{"price": 190.0}', run_id=run_id)

    events = read_trace_events(trace_path)
    assert len(events) == 2
    assert events[0]["event"] == "tool_start"
    assert events[0]["name"] == "stock_quote"
    assert events[1]["event"] == "tool_end"
    assert events[1]["latency_ms"] is not None
    assert events[1]["latency_ms"] >= 0


def test_handler_redacts_classified_tool_end() -> None:
    reset_tracing_state_for_tests()
    trace_path = build_trace_path("classified")
    handler = LocalTraceHandler(
        trace_path=trace_path,
        session_name="database_agent",
        run_id="turn-1",
    )
    run_id = uuid4()
    handler.on_tool_start({"name": "read_classified_file"}, '{"path": "secret/x.txt"}', run_id=run_id)
    handler.on_tool_end("TOP SECRET CONTENT", run_id=run_id)

    events = read_trace_events(trace_path)
    assert events[-1]["output_preview"] == REDACTED_CLASSIFIED


def test_start_trace_session_creates_path_when_enabled() -> None:
    reset_tracing_state_for_tests()
    os.environ["LOCAL_TRACING"] = "true"
    load_tracing_env(PROJECT_ROOT)
    session_log = LOGS_DIR / "test_session.logs"
    trace_path = start_trace_session("finance_agent", session_log)
    assert trace_path is not None
    assert trace_path.name.endswith(".trace.jsonl")


def test_prepare_turn_invoke_config_increments_turns() -> None:
    reset_tracing_state_for_tests()
    os.environ["LOCAL_TRACING"] = "true"
    load_tracing_env(PROJECT_ROOT)
    session_log = LOGS_DIR / "turn_session.logs"
    start_trace_session("finance_agent", session_log)

    first = prepare_turn_invoke_config()
    second = prepare_turn_invoke_config()
    assert first["callbacks"][0].run_id == "turn-1"
    assert second["callbacks"][0].run_id == "turn-2"


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_redact_classified_tool_output,
        test_redact_non_classified_tool_truncates,
        test_tracing_disabled_by_default,
        test_tracing_enabled_adds_callbacks,
        test_handler_writes_tool_events_with_latency,
        test_handler_redacts_classified_tool_end,
        test_start_trace_session_creates_path_when_enabled,
        test_prepare_turn_invoke_config_increments_turns,
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
    reset_tracing_state_for_tests()
    os.environ.pop("LOCAL_TRACING", None)
    return passed, failed


if __name__ == "__main__":
    passed_tests, failed_tests = run_all_tests()
    lines = [
        "Local tracing unit tests",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Passed: {len(passed_tests)}",
        f"Failed: {len(failed_tests)}",
    ]
    if passed_tests:
        lines.extend(["Passing tests:", *[f"  - {name}" for name in passed_tests]])
    if failed_tests:
        lines.extend(["Failing tests:", *[f"  - {entry}" for entry in failed_tests]])
    lines.append("")
    lines.append("Result: PASS" if not failed_tests else "Result: FAIL")
    print("\n".join(lines))
    if failed_tests:
        raise SystemExit(1)
