"""Automated checks for coordinator memory session wiring (no Ollama).

Run:
  python tests/test_coordinator_session.py
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import SESSION_LOG_PATH_ENV
from src.agents.coordinator_agent import build_memory_mcp_config
from src.data_retrieval.memory_client import MEMORY_STORE_PATH_ENV, MEMORY_USER_ENV

LOGS_DIR = PROJECT_ROOT / "logs"


def test_memory_mcp_config_sets_session_env() -> None:
    log_path = LOGS_DIR / "test_coordinator_session.logs"
    store_path = LOGS_DIR / "test_coordinator_session.memory.json"
    config = build_memory_mcp_config(
        PROJECT_ROOT,
        sys.executable,
        "alice",
        log_path,
        store_path,
    )
    assert config["env"][MEMORY_USER_ENV] == "alice"
    assert config["env"][MEMORY_STORE_PATH_ENV] == str(store_path)
    assert config["env"][SESSION_LOG_PATH_ENV] == str(log_path)


async def test_build_coordinator_session_structure() -> None:
    from src.agents.coordinator_agent import build_coordinator_session
    from src.data_retrieval.memory_client import initialize_memory_store

    log_path = LOGS_DIR / "test_coordinator_build.logs"
    store_path = initialize_memory_store(
        "alice",
        LOGS_DIR / "test_coordinator_build.memory.json",
    )
    session = await build_coordinator_session(
        PROJECT_ROOT,
        "alice",
        log_path,
        store_path,
    )
    assert session["username"] == "alice"
    assert session["memory_tools"]
    assert session["cached_subagents"]
    tool_names = {tool.name for tool in session["memory_tools"]}
    assert "save_memory" in tool_names
    assert "search_memories" in tool_names


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_memory_mcp_config_sets_session_env,
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

    async_name = "test_build_coordinator_session_structure"
    try:
        asyncio.run(test_build_coordinator_session_structure())
        passed.append(async_name)
    except Exception as exc:
        failed.append(f"{async_name}: {exc}")

    return passed, failed


if __name__ == "__main__":
    passed_tests, failed_tests = run_all_tests()
    lines = [
        "Coordinator session tests",
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
