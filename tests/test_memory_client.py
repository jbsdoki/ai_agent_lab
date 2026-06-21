"""Unit tests for user-scoped long-term memory storage (no Ollama required).

Run:
  python tests/test_memory_client.py
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_utils import SESSION_LOG_PATH_ENV
from src.data_retrieval.database_client import SESSION_USER_ENV
from src.data_retrieval.memory_client import (
    MEMORY_STORE_PATH_ENV,
    MEMORY_USER_ENV,
    build_memory_recall_prompt,
    delete_memory,
    format_memory_recall_block,
    get_user_memory_file,
    list_memories,
    load_memories,
    save_memory,
    search_memories,
)

LOGS_DIR = PROJECT_ROOT / "logs"


@contextmanager
def memory_session(username: str, store_path: Path, log_path: Path | None = None):
    previous_user = os.environ.get(MEMORY_USER_ENV)
    previous_agent = os.environ.get(SESSION_USER_ENV)
    previous_store = os.environ.get(MEMORY_STORE_PATH_ENV)
    previous_log = os.environ.get(SESSION_LOG_PATH_ENV)

    os.environ[MEMORY_USER_ENV] = username
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps({"user": username, "memories": []}, indent=2),
        encoding="utf-8",
    )
    os.environ[MEMORY_STORE_PATH_ENV] = str(store_path)
    if log_path is not None:
        os.environ[SESSION_LOG_PATH_ENV] = str(log_path)

    try:
        yield
    finally:
        if previous_user is None:
            os.environ.pop(MEMORY_USER_ENV, None)
        else:
            os.environ[MEMORY_USER_ENV] = previous_user
        if previous_agent is None:
            os.environ.pop(SESSION_USER_ENV, None)
        else:
            os.environ[SESSION_USER_ENV] = previous_agent
        if previous_store is None:
            os.environ.pop(MEMORY_STORE_PATH_ENV, None)
        else:
            os.environ[MEMORY_STORE_PATH_ENV] = previous_store
        if previous_log is None:
            os.environ.pop(SESSION_LOG_PATH_ENV, None)
        else:
            os.environ[SESSION_LOG_PATH_ENV] = previous_log


def build_store_path(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.memory.json"


def parse_response(response_text: str) -> dict:
    return json.loads(response_text)


def test_save_and_reload_persists() -> None:
    store_path = build_store_path("save_reload")
    with memory_session("alice", store_path):
        saved = parse_response(save_memory("Favorite ticker is AAPL.", ["finance"]))
        assert saved["id"] == "mem-001"
        reloaded = load_memories()
        assert len(reloaded["memories"]) == 1
        assert reloaded["memories"][0]["content"] == "Favorite ticker is AAPL."


def test_user_isolation() -> None:
    alice_store = build_store_path("alice_memory")
    bob_store = build_store_path("bob_memory")

    with memory_session("alice", alice_store):
        save_memory("Alice prefers Celsius.")

    with memory_session("bob", bob_store):
        payload = parse_response(list_memories())
        assert payload["count"] == 0


def test_search_by_content_and_tag() -> None:
    store_path = build_store_path("search_memory")
    with memory_session("alice", store_path):
        save_memory("Favorite ticker is AAPL.", ["finance"])
        save_memory("Likes tea in the morning.", ["preference"])

        content_hits = parse_response(search_memories("ticker"))
        assert content_hits["count"] == 1

        tag_hits = parse_response(search_memories("finance"))
        assert tag_hits["count"] == 1


def test_delete_memory() -> None:
    store_path = build_store_path("delete_memory")
    with memory_session("alice", store_path):
        saved = parse_response(save_memory("Temporary fact."))
        memory_id = saved["id"]
        deleted = parse_response(delete_memory(memory_id))
        assert deleted["deleted"] is True
        remaining = parse_response(list_memories())
        assert remaining["count"] == 0


def test_recall_block_respects_limit() -> None:
    memories = [{"content": f"Fact {index}"} for index in range(12)]
    block = format_memory_recall_block(memories, limit=10)
    assert block.count("- Fact") == 10


def test_build_memory_recall_prompt_reads_disk() -> None:
    store_path = get_user_memory_file("carol")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        store_path.unlink()
    with memory_session("carol", store_path):
        save_memory("Primary watchlist: MSFT.")
    prompt = build_memory_recall_prompt("carol")
    assert "Primary watchlist: MSFT." in prompt
    store_path.unlink(missing_ok=True)


def test_get_user_memory_file_normalizes() -> None:
    assert get_user_memory_file("Alice").name == "alice.json"


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_save_and_reload_persists,
        test_user_isolation,
        test_search_by_content_and_tag,
        test_delete_memory,
        test_recall_block_respects_limit,
        test_build_memory_recall_prompt_reads_disk,
        test_get_user_memory_file_normalizes,
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
        "Memory client unit tests",
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
