"""Cross-session memory tests: save in session A, recall in session B (no Ollama).

Run:
  python tests/test_memory_sessions.py
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
from src.data_retrieval.memory_client import (
    MEMORY_STORE_PATH_ENV,
    MEMORY_USER_ENV,
    build_memory_recall_prompt,
    get_user_memory_file,
    load_memories,
    save_memory,
    search_memories,
)

LOGS_DIR = PROJECT_ROOT / "logs"
SAVED_FACT = "Favorite ticker is AAPL."
SAVED_TAG = "finance"


def build_store_path(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.memory.json"


def build_log_path(name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.logs"


@contextmanager
def memory_env(username: str, store_path: Path, log_path: Path | None = None):
    previous_user = os.environ.get(MEMORY_USER_ENV)
    previous_store = os.environ.get(MEMORY_STORE_PATH_ENV)
    previous_log = os.environ.get(SESSION_LOG_PATH_ENV)

    os.environ[MEMORY_USER_ENV] = username
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
        if previous_store is None:
            os.environ.pop(MEMORY_STORE_PATH_ENV, None)
        else:
            os.environ[MEMORY_STORE_PATH_ENV] = previous_store
        if previous_log is None:
            os.environ.pop(SESSION_LOG_PATH_ENV, None)
        else:
            os.environ[SESSION_LOG_PATH_ENV] = previous_log


def parse_response(response_text: str) -> dict:
    return json.loads(response_text)


def simulate_session_a(store_path: Path, log_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps({"user": "alice", "memories": []}, indent=2),
        encoding="utf-8",
    )
    log_path.write_text("", encoding="utf-8")

    with memory_env("alice", store_path, log_path):
        saved = parse_response(save_memory(SAVED_FACT, [SAVED_TAG]))
        assert saved.get("id") == "mem-001"


def simulate_session_b(store_path: Path) -> tuple[dict, str, dict]:
    with memory_env("alice", store_path, None):
        store = load_memories()
        recall_prompt = build_memory_recall_prompt("alice")
        search = parse_response(search_memories("ticker"))
    return store, recall_prompt, search


def test_cross_session_recall_from_disk() -> None:
    store_path = get_user_memory_file("alice")
    log_path = build_log_path("session_a_b")
    if store_path.exists():
        store_path.unlink()

    simulate_session_a(store_path, log_path)
    store, recall_prompt, search = simulate_session_b(store_path)

    assert len(store["memories"]) == 1
    assert store["memories"][0]["content"] == SAVED_FACT
    assert SAVED_FACT in recall_prompt
    assert search["count"] == 1
    store_path.unlink(missing_ok=True)


def test_recall_uses_user_file_not_env_store() -> None:
    store_path = get_user_memory_file("alice")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        store_path.unlink()

    with memory_env("alice", build_store_path("temp_write"), None):
        save_memory("Temporary env-only fact.")

    store_path.write_text(
        json.dumps(
            {
                "user": "alice",
                "memories": [
                    {
                        "id": "mem-001",
                        "content": SAVED_FACT,
                        "tags": [SAVED_TAG],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    recall_prompt = build_memory_recall_prompt("alice")
    assert SAVED_FACT in recall_prompt
    store_path.unlink(missing_ok=True)


def test_session_a_writes_memory_log_lines() -> None:
    store_path = build_store_path("session_log")
    log_path = build_log_path("session_log")

    simulate_session_a(store_path, log_path)
    log_text = log_path.read_text(encoding="utf-8")

    assert "[MEMORY]" in log_text
    assert "action='save'" in log_text or "action=save" in log_text


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_cross_session_recall_from_disk,
        test_recall_uses_user_file_not_env_store,
        test_session_a_writes_memory_log_lines,
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


if __name__ == "__main__":
    passed_tests, failed_tests = run_all_tests()
    lines = [
        "Cross-session memory tests",
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
