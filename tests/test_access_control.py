"""Unit tests for classified database access control (no Ollama required).

Run:
  python tests/test_access_control.py
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_retrieval.database_client import (
    SESSION_USER_ENV,
    clearance_allows,
    get_user_clearance,
    list_accessible_files,
    parse_relative_path,
    read_classified_file,
    reject_unsafe_path,
)

LOGS_DIR = PROJECT_ROOT / "logs"

STANDARD_FILE = "standard/public_briefing.txt"
SECRET_FILE = "secret/project_notes.txt"
TOP_SECRET_FILE = "top_secret/classified_plan.txt"


@contextmanager
def session_user(username: str | None):
    previous = os.environ.get(SESSION_USER_ENV)
    if username is None:
        os.environ.pop(SESSION_USER_ENV, None)
    else:
        os.environ[SESSION_USER_ENV] = username
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SESSION_USER_ENV, None)
        else:
            os.environ[SESSION_USER_ENV] = previous


def parse_response(response_text: str) -> dict:
    return json.loads(response_text)


def assert_allowed(response_text: str, expected_path: str) -> None:
    payload = parse_response(response_text)
    assert "error" not in payload, payload
    assert payload["path"] == expected_path
    assert payload.get("content")


def assert_denied(response_text: str, expected_path: str, expected_clearance: str) -> None:
    payload = parse_response(response_text)
    assert payload["error"] == "Access denied."
    assert payload["path"] == expected_path
    assert payload["user_clearance"] == expected_clearance


def assert_error(response_text: str, expected_fragment: str) -> None:
    payload = parse_response(response_text)
    assert "error" in payload
    assert expected_fragment in payload["error"]


def test_clearance_hierarchy() -> None:
    assert clearance_allows("standard", "standard")
    assert not clearance_allows("standard", "secret")
    assert clearance_allows("secret", "standard")
    assert clearance_allows("secret", "secret")
    assert not clearance_allows("secret", "top_secret")
    assert clearance_allows("top_secret", "top_secret")
    assert clearance_allows("top_secret", "secret")


def test_get_user_clearance() -> None:
    assert get_user_clearance("alice") == "top_secret"
    assert get_user_clearance("bob") == "secret"
    assert get_user_clearance("carol") == "standard"


def test_carol_reads_standard() -> None:
    with session_user("carol"):
        assert_allowed(read_classified_file(STANDARD_FILE), STANDARD_FILE)


def test_carol_denied_secret() -> None:
    with session_user("carol"):
        assert_denied(read_classified_file(SECRET_FILE), SECRET_FILE, "standard")


def test_carol_denied_top_secret() -> None:
    with session_user("carol"):
        assert_denied(
            read_classified_file(TOP_SECRET_FILE),
            TOP_SECRET_FILE,
            "standard",
        )


def test_bob_reads_secret() -> None:
    with session_user("bob"):
        assert_allowed(read_classified_file(SECRET_FILE), SECRET_FILE)


def test_bob_denied_top_secret() -> None:
    with session_user("bob"):
        assert_denied(
            read_classified_file(TOP_SECRET_FILE),
            TOP_SECRET_FILE,
            "secret",
        )


def test_alice_reads_all_classifications() -> None:
    with session_user("alice"):
        assert_allowed(read_classified_file(STANDARD_FILE), STANDARD_FILE)
        assert_allowed(read_classified_file(SECRET_FILE), SECRET_FILE)
        assert_allowed(read_classified_file(TOP_SECRET_FILE), TOP_SECRET_FILE)


def test_missing_agent_user() -> None:
    with session_user(None):
        assert_error(
            read_classified_file(STANDARD_FILE),
            f"{SESSION_USER_ENV} is not set",
        )


def test_unknown_user() -> None:
    with session_user("not_a_real_user"):
        assert_error(read_classified_file(STANDARD_FILE), "Unknown user")


def test_path_traversal_rejected() -> None:
    with session_user("alice"):
        assert_error(read_classified_file("../README.md"), "relative to the database root")


def test_invalid_classification() -> None:
    with session_user("alice"):
        assert_error(read_classified_file("public/data.txt"), "Invalid classification")


def test_reject_unsafe_path_helpers() -> None:
    try:
        reject_unsafe_path("../README.md")
        raise AssertionError("Expected ValueError for traversal path")
    except ValueError:
        pass


def test_parse_relative_path_normalizes() -> None:
    normalized, resolved = parse_relative_path("./standard/public_briefing.txt")
    assert normalized == STANDARD_FILE
    assert resolved.is_file()


def test_list_accessible_files_carol() -> None:
    with session_user("carol"):
        payload = parse_response(list_accessible_files())
    paths = [entry["path"] for entry in payload["files"]]
    assert paths == [STANDARD_FILE]
    assert payload["count"] == 1


def test_list_accessible_files_alice() -> None:
    with session_user("alice"):
        payload = parse_response(list_accessible_files())
    paths = [entry["path"] for entry in payload["files"]]
    assert STANDARD_FILE in paths
    assert SECRET_FILE in paths
    assert TOP_SECRET_FILE in paths
    assert payload["count"] == 3


def build_log_path(logs_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"test_access_control_{timestamp}.logs"


def write_log_file(log_path: Path, lines: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_clearance_hierarchy,
        test_get_user_clearance,
        test_carol_reads_standard,
        test_carol_denied_secret,
        test_carol_denied_top_secret,
        test_bob_reads_secret,
        test_bob_denied_top_secret,
        test_alice_reads_all_classifications,
        test_missing_agent_user,
        test_unknown_user,
        test_path_traversal_rejected,
        test_invalid_classification,
        test_reject_unsafe_path_helpers,
        test_parse_relative_path_normalizes,
        test_list_accessible_files_carol,
        test_list_accessible_files_alice,
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
        "Access control unit tests",
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
