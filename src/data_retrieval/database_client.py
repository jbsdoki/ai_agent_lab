"""Classified local file access with user clearance checks."""

import json
import os
from pathlib import Path

from src.agents.agent_utils import append_session_log, get_project_root

CLASSIFICATIONS = ("standard", "secret", "top_secret")
CLEARANCE_RANK = {
    "standard": 0,
    "secret": 1,
    "top_secret": 2,
}
SESSION_USER_ENV = "AGENT_USER"
USERS_CONFIG_PATH = get_project_root() / "config" / "users.json"
DATABASE_ROOT = get_project_root() / "database"


def get_users_config_path() -> Path:
    return USERS_CONFIG_PATH


def get_database_root() -> Path:
    return DATABASE_ROOT


def load_users() -> dict[str, dict]:
    config_path = get_users_config_path()
    if not config_path.exists():
        raise ValueError(f"Users config not found at {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Users config must be a JSON object.")
    return payload


def normalize_username(username: str) -> str:
    return username.strip().lower()


def get_user_clearance(username: str) -> str:
    users = load_users()
    normalized = normalize_username(username)
    if normalized not in users:
        raise ValueError(f"Unknown user '{username}'.")
    clearance = users[normalized].get("clearance")
    if clearance not in CLEARANCE_RANK:
        raise ValueError(f"Invalid clearance for user '{username}'.")
    return clearance


def get_session_user() -> str:
    username = os.getenv(SESSION_USER_ENV, "").strip()
    if not username:
        raise ValueError(
            f"{SESSION_USER_ENV} is not set. Session identity is required."
        )
    get_user_clearance(username)
    return normalize_username(username)


def clearance_allows(user_clearance: str, resource_classification: str) -> bool:
    return CLEARANCE_RANK[user_clearance] >= CLEARANCE_RANK[resource_classification]


def normalize_relative_path(relative_path: str) -> str:
    cleaned = relative_path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def reject_unsafe_path(relative_path: str) -> None:
    if not relative_path:
        raise ValueError("Path must not be empty.")
    if relative_path.startswith("/") or ".." in relative_path.split("/"):
        raise ValueError("Path must be relative to the database root.")


def get_resource_classification(relative_path: str) -> str:
    classification = relative_path.split("/", 1)[0]
    if classification not in CLASSIFICATIONS:
        raise ValueError(
            f"Invalid classification '{classification}'. "
            f"Expected one of: {', '.join(CLASSIFICATIONS)}."
        )
    return classification


def resolve_database_path(relative_path: str) -> Path:
    normalized = normalize_relative_path(relative_path)
    reject_unsafe_path(normalized)
    get_resource_classification(normalized)
    database_root = get_database_root().resolve()
    resolved = (database_root / normalized).resolve()
    if not str(resolved).startswith(str(database_root)):
        raise ValueError("Path escapes the database root.")
    return resolved


def parse_relative_path(relative_path: str) -> tuple[str, Path]:
    normalized = normalize_relative_path(relative_path)
    resolved = resolve_database_path(normalized)
    classification = get_resource_classification(normalized)
    return normalized, resolved


def log_access_attempt(
    username: str,
    action: str,
    resource: str,
    allowed: bool,
) -> None:
    append_session_log(
        f"[ACCESS] user={username} action={action} resource={resource!r} "
        f"allowed={allowed}"
    )


def format_error(message: str, hint: str | None = None) -> str:
    payload: dict = {"error": message}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, indent=2)


def format_access_denied(
    username: str,
    resource_path: str,
    classification: str,
    user_clearance: str,
) -> str:
    return json.dumps(
        {
            "error": "Access denied.",
            "user": username,
            "path": resource_path,
            "resource_classification": classification,
            "user_clearance": user_clearance,
            "hint": "Request a file within your clearance level.",
        },
        indent=2,
    )


def format_file_list(files: list[dict]) -> str:
    return json.dumps({"files": files, "count": len(files)}, indent=2)


def format_file_content(path: str, classification: str, content: str) -> str:
    return json.dumps(
        {
            "path": path,
            "classification": classification,
            "content": content,
        },
        indent=2,
    )


def list_files_in_classification(classification: str) -> list[dict]:
    folder = get_database_root() / classification
    if not folder.exists():
        return []

    files: list[dict] = []
    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(get_database_root()).as_posix()
        files.append({"path": relative, "classification": classification})
    return files


def list_accessible_files() -> str:
    try:
        username = get_session_user()
        clearance = get_user_clearance(username)
    except ValueError as exc:
        return format_error(str(exc))

    accessible: list[dict] = []
    for classification in CLASSIFICATIONS:
        if not clearance_allows(clearance, classification):
            continue
        accessible.extend(list_files_in_classification(classification))

    log_access_attempt(username, "list_accessible_files", "*", allowed=True)
    return format_file_list(accessible)


def read_classified_file(relative_path: str) -> str:
    try:
        username = get_session_user()
        normalized, resolved = parse_relative_path(relative_path)
        classification = get_resource_classification(normalized)
        clearance = get_user_clearance(username)
    except ValueError as exc:
        return format_error(str(exc))

    allowed = clearance_allows(clearance, classification)
    log_access_attempt(username, "read_classified_file", normalized, allowed)
    if not allowed:
        return format_access_denied(username, normalized, classification, clearance)

    if not resolved.is_file():
        return format_error(f"File not found: '{normalized}'.")

    content = resolved.read_text(encoding="utf-8")
    return format_file_content(normalized, classification, content)
