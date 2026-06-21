"""Persistent user-scoped memory storage for long-term agent recall."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.agents.agent_utils import append_session_log, get_project_root
from src.data_retrieval.database_client import SESSION_USER_ENV, normalize_username

MEMORY_STORE_PATH_ENV = "MEMORY_STORE_PATH"
MEMORY_USER_ENV = "MEMORY_USER"
MEMORY_ROOT = get_project_root() / "data" / "memory"
RECALL_LIMIT = 10


def get_memory_root() -> Path:
    return MEMORY_ROOT


def get_memory_store_path() -> Path | None:
    path_value = os.getenv(MEMORY_STORE_PATH_ENV, "").strip()
    if not path_value:
        return None
    return Path(path_value)


def get_user_memory_file(username: str) -> Path:
    normalized = normalize_username(username)
    return get_memory_root() / f"{normalized}.json"


def resolve_memory_store_path() -> Path:
    configured = get_memory_store_path()
    if configured is not None:
        return configured
    username = get_memory_user()
    return get_user_memory_file(username)


def get_memory_user() -> str:
    memory_user = os.getenv(MEMORY_USER_ENV, "").strip()
    if memory_user:
        return normalize_username(memory_user)
    session_user = os.getenv(SESSION_USER_ENV, "").strip()
    if session_user:
        return normalize_username(session_user)
    raise ValueError(
        f"{MEMORY_USER_ENV} or {SESSION_USER_ENV} must be set for memory operations."
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_store(username: str) -> dict:
    return {"user": normalize_username(username), "memories": []}


def read_store_file(store_path: Path) -> dict:
    if not store_path.exists():
        return empty_store(get_memory_user())
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Memory store must be a JSON object.")
    memories = payload.get("memories", [])
    if not isinstance(memories, list):
        raise ValueError("Memory store 'memories' must be a list.")
    return payload


def write_store_file(store_path: Path, payload: dict) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_memories() -> dict:
    store_path = resolve_memory_store_path()
    if not store_path.exists():
        return empty_store(get_memory_user())
    return read_store_file(store_path)


def save_store(payload: dict) -> None:
    write_store_file(resolve_memory_store_path(), payload)


def next_memory_id(memories: list[dict]) -> str:
    return f"mem-{len(memories) + 1:03d}"


def normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    return [tag.strip().lower() for tag in tags if tag and tag.strip()]


def log_memory_event(action: str, **fields) -> None:
    parts = [f"[MEMORY] action={action}"]
    for key, value in fields.items():
        parts.append(f"{key}={value!r}")
    append_session_log(" ".join(parts))


def format_error(message: str) -> str:
    return json.dumps({"error": message}, indent=2)


def format_memory_entry(entry: dict) -> str:
    return json.dumps(entry, indent=2)


def format_memory_list(memories: list[dict], count: int | None = None) -> str:
    payload = {"memories": memories, "count": count if count is not None else len(memories)}
    return json.dumps(payload, indent=2)


def format_delete_result(memory_id: str, deleted: bool) -> str:
    if deleted:
        return json.dumps({"deleted": True, "id": memory_id}, indent=2)
    return format_error(f"Memory '{memory_id}' not found.")


def save_memory(content: str, tags: list[str] | None = None) -> str:
    try:
        username = get_memory_user()
        cleaned_content = content.strip()
        if not cleaned_content:
            return format_error("Memory content must not be empty.")
        store = load_memories()
        entry = {
            "id": next_memory_id(store["memories"]),
            "content": cleaned_content,
            "tags": normalize_tags(tags),
            "created_at": utc_timestamp(),
        }
        store["user"] = username
        store["memories"].append(entry)
        save_store(store)
        log_memory_event("save", user=username, id=entry["id"])
        return format_memory_entry(entry)
    except ValueError as exc:
        return format_error(str(exc))


def memory_matches_query(memory: dict, query: str) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    content = str(memory.get("content", "")).lower()
    if needle in content:
        return True
    for tag in memory.get("tags", []):
        if needle in str(tag).lower():
            return True
    return False


def search_memories(query: str) -> str:
    try:
        username = get_memory_user()
        store = load_memories()
        matches = [
            memory
            for memory in store["memories"]
            if memory_matches_query(memory, query)
        ]
        log_memory_event(
            "search", user=username, query=query, count=len(matches)
        )
        return format_memory_list(matches)
    except ValueError as exc:
        return format_error(str(exc))


def list_memories() -> str:
    try:
        username = get_memory_user()
        store = load_memories()
        memories = store["memories"]
        log_memory_event("list", user=username, count=len(memories))
        return format_memory_list(memories)
    except ValueError as exc:
        return format_error(str(exc))


def delete_memory(memory_id: str) -> str:
    try:
        username = get_memory_user()
        store = load_memories()
        original_count = len(store["memories"])
        store["memories"] = [
            memory for memory in store["memories"] if memory.get("id") != memory_id
        ]
        deleted = len(store["memories"]) < original_count
        if deleted:
            save_store(store)
            log_memory_event("delete", user=username, id=memory_id)
        return format_delete_result(memory_id, deleted)
    except ValueError as exc:
        return format_error(str(exc))


def get_recent_memories(limit: int = RECALL_LIMIT) -> list[dict]:
    store = load_memories()
    return list(reversed(store["memories"]))[:limit]


def format_memory_recall_block(memories: list[dict], limit: int = RECALL_LIMIT) -> str:
    recent = memories[:limit]
    if not recent:
        return ""
    lines = ["Known facts about this user (from prior sessions):"]
    for memory in recent:
        lines.append(f"- {memory.get('content', '')}")
    return "\n".join(lines)


def build_memory_recall_prompt(username: str, limit: int = RECALL_LIMIT) -> str:
    store_path = get_user_memory_file(username)
    if not store_path.exists():
        return ""
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    recent = list(reversed(payload.get("memories", [])))[:limit]
    return format_memory_recall_block(recent, limit=limit)


def initialize_memory_store(username: str, store_path: Path | None = None) -> Path:
    normalized = normalize_username(username)
    resolved = store_path or get_user_memory_file(normalized)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        write_store_file(resolved, empty_store(normalized))
    return resolved
