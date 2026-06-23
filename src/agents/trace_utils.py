"""Opt-in local JSONL tracing for agent LLM and tool steps (no external services)."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_TRACING_ENV = "LOCAL_TRACING"
CLASSIFIED_REDACT_TOOLS = frozenset({"read_classified_file"})
REDACTED_CLASSIFIED = "[REDACTED classified content]"
PREVIEW_MAX_LEN = 300

_tracing_env_loaded = False
_local_tracing_enabled = False
_active_trace_path: Path | None = None
_active_session_name: str | None = None
_turn_counter = 0
_current_run_id: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tracing_env(project_root: Path | None = None) -> None:
    global _tracing_env_loaded, _local_tracing_enabled
    if _tracing_env_loaded:
        return
    root = project_root or PROJECT_ROOT
    load_dotenv(root / ".env")
    raw_value = os.getenv(LOCAL_TRACING_ENV, "false").strip().lower()
    _local_tracing_enabled = raw_value in {"true", "1", "yes"}
    _tracing_env_loaded = True


def is_local_tracing_enabled() -> bool:
    load_tracing_env()
    return _local_tracing_enabled


def create_trace_log_path(session_log_path: Path) -> Path:
    return session_log_path.with_suffix(".trace.jsonl")


def preview_text(value: Any, max_len: int = PREVIEW_MAX_LEN) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, default=str)
    return text[:max_len]


def redact_tool_output(tool_name: str, content: Any) -> str:
    if tool_name in CLASSIFIED_REDACT_TOOLS:
        return REDACTED_CLASSIFIED
    return preview_text(content)


def start_trace_session(session_name: str, session_log_path: Path) -> Path | None:
    global _active_trace_path, _active_session_name, _turn_counter, _current_run_id
    load_tracing_env()
    if not is_local_tracing_enabled():
        _active_trace_path = None
        _active_session_name = None
        _turn_counter = 0
        _current_run_id = None
        return None

    trace_path = create_trace_log_path(session_log_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _active_trace_path = trace_path
    _active_session_name = session_name
    _turn_counter = 0
    _current_run_id = None
    return trace_path


def get_active_trace_path() -> Path | None:
    return _active_trace_path


def get_current_trace_run_id() -> str | None:
    return _current_run_id


def begin_trace_turn() -> str:
    global _turn_counter, _current_run_id
    _turn_counter += 1
    run_id = f"turn-{_turn_counter}"
    _current_run_id = run_id
    return run_id


def format_run_id(base_run_id: str, retry_index: int | None = None) -> str:
    if retry_index is None or retry_index == 0:
        return base_run_id
    return f"{base_run_id}-retry-{retry_index}"


def append_trace_event(trace_path: Path, event: dict) -> None:
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, default=str) + "\n")
    except OSError:
        pass


def build_base_event(
    event_name: str,
    session_name: str,
    run_id: str,
    parent_run_id: str | None = None,
) -> dict:
    return {
        "ts": utc_timestamp(),
        "event": event_name,
        "session": session_name,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "name": None,
        "latency_ms": None,
        "input_preview": None,
        "output_preview": None,
        "error": None,
    }


class LocalTraceHandler(BaseCallbackHandler):
    """Append structured trace events to a session JSONL file."""

    def __init__(
        self,
        trace_path: Path,
        session_name: str,
        run_id: str,
        parent_run_id: str | None = None,
    ) -> None:
        self.trace_path = trace_path
        self.session_name = session_name
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self._start_times: dict[UUID, float] = {}
        self._tool_names: dict[UUID, str] = {}

    def _write_event(self, event_name: str, **fields) -> None:
        try:
            event = build_base_event(
                event_name,
                self.session_name,
                self.run_id,
                self.parent_run_id,
            )
            event.update({key: value for key, value in fields.items() if value is not None})
            append_trace_event(self.trace_path, event)
        except Exception:
            pass

    def _record_start(self, run_id: UUID) -> None:
        self._start_times[run_id] = time.perf_counter()

    def _latency_ms(self, run_id: UUID) -> int | None:
        started = self._start_times.pop(run_id, None)
        if started is None:
            return None
        return int((time.perf_counter() - started) * 1000)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)
        name = serialized.get("name") or serialized.get("id")
        self._write_event("run_start", name=name)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._write_event("run_end", latency_ms=self._latency_ms(run_id))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "chain_error",
            latency_ms=self._latency_ms(run_id),
            error=str(error),
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)
        name = serialized.get("name") or serialized.get("id")
        input_preview = preview_text(prompts[0] if prompts else "")
        self._write_event("llm_start", name=name, input_preview=input_preview)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        output_preview = None
        generations = getattr(response, "generations", None)
        if generations and generations[0]:
            text = generations[0][0].text
            output_preview = preview_text(text)
        self._write_event(
            "llm_end",
            latency_ms=self._latency_ms(run_id),
            output_preview=output_preview,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)
        name = serialized.get("name") or serialized.get("id") or ""
        if name:
            self._tool_names[run_id] = str(name)
        self._write_event(
            "tool_start",
            name=name or None,
            input_preview=preview_text(input_str),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = self._tool_names.pop(run_id, "") or kwargs.get("name") or ""
        output_text = output if isinstance(output, str) else str(output)
        self._write_event(
            "tool_end",
            name=tool_name or None,
            latency_ms=self._latency_ms(run_id),
            output_preview=redact_tool_output(tool_name, output_text),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = self._tool_names.pop(run_id, "") or kwargs.get("name")
        self._write_event(
            "tool_end",
            name=tool_name,
            latency_ms=self._latency_ms(run_id),
            error=str(error),
        )


def build_invoke_config(
    session_name: str,
    run_id: str,
    trace_path: Path | None,
    parent_run_id: str | None = None,
) -> dict:
    if not is_local_tracing_enabled() or trace_path is None:
        return {}
    handler = LocalTraceHandler(
        trace_path=trace_path,
        session_name=session_name,
        run_id=run_id,
        parent_run_id=parent_run_id,
    )
    return {"callbacks": [handler]}


def prepare_turn_invoke_config() -> dict:
    if not is_local_tracing_enabled() or _active_trace_path is None:
        return {}
    run_id = begin_trace_turn()
    session_name = _active_session_name or "agent"
    return build_invoke_config(session_name, run_id, _active_trace_path)


def prepare_retry_invoke_config(retry_index: int) -> dict:
    if not is_local_tracing_enabled() or _active_trace_path is None or _current_run_id is None:
        return {}
    session_name = _active_session_name or "agent"
    run_id = format_run_id(_current_run_id, retry_index)
    return build_invoke_config(session_name, run_id, _active_trace_path)


def reset_tracing_state_for_tests() -> None:
    global _tracing_env_loaded, _local_tracing_enabled
    global _active_trace_path, _active_session_name, _turn_counter, _current_run_id
    _tracing_env_loaded = False
    _local_tracing_enabled = False
    _active_trace_path = None
    _active_session_name = None
    _turn_counter = 0
    _current_run_id = None


def prepare_subagent_invoke_config(subagent_name: str) -> dict:
    if not is_local_tracing_enabled() or _active_trace_path is None:
        return {}
    parent_run_id = _current_run_id
    run_id = f"{parent_run_id}/{subagent_name}" if parent_run_id else subagent_name
    session_name = _active_session_name or "coordinator_agent"
    return build_invoke_config(
        session_name,
        run_id,
        _active_trace_path,
        parent_run_id=parent_run_id,
    )
