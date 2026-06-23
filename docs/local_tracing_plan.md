# Local Agent Tracing

Opt-in structured tracing for agent LLM and tool steps. All trace data stays on disk under `logs/`; nothing is sent to LangSmith or any external service.

## Enable tracing

Add to `.env`:

```env
LOCAL_TRACING=true
```

Default is `false`. Restart the agent after changing the setting.

## Output files

Each session produces a sibling JSONL file next to the existing session log:

```
logs/finance_agent_20260622_191405.logs
logs/finance_agent_20260622_191405.trace.jsonl
```

When tracing is enabled, the agent prints the trace path at session start alongside the session log path.

## Event schema

One JSON object per line:

```json
{
  "ts": "2026-06-22T19:14:05.828904+00:00",
  "event": "tool_end",
  "session": "finance_agent",
  "run_id": "turn-3",
  "parent_run_id": null,
  "name": "stock_quote",
  "latency_ms": 142,
  "input_preview": "{\"ticker\": \"AAPL\"}",
  "output_preview": "{\"price\": 190.0}",
  "error": null
}
```

Event types: `run_start`, `run_end`, `llm_start`, `llm_end`, `tool_start`, `tool_end`, `chain_error`.

Coordinator subagent delegation uses `parent_run_id` linking nested runs (e.g. `turn-2` parent, `turn-2/finance_subagent` child). Database approval loops use `turn-N-retry-M` run ids.

## Redaction

- Tool outputs are truncated to 300 characters (same preview limit as session logs).
- `read_classified_file` outputs are replaced with `[REDACTED classified content]` so classified file bodies never appear in trace files.

## Implementation

| File | Role |
|------|------|
| `src/agents/trace_utils.py` | Handler, env loading, invoke config builders |
| `src/agents/agent_utils.py` | Wires tracing into `start_session_log` and `run_prompt_with_history` |
| `src/agents/database_agent.py` | Approval-loop tracing with retry run ids |
| `src/agents/coordinator_agent.py` | Subagent delegation with `parent_run_id` |

## Tests

```bash
python tests/test_local_tracing.py
```

## Out of scope

- LangSmith cloud tracing
- Tracing inside MCP server subprocesses (only agent-side tool boundaries are traced)
- Trace viewer UI
