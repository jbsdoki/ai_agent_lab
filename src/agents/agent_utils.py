"""Shared helpers for local Ollama agents.

Provides configuration, session logging, LLM setup, message parsing, and the
interactive chat loop used by finance_agent, news_agent, and coordinator_agent.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

################################################################################
#*************************** Shared configuration ******************************
################################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:latest"
EMBED_MODEL = "nomic-embed-text:latest"

# Points at the current session log file so any code can append to one log.
_active_session_log: Path | None = None
SESSION_LOG_PATH_ENV = "SESSION_LOG_PATH"


def get_project_root() -> Path:
    """Return the repository root (used for MCP server working directories)."""
    return PROJECT_ROOT


def get_logs_dir() -> Path:
    """Return the directory where session .logs files are written."""
    return LOGS_DIR


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()

################################################################################
#********************** Session log file management ****************************
################################################################################

def create_session_log_path(session_name: str) -> Path:
    """Build a timestamped log file path, e.g. logs/finance_agent_20260608.logs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{session_name}_{timestamp}.logs"


def set_active_session_log(log_path: Path | None) -> None:
    """Set which log file receives append_session_log() writes."""
    global _active_session_log
    _active_session_log = log_path


def get_active_session_log() -> Path | None:
    """Return the active session log path, or None if no session is running."""
    return _active_session_log


def resolve_session_log_path() -> Path | None:
    """Return the log file path from env (MCP subprocess) or the active session."""
    env_path = os.getenv(SESSION_LOG_PATH_ENV, "").strip()
    if env_path:
        return Path(env_path)
    return _active_session_log


def append_session_log(line: str) -> None:
    """Append one line to the active session log. No-op if no session is active."""
    log_path = resolve_session_log_path()
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def start_session_log(session_name: str) -> Path:
    """Create a new log file, mark it active, and write the session header."""
    from src.agents.trace_utils import get_active_trace_path, start_trace_session

    log_path = create_session_log_path(session_name)
    set_active_session_log(log_path)
    os.environ[SESSION_LOG_PATH_ENV] = str(log_path)
    start_trace_session(session_name, log_path)
    append_session_log(f"=== SESSION START: {session_name} ===")
    append_session_log(f"Timestamp (UTC): {utc_timestamp()}")
    trace_path = get_active_trace_path()
    if trace_path is not None:
        append_session_log(f"Trace log: {trace_path}")
    return log_path


def log_api_request(provider: str, action: str, params: dict) -> None:
    """Log an external API call (yfinance, NewsAPI) into the active session."""
    # Never write secrets such as apiKey into log files.
    safe_params = {key: value for key, value in params.items() if key != "apiKey"}
    append_session_log(
        f"[API] {provider} {action} params={json.dumps(safe_params, default=str)}"
    )

################################################################################
#*************************** LLM and message parsing ***************************
################################################################################

def build_llm(model: str = MODEL, base_url: str = OLLAMA_HOST) -> ChatOllama:
    """Create a ChatOllama instance pointed at the local Ollama server."""
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def build_embeddings(
    model: str = EMBED_MODEL,
    base_url: str = OLLAMA_HOST,
) -> OllamaEmbeddings:
    """Create an OllamaEmbeddings instance for vector indexing and search."""
    return OllamaEmbeddings(model=model, base_url=base_url)


def extract_reply(result: dict) -> str:
    """Pull the final AI text response from a LangChain agent result dict."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "No response from agent."


def extract_tool_calls(messages: list) -> list[dict]:
    """Collect tool calls and tool responses from an agent message history."""
    entries = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                entries.append(
                    {
                        "type": "tool_call",
                        "tool": tool_call.get("name"),
                        "args": tool_call.get("args", {}),
                    }
                )
        if isinstance(message, ToolMessage):
            content = str(message.content)
            entries.append(
                {
                    "type": "tool_response",
                    "tool": message.name,
                    "content_preview": content[:300],
                }
            )
    return entries


################################################################################
#********************** Structured session logging helpers *********************
################################################################################

def log_tool_activity(messages: list, label: str) -> None:
    """Write all tool calls/responses from a message list to the session log."""
    entries = extract_tool_calls(messages)
    if not entries:
        append_session_log(f"[{label}] No tool calls recorded.")
        return

    for entry in entries:
        if entry["type"] == "tool_call":
            append_session_log(
                f"[{label} TOOL CALL] {entry['tool']} "
                f"args={json.dumps(entry['args'], default=str)}"
            )
        else:
            append_session_log(
                f"[{label} TOOL RESPONSE] {entry['tool']} "
                f"preview={entry['content_preview']!r}"
            )


def log_user_message(user_prompt: str) -> None:
    """Log the user's input for the current turn."""
    append_session_log(f"[USER] {user_prompt}")


def log_agent_reply(reply: str, label: str = "AGENT") -> None:
    """Log the final natural-language reply from an agent or subagent."""
    append_session_log(f"[{label} REPLY] {reply}")


def log_agent_result(result: dict, label: str) -> None:
    """Log tool activity from a full agent invoke() result."""
    messages = result.get("messages", [])
    log_tool_activity(messages, label)


def log_subagent_invocation(subagent_name: str, query: str, result: dict) -> None:
    """Log coordinator delegation: which subagent ran and what it returned."""
    append_session_log(f"[SUBAGENT INVOKE] {subagent_name} query={query!r}")
    log_agent_result(result, subagent_name.upper())
    append_session_log(
        f"[SUBAGENT REPLY] {subagent_name} {extract_reply(result)!r}"
    )


def log_router_decision(prompt: str, enabled_subagents: set[str]) -> None:
    """Log which subagents the intent router enabled for a turn."""
    from src.agents.intent_router import format_router_log_line

    append_session_log(format_router_log_line(prompt, enabled_subagents))


################################################################################
#*************************** Session identity helpers **************************
################################################################################

def read_username_input() -> str:
    return input("Enter username: ").strip()


def validate_username(username: str) -> str:
    from src.data_retrieval.database_client import get_user_clearance, normalize_username

    clearance = get_user_clearance(username)
    normalized = normalize_username(username)
    print(f"Signed in as {normalized} ({clearance} clearance).")
    return normalized


def prompt_session_username() -> str:
    """Prompt until the user enters a known username from config/users.json."""
    while True:
        username = read_username_input()
        if not username:
            print("Username is required.")
            continue
        try:
            return validate_username(username)
        except ValueError as exc:
            print(f"{exc} Try again.")


################################################################################
#*************************** Conversation history ******************************
################################################################################

def create_conversation_history() -> list:
    """Return an empty message list for within-session agent memory."""
    return []


def extend_conversation_history(conversation_messages: list, result: dict) -> None:
    """Replace the in-session history with the agent's returned message list."""
    result_messages = result.get("messages", [])
    if not result_messages:
        return
    conversation_messages.clear()
    conversation_messages.extend(result_messages)


################################################################################
#*************************** Agent interaction loop ****************************
################################################################################

async def run_prompt_with_history(
    agent,
    user_prompt: str,
    conversation_messages: list,
    log_label: str = "AGENT",
) -> str:
    """Send a user message with prior turns and update session history."""
    from src.agents.trace_utils import prepare_turn_invoke_config

    log_user_message(user_prompt)
    conversation_messages.append(HumanMessage(content=user_prompt))
    config = prepare_turn_invoke_config()
    result = await agent.ainvoke({"messages": conversation_messages}, config=config)
    extend_conversation_history(conversation_messages, result)
    log_agent_result(result, log_label)
    reply = extract_reply(result)
    log_agent_reply(reply, log_label)
    return reply


async def run_prompt(agent, user_prompt: str, log_label: str = "AGENT") -> str:
    """Send one stateless user message to an agent, log the turn, and return."""
    conversation_messages = create_conversation_history()
    return await run_prompt_with_history(
        agent, user_prompt, conversation_messages, log_label
    )


async def interactive_loop(
    agent,
    welcome_message: str,
    session_name: str = "agent",
) -> None:
    """Run a REPL chat loop until the user types quit or exit."""
    log_path = start_session_log(session_name)
    conversation_messages = create_conversation_history()
    print(welcome_message)
    print(f"Session log: {log_path}")
    from src.agents.trace_utils import get_active_trace_path

    trace_path = get_active_trace_path()
    if trace_path is not None:
        print(f"Trace log: {trace_path}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            append_session_log("=== SESSION END ===")
            break

        reply = await run_prompt_with_history(
            agent,
            user_prompt,
            conversation_messages,
            log_label=session_name.upper(),
        )
        print(f"\nAgent: {reply}\n")
