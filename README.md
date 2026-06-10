# AI Agents Lab

A hands-on learning lab for building agentic systems with local LLMs, LangChain, and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). The repo experiments with how agents plan, call tools, delegate work, and interact with external data — without committing to a single vendor stack.

For the full learning roadmap and reference links, see [blueprint.md](blueprint.md).

## What This Repo Does

The current implementation is a **multi-agent assistant** that runs entirely on your machine (via [Ollama](https://ollama.com/)) and pulls live data from two external sources:

| Domain | Data source | MCP tools |
|--------|-------------|-----------|
| Finance | [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance) | `stock_quote`, `stock_history` |
| News | [NewsAPI](https://newsapi.org/) | `news_search`, `news_headlines` |

You can run three entry points:

- **Coordinator agent** — routes questions to finance and news specialists, or calls both for mixed queries
- **Finance agent** — standalone stock and market data assistant
- **News agent** — standalone headlines and topic search assistant

Each agent session writes a timestamped log to `logs/` with user prompts, tool calls, API requests, and replies.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python agent layer                      │
│  ┌──────────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ coordinator_agent│  │finance_agent│  │  news_agent   │  │
│  │  (delegates to   │  │             │  │               │  │
│  │   subagents)     │  │             │  │               │  │
│  └────────┬─────────┘  └──────┬──────┘  └───────┬───────┘  │
│           │                   │                  │          │
│           └───────────────────┼──────────────────┘          │
│                               │                             │
│              langchain-mcp-adapters (MultiServerMCPClient)  │
│                               │                             │
│                    ChatOllama (llama3.2 via localhost)      │
└───────────────────────────────┼─────────────────────────────┘
                                │ MCP (stdio)
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   ┌─────────────────────┐             ┌─────────────────────┐
   │ yfinance_server     │             │ newsapi_server      │
   │ (FastMCP)           │             │ (FastMCP)           │
   └──────────┬──────────┘             └──────────┬──────────┘
              ▼                                   ▼
   ┌─────────────────────┐             ┌─────────────────────┐
   │ yfinance_client     │             │ newsapi_client      │
   └─────────────────────┘             └─────────────────────┘
```

**Layer breakdown:**

1. **Agents** (`src/agents/`) — LangChain `create_agent` loops backed by a local Ollama model. The coordinator wraps each specialist agent as a tool so the top-level agent can delegate.
2. **MCP servers** (`src/mcp_servers/`) — thin [FastMCP](https://github.com/modelcontextprotocol/python-sdk) wrappers that expose domain tools over stdio.
3. **Data retrieval** (`src/data_retrieval/`) — fetch and format data from Yahoo Finance and NewsAPI; shared by the MCP servers.
4. **Shared utilities** (`src/agents/agent_utils.py`) — LLM setup, interactive chat loop, and session logging.

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** running locally with a tool-capable model:

  ```bash
  ollama pull llama3.2
  ```

- **NewsAPI key** (free tier) for the news agent — copy `.env.example` to `.env` and set `NEWSAPI_API_KEY`

## Setup

```bash
# From the project root
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Verify Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

## Running the Agents

Run from the project root so MCP server subprocesses resolve correctly:

```bash
# Multi-agent coordinator (finance + news)
python -m src.agents.coordinator_agent

# Standalone specialists
python -m src.agents.finance_agent
python -m src.agents.news_agent
```

Example prompts:

- Coordinator: *"What is AAPL trading at and any recent Apple news?"*
- Finance: *"What is MSFT trading at?"*
- News: *"Latest AI headlines"*

Type `quit` or `exit` to end a session. Logs are written to `logs/<session_name>_<timestamp>.logs`.

## Testing

Smoke test that the NewsAPI MCP server registers its tools:

```bash
python -m tests.test_newsapi_mcp
```

## Project Layout

```
AI_agent_lab/
├── blueprint.md                 # Learning path, references, and target architecture
├── README.md                    # This file
├── .env.example                 # NEWSAPI_API_KEY template
├── requirements.txt             # Pinned Python dependencies
├── logs/                        # Session and test logs (gitignored)
├── src/
│   ├── agents/
│   │   ├── agent_utils.py       # LLM, logging, interactive loop
│   │   ├── coordinator_agent.py # Top-level multi-agent orchestrator
│   │   ├── finance_agent.py     # Standalone finance agent
│   │   └── news_agent.py        # Standalone news agent
│   ├── data_retrieval/
│   │   ├── yfinance_client.py   # Stock quote and history fetching
│   │   └── newsapi_client.py    # News search and headlines fetching
│   ├── mcp_servers/
│   │   ├── yfinance_server.py   # MCP server for stock tools
│   │   └── newsapi_server.py    # MCP server for news tools
│   └── experiments/
│       └── 01-ollama-chat/      # Early Ollama connectivity check
└── tests/
    └── test_newsapi_mcp.py      # MCP tool registration smoke test
```

## Core Stack

| Tool | Role |
|------|------|
| [Ollama](https://docs.ollama.com/quickstart) | Run LLMs locally (free, private) |
| [LangChain / LangGraph](https://docs.langchain.com/oss/python/langchain/overview) | Agent harness, tools, orchestration |
| [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) | Bridge MCP tools into LangChain agents |
| [MCP](https://modelcontextprotocol.io/docs/develop/build-server) | Standard protocol for exposing tools to agents |

## References

- [Ollama quickstart](https://docs.ollama.com/quickstart)
- [Build an MCP server (Python)](https://modelcontextprotocol.io/docs/develop/build-server)
- [LangChain MCP integration](https://docs.langchain.com/oss/python/langchain/mcp)
