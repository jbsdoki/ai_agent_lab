# AI Agents Lab — Project Blueprint

A hands-on learning lab for building complex, interactive agentic systems using open-source and free tools. This document is your starting map: what to learn, in what order, and where to find the best references.

---

## 1. Project Vision

**Goal:** Experiment with how agents plan, use tools, maintain state, and interact with humans — without committing to a single vendor stack.

**Core stack (Phase 1):**

| Tool | Role |
|------|------|
| **Ollama** | Run LLMs locally (free, private, no API key required) |
| **LangChain / LangGraph** | Agent harness, tools, memory, orchestration |
| **MCP (Model Context Protocol)** | Standard way to expose tools and data to agents |
| **Cursor / Claude Desktop** | MCP clients for day-to-day experimentation |

**Stretch tools (Phase 2+):** CrewAI (multi-agent teams), LlamaIndex (RAG), Open WebUI (local chat UI), LangSmith (tracing — free tier available), Pydantic AI, Hugging Face Smolagents.

---

## 2. Recommended Learning Path

Work through these phases in order. Each phase produces a small, runnable artifact you can keep in this repo.

### Phase 0 — Environment Setup (Day 1)

- [ ] Install [Ollama for Windows](https://ollama.com/download)
- [ ] Pull a tool-capable model: `ollama pull llama3.2` or `ollama pull qwen2.5:7b`
- [ ] Verify the API: `curl http://localhost:11434/api/tags`
- [ ] Create a Python 3.11+ virtual environment in this repo
- [ ] Install baseline packages (adjust as you go):

```bash
pip install langchain langchain-ollama langgraph langchain-mcp-adapters python-dotenv
```

- [ ] Configure MCP in Cursor (Settings → MCP) so you can use servers while coding

**Success criteria:** You can chat with a local model via CLI (`ollama run llama3.2`) and via Python.

---

### Phase 1 — Ollama + LangChain Basics (Week 1)

Build a simple script that calls Ollama through LangChain.

**Learn:**
- Chat models vs completion models
- Prompt templates and message history
- Structured output (JSON mode where supported)

**Suggested experiments:**
1. Hello-world chat loop in Python
2. Summarize a local text file
3. Switch models without changing application logic

**Success criteria:** A Python script that uses `ChatOllama` and handles multi-turn conversation.

---

### Phase 2 — Tools and Single Agents (Week 2)

Add function calling / tool use so the model can act, not just respond.

**Learn:**
- LangChain `@tool` decorator
- `create_agent` (LangChain 1.x agent harness)
- ReAct pattern: reason → act → observe → repeat

**Suggested experiments:**
1. Calculator + datetime tools
2. Read/write files in an allowed directory
3. Call a public REST API (weather, GitHub, etc.)

**Success criteria:** An agent that chooses the right tool based on user input.

---

### Phase 3 — MCP Fundamentals (Week 3)

Separate **tools** from **agent logic** using MCP servers.

**Learn:**
- MCP concepts: tools, resources, prompts, transports (stdio vs HTTP)
- Build a minimal MCP server (Python FastMCP or TypeScript SDK)
- Register servers in Cursor / Claude Desktop
- Test with MCP Inspector: `npx @modelcontextprotocol/inspector`

**Suggested experiments:**
1. Filesystem MCP server pointed at a sandbox folder
2. Custom MCP server with 2–3 domain-specific tools
3. Compare: same tool as a Python function vs as an MCP server

**Success criteria:** A custom MCP server you can invoke from Cursor chat.

---

### Phase 4 — LangChain + MCP + Ollama Together (Week 4)

Wire everything into one local agent pipeline.

**Learn:**
- `langchain-mcp-adapters` and `MultiServerMCPClient`
- Connecting MCP tools to `create_agent`
- Using Ollama as the model backend for MCP-powered agents

**Suggested experiments:**
1. Agent that uses official filesystem + memory MCP servers
2. Agent that calls your custom MCP server + a web fetch tool
3. Log tool calls and model reasoning to stdout (later: LangSmith)

**Success criteria:** One Python entrypoint that loads MCP tools and runs a local Ollama agent.

---

### Phase 5 — Stateful & Complex Agents (Weeks 5–6)

Move from single-turn demos to durable, multi-step workflows.

**Learn:**
- LangGraph: `StateGraph`, nodes, edges, checkpointing
- Human-in-the-loop (approve before destructive actions)
- Memory across sessions
- Error recovery and retries

**Suggested experiments:**
1. Research agent: search → summarize → write report (multi-step graph)
2. Coding assistant with file edit approval step
3. Multi-agent handoff (researcher → writer) using LangGraph or CrewAI

**Success criteria:** An agent workflow that survives more than 3 tool calls without losing context.

---

### Phase 6 — Capstone Project (Ongoing)

Pick one interactive tool to build end-to-end. Examples:

- **Local knowledge assistant** — ingest PDFs/Markdown, RAG via LlamaIndex, tools via MCP
- **Dev workflow agent** — Git MCP + filesystem MCP + test runner
- **Personal ops dashboard** — calendar, notes, task list via MCP integrations
- **Game or simulation NPC** — stateful character with memory MCP server

Document each experiment in this repo with a short README per folder.

---

## 3. Suggested Repo Structure

```
AI_Agents/
├── blueprint.md              # This file
├── README.md                 # Quick start (create when first experiment lands)
├── .env.example              # API keys (optional providers)
├── requirements.txt          # Pin dependencies per phase
├── experiments/
│   ├── 01-ollama-chat/
│   ├── 02-langchain-tools/
│   ├── 03-mcp-server/
│   ├── 04-mcp-agent/
│   └── 05-langgraph-workflow/
├── mcp-servers/              # Custom MCP servers you build
└── notes/                    # Learning journal, decisions, gotchas
```

---

## 4. Documentation — How to Use Each Tool

### MCP (Model Context Protocol)

MCP is an open standard for connecting AI apps to external tools and data. You build **servers** (expose capabilities) and **clients** (Cursor, Claude Desktop, your Python agent) consume them.

| Resource | Link |
|----------|------|
| Official intro | https://modelcontextprotocol.io/docs/getting-started/intro |
| Full docs index (for LLMs) | https://modelcontextprotocol.io/llms.txt |
| Specification | https://modelcontextprotocol.io/specification |
| Build a server (Python) | https://modelcontextprotocol.io/docs/develop/build-server |
| Build a client | https://modelcontextprotocol.io/docs/develop/build-client |
| MCP Inspector (local testing) | https://github.com/modelcontextprotocol/inspector |
| Python SDK | https://github.com/modelcontextprotocol/python-sdk |
| TypeScript SDK | https://github.com/modelcontextprotocol/typescript-sdk |
| Cursor MCP docs | https://docs.cursor.com/context/model-context-protocol |
| Anthropic MCP announcement | https://www.anthropic.com/news/model-context-protocol |

**Quick mental model:**

```
┌─────────────┐     MCP protocol      ┌─────────────┐
│  MCP Client │ ◄──────────────────► │  MCP Server │
│ (Cursor,    │   tools / resources  │ (filesystem,│
│  your agent)│                      │  git, custom)│
└─────────────┘                      └─────────────┘
```

---

### LangChain & LangGraph

LangChain provides the agent **harness** (model + tools + prompt + loop). LangGraph provides explicit **graph-based orchestration** for complex, stateful workflows. In LangChain 1.x, `create_agent` runs on LangGraph under the hood.

| Resource | Link |
|----------|------|
| LangChain overview | https://docs.langchain.com/oss/python/langchain/overview |
| Python quickstart | https://docs.langchain.com/oss/python/langchain/quickstart |
| JavaScript quickstart | https://docs.langchain.com/oss/javascript/langchain/quickstart |
| LangGraph overview | https://docs.langchain.com/oss/python/langgraph/overview |
| LangGraph quickstart | https://docs.langchain.com/oss/python/langgraph/quickstart |
| MCP + LangChain integration | https://docs.langchain.com/oss/python/langchain/mcp |
| LangChain + Ollama integration | https://python.langchain.com/docs/integrations/chat/ollama/ |
| LangChain Academy (free courses) | https://academy.langchain.com/ |
| LangSmith tracing (optional) | https://docs.langchain.com/langsmith/home |
| Main GitHub repo | https://github.com/langchain-ai/langchain |
| LangGraph GitHub repo | https://github.com/langchain-ai/langgraph |

**Minimal agent pattern (LangChain 1.x):**

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
agent = create_agent(model=llm, tools=[...], system_prompt="You are a helpful assistant.")
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

---

### Ollama

Ollama runs open-weight models locally with a simple CLI and REST API on port `11434`. Free, offline-capable, and OpenAI-compatible API available.

| Resource | Link |
|----------|------|
| Download (Windows/macOS/Linux) | https://ollama.com/download |
| Quickstart | https://docs.ollama.com/quickstart |
| Full docs index | https://docs.ollama.com/llms.txt |
| Model library | https://ollama.com/library |
| REST API reference | https://docs.ollama.com/api |
| CLI reference | https://docs.ollama.com/cli |
| Modelfile (custom models) | https://docs.ollama.com/modelfile |
| GitHub repo | https://github.com/ollama/ollama |
| Open WebUI (optional local chat UI) | https://github.com/open-webui/open-webui |

**Essential commands:**

```bash
ollama pull llama3.2          # Download a model
ollama run llama3.2           # Interactive chat
ollama list                   # Installed models
ollama ps                     # Running models
```

**Python integration:**

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2", temperature=0)
response = llm.invoke("Explain MCP in one paragraph.")
```

---

## 5. GitHub Projects to Study

Review these repos to see patterns, project layout, and integration approaches. Star the ones that match your current phase.

### MCP — Official & Reference

| Project | Stars | Why study it |
|---------|-------|--------------|
| [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | Official | Reference implementations: filesystem, git, fetch, memory, postgres |
| [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | Official | Python server/client SDK |
| [modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk) | Official | TypeScript server/client SDK |
| [modelcontextprotocol/inspector](https://github.com/modelcontextprotocol/inspector) | Official | Debug and test MCP servers locally |
| [github/github-mcp-server](https://github.com/github/github-mcp-server) | High | Production-quality GitHub integration via MCP |
| [microsoft/mcp](https://github.com/microsoft/mcp) | Official | Microsoft's MCP server catalog |

### MCP — Curated Lists & Discovery

| Project | Why study it |
|---------|--------------|
| [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) | Largest curated list of community MCP servers |
| [appcypher/awesome-mcp-servers](https://github.com/appcypher/awesome-mcp-servers) | Well-organized list with client compatibility notes |
| [JustInCache/awesome-mcp-collection](https://github.com/JustInCache/awesome-mcp-collection) | Quality-focused collection with config examples |
| [rodert/awesome-mcp (site)](https://rodert.github.io/awesome-mcp/) | Searchable index of 4,800+ MCP projects |

### LangChain + Ollama + Agents

| Project | Why study it |
|---------|--------------|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Core framework; browse `libs/` for patterns |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Graph-based agent orchestration |
| [langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) | **Key bridge:** MCP tools → LangChain agents |
| [vikas-sachdeva/sample-ai-agent-ollama-langchain](https://github.com/vikas-sachdeva/sample-ai-agent-ollama-langchain) | Multi-provider agent with Ollama support |
| [PRASADNY/agents-langgraph-sample](https://github.com/PRASADNY/agents-langgraph-sample) | LangGraph + Ollama patterns: tools, memory, routing |
| [pinecone-io/examples (Ollama + LangGraph notebook)](https://github.com/pinecone-io/examples/blob/main/learn/generation/langchain/langgraph/02-ollama-langgraph-agent/02-ollama-langgraph-agent.ipynb) | End-to-end graph agent with Ollama |

### MCP + LangChain Combined

| Project | Why study it |
|---------|--------------|
| [langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) | `MultiServerMCPClient` + `create_agent` examples in README |
| [pietrozullo/mcp-use](https://github.com/pietrozullo/mcp-use) | Alternative MCP client framework with LangChain support |

### Multi-Agent & RAG (Phase 2+)

| Project | Why study it |
|---------|--------------|
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Role-based multi-agent teams; fast prototyping |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | RAG-first framework; great for document agents |
| [microsoft/autogen](https://github.com/microsoft/autogen) | Conversational multi-agent (see also Microsoft Agent Framework) |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | Minimal code-first agents (~1k LOC core) |

### Full-Stack Local AI Setups

| Project | Why study it |
|---------|--------------|
| [ollama/ollama](https://github.com/ollama/ollama) | Local model runtime itself |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | Web UI for Ollama with RAG and plugins |
| [langchain-ai/langgraph-example](https://github.com/langchain-ai/langgraph-example) | LangGraph deployment patterns |

---

## 6. Integration Architecture (Target State)

This is the architecture to aim for by Phase 4:

```
┌──────────────────────────────────────────────────────────┐
│                    Your Python Application               │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  LangGraph  │───►│ create_agent │───►│  ChatOllama  │ │
│  │  (optional) │    │  or graph    │    │  (local LLM) │ │
│  └─────────────┘    └──────┬───────┘    └──────────────┘ │
│                            │                             │
│                   langchain-mcp-adapters                 │
│                            │                             │
└────────────────────────────┼─────────────────────────────┘
                             │ MCP (stdio / HTTP)
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ Filesystem  │    │ Your Custom │    │ Git / Fetch │
  │ MCP Server  │    │ MCP Server  │    │ MCP Server  │
  └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 7. Model Selection Notes (Local / Free)

Not all local models handle tool calling equally well. Start with models known for function-calling support:

| Model | Size | Good for |
|-------|------|----------|
| `llama3.2` | 3B / 1B | Fast iteration, basic tools |
| `llama3.1:8b` | 8B | Better reasoning, tool use |
| `qwen2.5:7b` | 7B | Strong tool calling, coding |
| `mistral` | 7B | General agent tasks |
| `deepseek-r1:7b` | 7B | Reasoning-heavy workflows |

Pull with `ollama pull <model>` and benchmark on your hardware. On Windows with 16 GB RAM, prefer 3B–8B quantized models.

---

## 8. Cost & Privacy Posture

| Approach | Cost | Data leaves machine? |
|----------|------|----------------------|
| Ollama only | Free | No |
| LangChain + Ollama | Free | No |
| MCP local servers | Free | No (unless server calls external APIs) |
| LangSmith tracing | Free tier available | Yes (traces sent to LangSmith) |
| Cloud models (OpenAI, etc.) | Pay per token | Yes |

**Recommendation:** Stay on Ollama for learning. Add cloud models only when you need capabilities local models lack.

---

## 9. Common Pitfalls to Watch For

1. **Tool calling unsupported** — Model returns text instead of structured tool calls. Fix: switch model or enable JSON mode.
2. **MCP path errors on Windows** — Use absolute paths in MCP config; wrap `npx` with `cmd /c` when needed.
3. **Port conflicts** — Ollama uses `11434`; MCP HTTP servers need their own ports.
4. **LangChain API churn** — Prefer `create_agent` (LangChain 1.x) over deprecated `AgentExecutor`.
5. **Context window overflow** — Trim message history or summarize older turns in long sessions.
6. **Over-building early** — Finish Phase 1–3 before adding RAG, multi-agent, or cloud fallbacks.

---

## 10. Next Actions

1. Complete **Phase 0** environment setup on your Windows machine.
2. Create `experiments/01-ollama-chat/` with a minimal Python chat script.
3. Register the official [filesystem MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) in Cursor.
4. Skim [langchain-mcp-adapters README](https://github.com/langchain-ai/langchain-mcp-adapters) — this is the glue for Phase 4.
5. Keep a short log in `notes/` after each experiment (what worked, what broke, model used).

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **Agent** | LLM + tools + loop that can take actions to reach a goal |
| **Tool / Function calling** | Structured API the model invokes (search, write file, etc.) |
| **MCP Server** | Process that exposes tools/resources over the MCP protocol |
| **MCP Client** | App that connects to MCP servers (Cursor, your Python script) |
| **RAG** | Retrieval-Augmented Generation — grounding answers in your documents |
| **ReAct** | Reason + Act loop — model thinks, calls a tool, reads result, repeats |
| **LangGraph** | State-machine framework for multi-step agent workflows |
| **Checkpointing** | Saving agent state so workflows can pause, resume, or recover |

---

*Last updated: June 2025. Links and APIs evolve quickly — check official docs when something breaks.*
