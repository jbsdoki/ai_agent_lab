# RAG Vector Store

Local retrieval-augmented generation for the AI Agents Lab using Chroma + Ollama embeddings.

## Security boundary

Secret and top_secret material is **never** embedded or semantically searchable. Classified files are accessed only at runtime through the database MCP (`read_classified_file`) with clearance checks and operator grants.

| Content | RAG index | Database MCP |
|---------|-----------|--------------|
| `README.md` | Yes | N/A |
| `database/standard/**` | Yes | Optional full read |
| `database/secret/**` | **Never** | Grant required |
| `database/top_secret/**` | **Never** | Grant required |

Defense in depth:

- `config/rag_sources.json` lists only standard-safe paths
- `discover_source_files()` rejects paths under forbidden prefixes
- Ingest tests assert classified paths are excluded

## Architecture

```
README + database/standard  -->  ingest CLI  -->  Chroma (data/chroma/)
                                                      ^
rag_agent / rag_subagent  -->  rag_server MCP  -->  rag_client.py
```

## Prerequisites

```bash
ollama pull nomic-embed-text
pip install chromadb langchain-chroma langchain-text-splitters
```

Embedding model is configured in `src/agents/agent_utils.py` as `EMBED_MODEL`.

## Usage

Build or refresh the index:

```bash
python -m src.scripts.ingest_rag_corpus
python -m src.scripts.ingest_rag_corpus --force
```

Run the standalone RAG agent:

```bash
python -m src.agents.rag_agent
```

The coordinator routes document-search prompts to `rag_subagent` when keywords match (see `src/agents/intent_router.py`).

## MCP tools

| Tool | Purpose |
|------|---------|
| `search_documents` | Similarity search over indexed passages |
| `list_indexed_sources` | List indexed documents and chunk counts |

## Test corpus

Sample files under `database/standard/rag_corpus/` include distinctive KEYWORDs for automated tests:

- `RAG-POLICY-ALPHA` in `sample_policy.txt`
- `RAG-FAQ-BETA` in `sample_faq.md`
- `ORION-BRIEF` in `database/standard/public_briefing.txt`

Classified KEYWORDs (`NIGHTSHADE-NOTES`, `BLACKLANTERN-PLAN`) exist only in secret/top_secret files and must not appear in RAG search results.

## Tests

```bash
python tests/test_rag_client.py
python tests/test_rag_mcp.py
python tests/test_intent_router.py
```

Unit tests use `FakeEmbeddings` so Ollama is not required for CI-style runs.

## Related systems

| System | Role |
|--------|------|
| `database_agent` | Whole-file classified reads with grants |
| `memory_client` | Per-user facts and preferences |
| `FILES_SUBAGENT` | Reserved for future database coordinator wiring (not RAG) |
