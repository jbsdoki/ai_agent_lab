"""Vector RAG over a standard-only document corpus (Chroma + Ollama embeddings)."""

import hashlib
import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.agent_utils import append_session_log, build_embeddings, get_project_root

RAG_SOURCES_PATH = get_project_root() / "config" / "rag_sources.json"
CHROMA_ROOT = get_project_root() / "data" / "chroma"
RAG_COLLECTION_NAME = "ai_agent_lab_corpus"
MANIFEST_FILENAME = "ingest_manifest.json"
CHROMA_PERSIST_DIR_ENV = "CHROMA_PERSIST_DIR"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_FORBIDDEN_PREFIXES = ("database/secret/", "database/top_secret/")
ALLOWED_INGEST_SUFFIXES = {".txt", ".md"}


def get_rag_sources_path() -> Path:
    return RAG_SOURCES_PATH


def get_chroma_root() -> Path:
    return CHROMA_ROOT


def resolve_chroma_persist_path() -> Path:
    override = os.getenv(CHROMA_PERSIST_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return get_chroma_root()


def get_manifest_path() -> Path:
    return resolve_chroma_persist_path() / MANIFEST_FILENAME


def load_rag_sources() -> dict:
    config_path = get_rag_sources_path()
    if not config_path.exists():
        raise ValueError(f"RAG sources config not found at {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RAG sources config must be a JSON object.")
    return payload


def get_forbidden_prefixes(config: dict | None = None) -> tuple[str, ...]:
    config = config or load_rag_sources()
    prefixes = config.get("forbidden_path_prefixes", list(DEFAULT_FORBIDDEN_PREFIXES))
    return tuple(str(prefix).replace("\\", "/") for prefix in prefixes)


def normalize_relative_path(path: str | Path) -> str:
    if isinstance(path, Path):
        relative = path.as_posix()
    else:
        relative = path.strip().replace("\\", "/")
    while relative.startswith("./"):
        relative = relative[2:]
    return relative.lstrip("/")


def is_forbidden_ingest_path(relative_path: str, config: dict | None = None) -> bool:
    normalized = normalize_relative_path(relative_path)
    for prefix in get_forbidden_prefixes(config):
        if normalized.startswith(prefix):
            return True
    return False


def reject_forbidden_ingest_path(relative_path: str, config: dict | None = None) -> None:
    if is_forbidden_ingest_path(relative_path, config):
        raise ValueError(
            f"Forbidden RAG ingest path '{relative_path}'. "
            "Classified files must not be embedded."
        )


def log_rag_event(action: str, **fields) -> None:
    parts = [f"[RAG] action={action}"]
    for key, value in fields.items():
        parts.append(f"{key}={value!r}")
    append_session_log(" ".join(parts))


def format_error(message: str) -> str:
    return json.dumps({"error": message}, indent=2)


def format_search_results(query: str, results: list[dict]) -> str:
    return json.dumps(
        {"query": query, "results": results, "count": len(results)},
        indent=2,
    )


def format_indexed_sources(sources: list[dict]) -> str:
    return json.dumps({"sources": sources, "count": len(sources)}, indent=2)


def format_ingest_summary(summary: dict) -> str:
    return json.dumps(summary, indent=2)


def file_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_ingest_manifest() -> dict:
    manifest_path = get_manifest_path()
    if not manifest_path.exists():
        return {"documents": {}}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ingest manifest must be a JSON object.")
    payload.setdefault("documents", {})
    return payload


def save_ingest_manifest(manifest: dict) -> None:
    manifest_path = get_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def resolve_source_path(source: dict) -> Path:
    return get_project_root() / normalize_relative_path(source["path"])


def file_matches_ingest_pattern(file_path: Path) -> bool:
    return file_path.suffix.lower() in ALLOWED_INGEST_SUFFIXES


def collect_files_from_source(source: dict) -> list[tuple[str, str]]:
    classification = source.get("classification", "standard")
    root_path = resolve_source_path(source)
    discovered: list[tuple[str, str]] = []

    if root_path.is_file():
        relative = normalize_relative_path(root_path.relative_to(get_project_root()))
        reject_forbidden_ingest_path(relative)
        discovered.append((relative, classification))
        return discovered

    if not root_path.is_dir():
        return discovered

    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file():
            continue
        if not file_matches_ingest_pattern(file_path):
            continue
        relative = normalize_relative_path(file_path.relative_to(get_project_root()))
        reject_forbidden_ingest_path(relative)
        discovered.append((relative, classification))
    return discovered


def discover_source_files(config: dict | None = None) -> list[tuple[str, str]]:
    config = config or load_rag_sources()
    files: list[tuple[str, str]] = []
    for source in config.get("sources", []):
        files.extend(collect_files_from_source(source))
    return files


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def split_document(text: str, metadata: dict) -> list[Document]:
    splitter = build_text_splitter()
    chunks = splitter.split_text(text)
    documents: list[Document] = []
    for index, chunk in enumerate(chunks):
        chunk_metadata = {
            **metadata,
            "chunk_index": index,
        }
        documents.append(Document(page_content=chunk, metadata=chunk_metadata))
    return documents


def build_vector_store(embeddings: Embeddings | None = None) -> Chroma:
    embedding_function = embeddings or build_embeddings()
    persist_path = resolve_chroma_persist_path()
    persist_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=RAG_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(persist_path),
    )


def delete_document_chunks(vector_store: Chroma, doc_id: str) -> None:
    collection = vector_store._collection
    existing = collection.get(where={"doc_id": doc_id})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def ingest_document(
    vector_store: Chroma,
    relative_path: str,
    classification: str,
    content: str,
) -> int:
    metadata = {
        "source_path": relative_path,
        "classification": classification,
        "doc_id": relative_path,
    }
    documents = split_document(content, metadata)
    if not documents:
        return 0
    delete_document_chunks(vector_store, relative_path)
    vector_store.add_documents(documents)
    return len(documents)


def ingest_corpus(
    embeddings: Embeddings | None = None,
    force: bool = False,
) -> dict:
    vector_store = build_vector_store(embeddings)
    manifest = load_ingest_manifest()
    documents_manifest = manifest["documents"]
    indexed = 0
    skipped = 0
    chunk_count = 0

    for relative_path, classification in discover_source_files():
        file_path = get_project_root() / relative_path
        content = file_path.read_text(encoding="utf-8")
        content_hash = file_content_hash(content)
        previous = documents_manifest.get(relative_path)
        if not force and previous and previous.get("hash") == content_hash:
            skipped += 1
            continue

        chunks = ingest_document(vector_store, relative_path, classification, content)
        documents_manifest[relative_path] = {
            "hash": content_hash,
            "classification": classification,
            "chunk_count": chunks,
        }
        indexed += 1
        chunk_count += chunks
        log_rag_event(
            "ingest",
            source=relative_path,
            chunks=chunks,
            classification=classification,
        )

    save_ingest_manifest(manifest)
    summary = {
        "indexed_files": indexed,
        "skipped_files": skipped,
        "total_chunks_written": chunk_count,
        "collection": RAG_COLLECTION_NAME,
    }
    log_rag_event("ingest_complete", **summary)
    return summary


def document_to_result(document: Document, score: float) -> dict:
    metadata = document.metadata or {}
    return {
        "source_path": metadata.get("source_path", ""),
        "classification": metadata.get("classification", "standard"),
        "chunk_index": metadata.get("chunk_index", 0),
        "score": round(float(score), 4),
        "content": document.page_content,
    }


def search_documents(
    query: str,
    max_results: int = 5,
    embeddings: Embeddings | None = None,
) -> str:
    try:
        cleaned_query = query.strip()
        if not cleaned_query:
            return format_error("Search query must not be empty.")

        vector_store = build_vector_store(embeddings)
        raw_results = vector_store.similarity_search_with_score(
            cleaned_query,
            k=max_results,
        )
        results = [
            document_to_result(document, score)
            for document, score in raw_results
        ]
        log_rag_event("search", query=cleaned_query, count=len(results))
        return format_search_results(cleaned_query, results)
    except Exception as exc:
        return format_error(str(exc))


def list_indexed_sources() -> str:
    try:
        manifest = load_ingest_manifest()
        sources = [
            {
                "doc_id": doc_id,
                "classification": entry.get("classification", "standard"),
                "chunk_count": entry.get("chunk_count", 0),
            }
            for doc_id, entry in sorted(manifest.get("documents", {}).items())
        ]
        log_rag_event("list_sources", count=len(sources))
        return format_indexed_sources(sources)
    except Exception as exc:
        return format_error(str(exc))
