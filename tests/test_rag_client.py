"""Unit tests for RAG client (FakeEmbeddings; no live Ollama required).

Run:
  python tests/test_rag_client.py
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_retrieval.rag_client import (
    CHROMA_PERSIST_DIR_ENV,
    discover_source_files,
    ingest_corpus,
    is_forbidden_ingest_path,
    reject_forbidden_ingest_path,
    search_documents,
)

LOGS_DIR = PROJECT_ROOT / "logs"


@contextmanager
def temp_chroma_dir(name: str):
    chroma_dir = LOGS_DIR / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    previous = os.environ.get(CHROMA_PERSIST_DIR_ENV)
    os.environ[CHROMA_PERSIST_DIR_ENV] = str(chroma_dir)
    try:
        yield chroma_dir
    finally:
        if previous is None:
            os.environ.pop(CHROMA_PERSIST_DIR_ENV, None)
        else:
            os.environ[CHROMA_PERSIST_DIR_ENV] = previous


def parse_response(response_text: str) -> dict:
    return json.loads(response_text)


def test_forbidden_path_detection() -> None:
    assert is_forbidden_ingest_path("database/secret/project_notes.txt")
    assert is_forbidden_ingest_path("database/top_secret/classified_plan.txt")
    assert not is_forbidden_ingest_path("database/standard/public_briefing.txt")


def test_reject_forbidden_ingest_path_raises() -> None:
    try:
        reject_forbidden_ingest_path("database/secret/project_notes.txt")
        raise AssertionError("Expected ValueError for secret path")
    except ValueError:
        pass


def test_discover_source_files_excludes_classified() -> None:
    paths = {path for path, _classification in discover_source_files()}
    assert "database/secret/project_notes.txt" not in paths
    assert "database/top_secret/classified_plan.txt" not in paths
    assert "README.md" in paths
    assert "database/standard/public_briefing.txt" in paths


def test_ingest_and_search_standard_keyword() -> None:
    embeddings = FakeEmbeddings(size=32)
    with temp_chroma_dir("rag_client_ingest"):
        summary = ingest_corpus(embeddings=embeddings, force=True)
        assert summary["indexed_files"] >= 1

        payload = parse_response(
            search_documents("ORION-BRIEF", max_results=5, embeddings=embeddings)
        )
        assert payload["count"] >= 1
        assert any("ORION-BRIEF" in hit["content"] for hit in payload["results"])


def test_search_does_not_return_classified_keywords() -> None:
    embeddings = FakeEmbeddings(size=32)
    with temp_chroma_dir("rag_client_negative"):
        ingest_corpus(embeddings=embeddings, force=True)
        payload = parse_response(
            search_documents("NIGHTSHADE-NOTES", max_results=5, embeddings=embeddings)
        )
        for hit in payload["results"]:
            assert "NIGHTSHADE" not in hit["content"]
            assert not hit["source_path"].startswith("database/secret/")
            assert not hit["source_path"].startswith("database/top_secret/")


def test_search_rag_corpus_keywords() -> None:
    embeddings = FakeEmbeddings(size=32)
    with temp_chroma_dir("rag_client_corpus"):
        ingest_corpus(embeddings=embeddings, force=True)
        payload = parse_response(
            search_documents("RAG-POLICY-ALPHA", max_results=5, embeddings=embeddings)
        )
        assert payload["count"] >= 1


def run_all_tests() -> tuple[list[str], list[str]]:
    tests = [
        test_forbidden_path_detection,
        test_reject_forbidden_ingest_path_raises,
        test_discover_source_files_excludes_classified,
        test_ingest_and_search_standard_keyword,
        test_search_does_not_return_classified_keywords,
        test_search_rag_corpus_keywords,
    ]
    passed: list[str] = []
    failed: list[str] = []
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed.append(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")
    return passed, failed


if __name__ == "__main__":
    passed_tests, failed_tests = run_all_tests()
    lines = [
        "RAG client unit tests",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Passed: {len(passed_tests)}",
        f"Failed: {len(failed_tests)}",
    ]
    if passed_tests:
        lines.extend(["Passing tests:", *[f"  - {name}" for name in passed_tests]])
    if failed_tests:
        lines.extend(["Failing tests:", *[f"  - {entry}" for entry in failed_tests]])
    lines.append("")
    lines.append("Result: PASS" if not failed_tests else "Result: FAIL")
    print("\n".join(lines))
    if failed_tests:
        raise SystemExit(1)
