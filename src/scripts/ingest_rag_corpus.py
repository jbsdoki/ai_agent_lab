"""CLI entrypoint to build or refresh the local RAG vector index."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_retrieval.rag_client import format_ingest_summary, ingest_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest standard-safe corpus into Chroma")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all source files even when unchanged",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ingest_corpus(force=args.force)
    print(format_ingest_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
