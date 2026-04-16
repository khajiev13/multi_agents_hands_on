from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.ingestion import restore_seeded_professors_to_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed the Lab 3 Neo4j database from committed local professor markdown and "
            "graph JSON files. No crawling, OCR, or LLM calls are required. "
            "Start Neo4j first with `docker compose up -d neo4j`."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests on a smaller professor subset.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Append to the current Neo4j database instead of clearing it first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"), require_llm=False)
    summary = restore_seeded_professors_to_graph(
        settings=settings,
        project_root=settings.project_root,
        limit=args.limit,
        reset_database=not args.skip_reset,
        graph_name="BIT_CSAT_PREPARED",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
