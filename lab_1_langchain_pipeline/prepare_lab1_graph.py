from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.ingestion import refresh_professors_to_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instructor-only live refresh from the BIT professor directory. "
            "A full successful run also updates the committed Lab 3 graph seed files. "
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
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum number of concurrent professor ingestions.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Append to the current Neo4j database instead of clearing it first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"))
    summary = refresh_professors_to_graph(
        settings=settings,
        project_root=settings.project_root,
        limit=args.limit,
        max_concurrency=args.max_concurrency,
        reset_database=not args.skip_reset,
        artifact_namespace="lab1-full-refresh",
        graph_name="BIT_CSAT_FULL",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
