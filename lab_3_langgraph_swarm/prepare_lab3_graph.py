from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.graph_ingestion import insert_structured_seed_to_neo4j


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate the Lab 3 Neo4j database from the committed structured professor "
            "seed set. No crawling, OCR, or LLM calls are required. Start Neo4j first "
            "with `docker compose up -d neo4j`."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests on a smaller professor subset.",
    )
    parser.add_argument(
        "--only-slugs",
        type=str,
        default="",
        help="Optional comma-separated professor slugs for targeted seed loads.",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="",
        help="Optional override for the structured seed directory.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the current Neo4j database before loading the structured seed set.",
    )
    return parser.parse_args()


def parse_only_slugs(value: str) -> list[str]:
    return [slug.strip() for slug in value.split(",") if slug.strip()]


def main() -> None:
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"), require_llm=False)
    summary = insert_structured_seed_to_neo4j(
        settings=settings,
        project_root=settings.project_root,
        seed_dir=Path(args.seed_dir) if args.seed_dir else None,
        only_slugs=parse_only_slugs(args.only_slugs),
        limit=args.limit,
        reset_database=args.reset,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
