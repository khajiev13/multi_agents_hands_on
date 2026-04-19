from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.ingestion import refresh_professors_to_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instructor-only live refresh from the BIT professor directory through "
            "OCR. This phase stops after writing raw professor OCR markdown "
            "artifacts for manual review. No graph JSON, HTML review files, or "
            "Neo4j insertion are produced."
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
        help="Optional comma-separated professor slugs for targeted smoke refreshes.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent professor ingestions.",
    )
    parser.add_argument(
        "--artifact-namespace",
        type=str,
        default="lab1-pre-insertion-review",
        help="Artifact namespace for this pre-insertion pipeline run.",
    )
    return parser.parse_args()


def parse_only_slugs(value: str) -> list[str]:
    return [slug.strip() for slug in value.split(",") if slug.strip()]


def main() -> None:
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"))
    summary = refresh_professors_to_artifacts(
        settings=settings,
        project_root=settings.project_root,
        limit=args.limit,
        only_slugs=parse_only_slugs(args.only_slugs),
        max_concurrency=args.max_concurrency,
        artifact_namespace=args.artifact_namespace,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
