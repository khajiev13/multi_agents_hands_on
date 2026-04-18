from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.structured_seed import promote_structured_review_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote reviewed Lab 1 structured artifacts into the committed Lab 3 "
            "structured seed directory for student Neo4j loading."
        )
    )
    parser.add_argument(
        "--artifact-namespace",
        type=str,
        default="lab1-pre-insertion-review",
        help="Artifact namespace that contains reviewed *-structured.json files.",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="",
        help="Optional override for the committed structured seed directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke promotion runs.",
    )
    parser.add_argument(
        "--only-slugs",
        type=str,
        default="",
        help="Optional comma-separated professor slugs for targeted promotion.",
    )
    return parser.parse_args()


def parse_only_slugs(value: str) -> list[str]:
    return [slug.strip() for slug in value.split(",") if slug.strip()]


def main() -> None:
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"), require_llm=False)
    summary = promote_structured_review_artifacts(
        project_root=settings.project_root,
        artifact_namespace=args.artifact_namespace,
        seed_dir=Path(args.seed_dir) if args.seed_dir else None,
        only_slugs=parse_only_slugs(args.only_slugs),
        limit=args.limit,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
