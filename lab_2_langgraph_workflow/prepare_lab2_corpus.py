from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.ingestion import RAW_OCR_MARKDOWN_MODE_MESSAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instructor-only prep script for refreshing the Lab 2 markdown corpus with page-by-page OCR."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing on a smaller professor subset.",
    )
    parser.add_argument(
        "--artifact-namespace",
        default="lab2-prep",
        help="Artifact namespace for instructor-prep markdown runs.",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    print(
        json.dumps(
            {
                "status": "downstream_paused",
                "message": RAW_OCR_MARKDOWN_MODE_MESSAGE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
