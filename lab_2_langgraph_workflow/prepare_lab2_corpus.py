from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.ingestion import (
    LISTING_URL,
    build_cached_markdown_result,
    build_requests_session,
    collect_professor_links,
    discover_listing_pages,
    ingest_professor,
    partition_professors_for_corpus,
    rebuild_markdown_corpus_from_results,
    rebuild_professor_summary,
    summarize_corpus_partition,
)
from bit_professor_chat.model_factory import build_ocr_model


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
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"))
    project_root = settings.project_root

    session = build_requests_session("agents-tutorial-lab2-prep/0.1")
    listing_pages = discover_listing_pages(LISTING_URL, session)
    listings = collect_professor_links(listing_pages, session)
    if args.limit is not None:
        listings = listings[: args.limit]

    partition = partition_professors_for_corpus(listings, project_root=project_root)
    ocr_model = build_ocr_model(settings)

    results = [
        build_cached_markdown_result(
            listing=listing,
            cache_entry=partition.cache_index[listing.detail_url],
            project_root=project_root,
            note="prepared from existing markdown",
        )
        for listing in partition.ready
    ]

    for listing in partition.needs_rebuild:
        results.append(
            ingest_professor(
                listing=listing,
                settings=settings,
                ocr_model=ocr_model,
                session=build_requests_session("agents-tutorial-lab2-prep-worker/0.1"),
                project_root=project_root,
                artifact_namespace=args.artifact_namespace,
                build_graph=False,
            )
        )

    successes = [result for result in results if result.status == "ok"]
    failures = [result for result in results if result.status != "ok"]
    rebuild_professor_summary(project_root, successes)
    corpus_result = rebuild_markdown_corpus_from_results(
        project_root=project_root,
        settings=settings,
        results=successes,
    )

    summary = {
        **summarize_corpus_partition(partition),
        "listing_page_count": len(listing_pages),
        "prepared_from_markdown": len(partition.ready),
        "refreshed_now": len(partition.needs_rebuild) - len(failures),
        "failed_refreshes": len(failures),
        "failed_professors": [result.name for result in failures[:10]],
        "corpus": corpus_result.to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
