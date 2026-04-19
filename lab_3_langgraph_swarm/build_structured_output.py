from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.config import TutorSettings
from bit_professor_chat.ingestion_models import ProfessorListing, StructuredOutputBuildResult
from bit_professor_chat.markdown_corpus import slugify_name
from bit_professor_chat.source_discovery import (
    LISTING_URL,
    build_requests_session,
    collect_professor_links,
    discover_listing_pages,
)
from bit_professor_chat.structured_profiles import (
    StructuredProfessorProfileArtifact,
    StructuredProfileMetadata,
    extract_structured_professor_profile,
)


OCR_FILE_SUFFIX = "-ocr.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instructor-only structured extraction from reviewed OCR markdown into "
            "committed Lab 3 typed profile JSON files."
        )
    )
    parser.add_argument(
        "--artifact-namespace",
        type=str,
        default="lab1-pre-insertion-review",
        help="Artifact namespace containing reviewed professor OCR markdown files.",
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
        help="Optional comma-separated professor slugs for targeted extraction runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional override for the committed structured profile output directory.",
    )
    return parser.parse_args()


def parse_only_slugs(value: str) -> list[str]:
    return [slug.strip() for slug in value.split(",") if slug.strip()]


def resolve_output_dir(project_root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return project_root / "lab_3_langgraph_swarm" / "structured_output"
    return output_dir if output_dir.is_absolute() else project_root / output_dir


def load_live_listing_index() -> dict[str, ProfessorListing]:
    session = build_requests_session("agents-tutorial-structured-output/0.1")
    try:
        listing_pages = discover_listing_pages(LISTING_URL, session)
        listings = collect_professor_links(listing_pages, session)
    finally:
        session.close()
    return {slugify_name(listing.name): listing for listing in listings}


def load_existing_structured_metadata(output_dir: Path) -> dict[str, ProfessorListing]:
    index: dict[str, ProfessorListing] = {}
    if not output_dir.exists():
        return index
    for path in sorted(output_dir.glob("*-profile.json")):
        try:
            artifact = StructuredProfessorProfileArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        index[artifact.metadata.slug] = ProfessorListing(
            name=artifact.metadata.name,
            detail_url=artifact.metadata.detail_url,
        )
    return index


def resolve_listing_for_slug(
    *,
    slug: str,
    existing_index: dict[str, ProfessorListing],
    live_index: dict[str, ProfessorListing],
) -> ProfessorListing:
    listing = existing_index.get(slug) or live_index.get(slug)
    if listing is None:
        raise ValueError(f"Could not resolve listing metadata for slug: {slug}")
    return listing


def select_ocr_paths(
    *,
    input_dir: Path,
    only_slugs: list[str] | None,
    limit: int | None,
) -> list[Path]:
    selected_slugs = {slug.strip() for slug in (only_slugs or []) if slug.strip()}
    paths = sorted(input_dir.glob(f"*{OCR_FILE_SUFFIX}"))
    if selected_slugs:
        paths = [
            path
            for path in paths
            if path.name.removesuffix(OCR_FILE_SUFFIX) in selected_slugs
        ]
    if limit is not None:
        paths = paths[:limit]
    return paths


def main() -> None:
    args = parse_args()
    settings = TutorSettings.from_env(Path(".env"))
    input_dir = settings.project_root / "artifacts" / args.artifact_namespace
    if not input_dir.exists():
        raise FileNotFoundError(f"OCR artifact namespace does not exist: {input_dir}")

    output_dir = resolve_output_dir(
        settings.project_root,
        Path(args.output_dir) if args.output_dir else None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_paths = select_ocr_paths(
        input_dir=input_dir,
        only_slugs=parse_only_slugs(args.only_slugs),
        limit=args.limit,
    )
    if not ocr_paths:
        raise FileNotFoundError(f"No OCR markdown files were found under {input_dir}.")

    existing_index = load_existing_structured_metadata(output_dir)
    try:
        live_index = load_live_listing_index()
    except Exception:
        live_index = {}
    failures: list[dict[str, str]] = []
    built_professors: list[str] = []
    skipped_count = 0

    for path in ocr_paths:
        slug = path.name.removesuffix(OCR_FILE_SUFFIX)
        try:
            ocr_markdown = path.read_text(encoding="utf-8")
            if not ocr_markdown.strip():
                skipped_count += 1
                failures.append({"slug": slug, "error": "OCR markdown is empty"})
                continue

            listing = resolve_listing_for_slug(
                slug=slug,
                existing_index=existing_index,
                live_index=live_index,
            )
            profile = extract_structured_professor_profile(
                listing=listing,
                ocr_markdown=ocr_markdown,
                settings=settings,
            )
            artifact = StructuredProfessorProfileArtifact(
                metadata=StructuredProfileMetadata(
                    name=listing.name,
                    slug=slug,
                    detail_url=listing.detail_url,
                    source_ocr_markdown_path=str(path.relative_to(settings.project_root)),
                ),
                profile=profile,
            )
            output_path = output_dir / f"{slug}-profile.json"
            output_path.write_text(
                json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            built_professors.append(slug)
        except Exception as exc:
            failures.append({"slug": slug, "error": str(exc)})

    summary = StructuredOutputBuildResult(
        artifact_namespace=args.artifact_namespace,
        input_dir=str(input_dir.relative_to(settings.project_root)),
        output_dir=str(output_dir.relative_to(settings.project_root)),
        professor_count=len(ocr_paths),
        success_count=len(built_professors),
        failure_count=len(failures),
        skipped_count=skipped_count,
        failures=failures,
        built_professors=sorted(built_professors),
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
