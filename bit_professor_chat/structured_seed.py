from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .ingestion_models import (
    ProfessorArtifactMetadata,
    ProfessorListing,
    StructuredProfessorReview,
    StructuredSeedExportResult,
)
from .structured_review import finalize_structured_review


STRUCTURED_SEED_DIR = Path("lab_3_langgraph_swarm") / "structured_seed"


def resolve_structured_seed_dir(project_root: Path, seed_dir: Path | None = None) -> Path:
    if seed_dir is None:
        return project_root / STRUCTURED_SEED_DIR
    if seed_dir.is_absolute():
        return seed_dir
    return project_root / seed_dir


def _select_review_paths(
    *,
    review_dir: Path,
    only_slugs: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    selected_slugs = {slug.strip() for slug in (only_slugs or []) if slug.strip()}
    paths = sorted(review_dir.glob("*-structured.json"))
    if selected_slugs:
        paths = [path for path in paths if path.name.removesuffix("-structured.json") in selected_slugs]
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_structured_review_file(path: Path) -> StructuredProfessorReview:
    return StructuredProfessorReview.model_validate_json(path.read_text(encoding="utf-8"))


def load_structured_seed_reviews(
    *,
    project_root: Path,
    seed_dir: Path | None = None,
    only_slugs: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[tuple[Path, StructuredProfessorReview]]:
    resolved_seed_dir = resolve_structured_seed_dir(project_root, seed_dir)
    if not resolved_seed_dir.exists():
        raise FileNotFoundError(f"Structured seed directory does not exist: {resolved_seed_dir}")
    review_paths = _select_review_paths(
        review_dir=resolved_seed_dir,
        only_slugs=only_slugs,
        limit=limit,
    )
    if not review_paths:
        raise FileNotFoundError(f"No structured seed files were found under {resolved_seed_dir}.")
    return [(path, load_structured_review_file(path)) for path in review_paths]


def promote_structured_review_artifacts(
    *,
    project_root: Path,
    artifact_namespace: str,
    seed_dir: Path | None = None,
    only_slugs: Iterable[str] | None = None,
    limit: int | None = None,
) -> StructuredSeedExportResult:
    artifact_dir = project_root / "artifacts" / artifact_namespace
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact namespace does not exist: {artifact_dir}")

    resolved_seed_dir = resolve_structured_seed_dir(project_root, seed_dir)
    resolved_seed_dir.mkdir(parents=True, exist_ok=True)

    review_paths = _select_review_paths(
        review_dir=artifact_dir,
        only_slugs=only_slugs,
        limit=limit,
    )
    if not review_paths:
        raise FileNotFoundError(f"No structured review artifacts were found under {artifact_dir}.")

    if limit is None and not any((only_slugs or [])):
        for stale_path in resolved_seed_dir.glob("*"):
            if stale_path.is_file():
                stale_path.unlink()

    exported_professors: list[str] = []
    failures: list[dict[str, str]] = []
    for review_path in review_paths:
        slug = review_path.name.removesuffix("-structured.json")
        try:
            review = load_structured_review_file(review_path)
            ocr_source_path = project_root / review.metadata.source_file
            if not ocr_source_path.exists():
                fallback_path = artifact_dir / f"{slug}-ocr.md"
                if fallback_path.exists():
                    ocr_source_path = fallback_path
                else:
                    raise FileNotFoundError(
                        f"OCR source markdown not found for {slug}: {review.metadata.source_file}"
                    )

            listing = ProfessorListing(
                name=review.metadata.name,
                detail_url=review.metadata.detail_url,
            )
            target_ocr_path = resolved_seed_dir / f"{slug}-ocr.md"
            target_review_path = resolved_seed_dir / f"{slug}-structured.json"
            target_ocr_path.write_text(ocr_source_path.read_text(encoding="utf-8"), encoding="utf-8")

            seed_metadata = ProfessorArtifactMetadata(
                name=review.metadata.name,
                detail_url=review.metadata.detail_url,
                slug=slug,
                page_count=review.metadata.page_count,
                image_urls=list(review.metadata.image_urls),
                artifact_namespace=resolved_seed_dir.name,
                source_file=str(target_ocr_path.relative_to(project_root)),
            )
            finalized = finalize_structured_review(
                review=review,
                listing=listing,
                metadata=seed_metadata,
            )
            target_review_path.write_text(
                json.dumps(finalized.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            exported_professors.append(slug)
        except Exception as exc:
            failures.append({"slug": slug, "error": str(exc)})

    result = StructuredSeedExportResult(
        artifact_namespace=artifact_namespace,
        seed_dir=str(resolved_seed_dir.relative_to(project_root)),
        professor_count=len(review_paths),
        success_count=len(exported_professors),
        failure_count=len(failures),
        failures=failures,
        exported_professors=sorted(exported_professors),
    )
    report_path = resolved_seed_dir / "seed-report.json"
    report_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


__all__ = [
    "STRUCTURED_SEED_DIR",
    "load_structured_review_file",
    "load_structured_seed_reviews",
    "promote_structured_review_artifacts",
    "resolve_structured_seed_dir",
]
