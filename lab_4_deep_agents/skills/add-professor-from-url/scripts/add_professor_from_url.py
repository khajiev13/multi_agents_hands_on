from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DETAIL_URL_PATTERN = re.compile(
    r"^https://isc\.bit\.edu\.cn/schools/csat/knowingprofessors5/b\d+\.htm$",
    re.IGNORECASE,
)


def discover_project_root(start: Path | None = None) -> Path:
    candidates = [start.resolve()] if start else []
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])

    seen: set[Path] = set()
    for candidate in candidates:
        for current in (candidate, *candidate.parents):
            if current in seen:
                continue
            seen.add(current)
            if (current / "pyproject.toml").exists():
                return current
    raise FileNotFoundError("Could not discover the project root from the current workspace")


def normalize_detail_url(detail_url: str) -> str:
    split = urlsplit(detail_url.strip())
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))


def resolve_professors_dir(project_root: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def resolve_incoming_dir(professors_dir: Path) -> Path:
    return professors_dir.parent / "incoming"


def serialize_artifact_path(path: Path, *, project_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    if resolved_path.is_relative_to(resolved_root):
        return str(resolved_path.relative_to(resolved_root))
    return str(resolved_path)


def display_name_from_slug(slug: str, fallback: str) -> str:
    parts = [part.capitalize() for part in slug.split("-") if part]
    return " ".join(parts) or fallback.strip()


def emit(payload: dict[str, Any], *, exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-url", required=True)
    parser.add_argument("--professors-dir", default="professors")
    parser.add_argument("--project-root")
    args = parser.parse_args()

    detail_url = normalize_detail_url(args.detail_url)
    if not DETAIL_URL_PATTERN.fullmatch(detail_url):
        emit(
            {
                "status": "failed",
                "professor_name": None,
                "slug": None,
                "markdown_path": None,
                "page_count": 0,
                "error": "Only official BIT CSAT detail URLs are supported.",
            },
            exit_code=1,
        )

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else discover_project_root(Path.cwd())
    )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from bit_professor_chat.config import TutorSettings
    from bit_professor_chat.ingestion_models import ProfessorListing
    from bit_professor_chat.legacy_cache import read_professor_markdown_metadata
    from bit_professor_chat.markdown_corpus import slugify_name, validate_professor_dossier
    from bit_professor_chat.markdown_render import render_professor_markdown
    from bit_professor_chat.model_factory import build_ocr_model
    from bit_professor_chat.ocr_transcript import extract_professor_poster_markdown
    from bit_professor_chat.source_discovery import (
        extract_image_urls,
        build_requests_session,
        find_professor_listing_by_detail_url,
    )
    from bit_professor_chat.structured_profiles import (
        StructuredProfessorProfileArtifact,
        StructuredProfileMetadata,
        extract_structured_professor_profile,
        structured_profile_to_dossier,
    )

    settings = TutorSettings.from_env(project_root / ".env")
    professors_dir = resolve_professors_dir(project_root, args.professors_dir)
    professors_dir.mkdir(parents=True, exist_ok=True)
    incoming_dir = resolve_incoming_dir(professors_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)

    session = build_requests_session("agents-tutorial-lab4-add/0.1")
    listing = find_professor_listing_by_detail_url(detail_url, session=session)
    if listing is None:
        emit(
            {
                "status": "failed",
                "professor_name": None,
                "slug": None,
                "markdown_path": None,
                "page_count": 0,
                "error": "Could not resolve this detail URL from the official BIT CSAT listing pages.",
            },
            exit_code=1,
        )

    slug = slugify_name(listing.name)
    display_name = display_name_from_slug(slug, listing.name)
    normalized_listing = ProfessorListing(name=display_name, detail_url=listing.detail_url)
    target_path = professors_dir / f"{slug}.md"
    if target_path.exists():
        metadata = read_professor_markdown_metadata(target_path)
        emit(
            {
                "status": "duplicate_found",
                "professor_name": metadata.get("markdown_title") or listing.name,
                "slug": slug,
                "markdown_path": f"/professors/{target_path.name}",
                "page_count": metadata.get("page_count") or 0,
            }
        )

    for existing_path in sorted(professors_dir.glob("*.md")):
        metadata = read_professor_markdown_metadata(existing_path)
        if (metadata.get("detail_url") or "").strip() != detail_url:
            continue
        emit(
            {
                "status": "duplicate_found",
                "professor_name": metadata.get("markdown_title") or listing.name,
                "slug": existing_path.stem,
                "markdown_path": f"/professors/{existing_path.name}",
                "page_count": metadata.get("page_count") or 0,
            }
        )

    image_urls: list[str] = []
    try:
        image_urls = list(extract_image_urls(detail_url, session))
        if not image_urls:
            raise ValueError("No poster image URLs found on the detail page")

        ocr_model = build_ocr_model(settings)
        ocr_markdown = extract_professor_poster_markdown(
            listing=normalized_listing,
            image_urls=image_urls,
            model=ocr_model,
            session=session,
        )
        (incoming_dir / f"{slug}-ocr.md").write_text(
            ocr_markdown.strip() + "\n",
            encoding="utf-8",
        )
        profile = extract_structured_professor_profile(
            listing=normalized_listing,
            ocr_markdown=ocr_markdown,
            settings=settings,
        )
        profile_artifact = StructuredProfessorProfileArtifact(
            metadata=StructuredProfileMetadata(
                name=normalized_listing.name,
                slug=slug,
                detail_url=normalized_listing.detail_url,
                source_ocr_markdown_path=serialize_artifact_path(
                    incoming_dir / f"{slug}-ocr.md",
                    project_root=project_root,
                ),
                source_page_urls=list(image_urls),
            ),
            profile=profile,
        )
        (incoming_dir / f"{slug}-profile.json").write_text(
            json.dumps(
                profile_artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        dossier = structured_profile_to_dossier(
            profile=profile,
            listing=normalized_listing,
            source_page_urls=image_urls,
        )
        canonical_markdown = render_professor_markdown(dossier).strip()
        validation = validate_professor_dossier(
            dossier=dossier,
            listing=normalized_listing,
            rendered_markdown=canonical_markdown,
        )
        if validation.status != "valid":
            raise ValueError(
                "Canonical dossier validation failed: "
                + "; ".join([*validation.notes, *validation.quality_notes])
            )
    except Exception as exc:
        emit(
            {
                "status": "failed",
                "professor_name": display_name,
                "slug": slug,
                "markdown_path": None,
                "page_count": len(image_urls),
                "error": str(exc),
            },
            exit_code=1,
        )

    target_path.write_text(canonical_markdown.strip() + "\n", encoding="utf-8")
    emit(
        {
            "status": "added",
            "professor_name": dossier.title,
            "slug": slug,
            "markdown_path": f"/professors/{target_path.name}",
            "page_count": len(image_urls),
        }
    )


if __name__ == "__main__":
    main()
