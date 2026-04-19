from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import requests
from langchain_openai import ChatOpenAI

from .config import TutorSettings
from .graph_ingestion import reset_neo4j_database, verify_neo4j_graph
from .ingestion_models import (
    ProfessorArtifactCacheRecord,
    ProfessorCorpusPartition,
    ProfessorGraphIngestionResult,
    ProfessorIngestionPartition,
    ProfessorIngestionResult,
    ProfessorListing,
    ProfessorOCRBundleResult,
    ProfessorPageNotesResult,
    ProfessorPreInsertionBatchResult,
    ProfessorPreInsertionResult,
)
from .legacy_cache import (
    build_cached_markdown_result,
    build_cached_summary_line,
    build_professor_cache_index,
    load_neo4j_detail_url_index,
    partition_professors_for_corpus,
    partition_professors_for_ingestion,
    rebuild_professor_summary,
    restore_professor_from_cache,
    summarize_corpus_partition,
)
from .markdown_corpus import (
    CorpusBuildResult,
    slugify_name,
)
from .model_factory import build_ocr_model
from .ocr_transcript import extract_professor_poster_page_markdowns
from .source_discovery import (
    LISTING_URL,
    build_requests_session,
    collect_professor_links,
    collect_professor_links_from_page,
    discover_listing_pages,
    extract_image_urls,
    fetch_soup,
)


GRAPH_INPUT_SOURCE_SECTION_PATTERN = re.compile(r"\n## Source Pages(?:\n.*)*$", re.S)
EXPECTED_COURSE_PROFESSOR_SLUGS = (
    "che-haiying",
    "cheng-cheng",
    "filippo-fabrocini",
    "gao-guangyu",
    "gao-yang",
    "huang-heyan",
    "huang-yonggang",
    "jin-fusheng",
    "li-dongni",
    "li-fan",
    "li-jianwu",
    "li-kan",
    "li-xin",
    "li-yugang",
    "liang-wei",
    "liu-hui",
    "lu-yao",
    "lv-kun",
    "mao-xianling",
    "niu-zhendong",
    "shang-jianyun",
    "shen-meng",
    "shi-feng",
    "shi-shumin",
    "song-hong",
    "song-tian",
    "tang-haijing",
    "tian-donghai",
    "wang-juan",
    "wang-quanyu",
    "wang-shuliang",
    "wang-yizhuo",
    "wei-jin",
    "yan-bo",
    "yang-song",
    "yu-yue",
    "yuan-hanning",
    "zhang-huaping",
    "zhang-wenyao",
    "zhao-fengnian",
    "zhao-qingjie",
    "zheng-hong",
    "zheng-jun",
    "zhu-liehuang",
)


RAW_OCR_MARKDOWN_MODE_MESSAGE = (
    "Raw OCR markdown mode is enabled. Downstream dossier synthesis, professors/*.md generation, "
    "corpus rebuilds, and graph generation are intentionally paused until a follow-up "
    "compatibility pass. Use prepare_professor_ocr_bundle(...) to produce "
    "artifacts/<namespace>/<slug>-ocr.md."
)


class RawOCRMarkdownModeError(RuntimeError):
    pass


def _raise_raw_ocr_markdown_mode_error(entrypoint: str) -> None:
    raise RawOCRMarkdownModeError(f"{entrypoint}: {RAW_OCR_MARKDOWN_MODE_MESSAGE}")


def _fetch_live_professor_listings(user_agent: str) -> tuple[list[str], list[ProfessorListing]]:
    listing_session = build_requests_session(user_agent)
    try:
        listing_pages = discover_listing_pages(LISTING_URL, listing_session)
        listings = collect_professor_links(listing_pages, listing_session)
    finally:
        listing_session.close()
    return listing_pages, listings


def _build_refresh_preflight(listings: Sequence[ProfessorListing]) -> dict[str, Any]:
    live_slugs = [slugify_name(item.name) for item in listings]
    live_slug_set = set(live_slugs)
    expected_slug_set = set(EXPECTED_COURSE_PROFESSOR_SLUGS)
    missing_live_slugs = sorted(expected_slug_set - live_slug_set)
    unexpected_live_slugs = sorted(live_slug_set - expected_slug_set)
    status = (
        "ok"
        if len(listings) == len(EXPECTED_COURSE_PROFESSOR_SLUGS)
        and not missing_live_slugs
        and not unexpected_live_slugs
        else "mismatch"
    )
    return {
        "status": status,
        "expected_count": len(EXPECTED_COURSE_PROFESSOR_SLUGS),
        "live_count": len(listings),
        "missing_live_slugs": missing_live_slugs,
        "unexpected_live_slugs": unexpected_live_slugs,
        "sample_live_slugs": live_slugs[:10],
    }


def _select_refresh_listings(
    listings: Sequence[ProfessorListing],
    *,
    only_slugs: Sequence[str] | None,
    limit: int | None,
) -> list[ProfessorListing]:
    selected = list(listings)
    if only_slugs:
        requested_slugs = list(dict.fromkeys(slug.strip() for slug in only_slugs if slug.strip()))
        live_by_slug = {slugify_name(item.name): item for item in listings}
        unknown_slugs = [slug for slug in requested_slugs if slug not in live_by_slug]
        if unknown_slugs:
            raise ValueError(f"Unknown professor slugs requested: {', '.join(unknown_slugs)}")
        selected = [live_by_slug[slug] for slug in requested_slugs]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _build_refresh_report_rows(
    results: Sequence[ProfessorIngestionResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "name": result.name,
                "slug": result.slug,
                "detail_url": result.detail_url,
                "status": result.status,
                "error": result.error,
                "validation_status": result.validation_status,
                "validation_notes": list(result.validation_notes),
                "validation_checks": dict(result.validation_checks),
                "markdown_path": result.markdown_path,
                "page_notes_path": result.page_notes_path,
                "graph_json_path": result.graph_json_path,
                "graph_html_path": result.graph_html_path,
                "page_count": result.page_count,
                "entity_count": result.entity_count,
                "relation_count": result.relation_count,
                "summary_line": result.summary_line,
            }
        )
    return rows
def prepare_professor_ocr_bundle(
    *,
    listing: ProfessorListing,
    ocr_model: ChatOpenAI,
    session: requests.Session,
    project_root: Path,
    artifact_namespace: str,
) -> ProfessorOCRBundleResult:
    artifact_dir = project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_name(listing.name)
    ocr_markdown_path = artifact_dir / f"{slug}-ocr.md"
    image_urls = extract_image_urls(listing.detail_url, session)
    if not image_urls:
        raise ValueError(f"No image URLs found for {listing.name}")

    page_markdowns = extract_professor_poster_page_markdowns(
        listing=listing,
        model=ocr_model,
        image_urls=image_urls,
        session=session,
    )
    ocr_markdown_text = "\n\n".join(page_markdowns)
    if not ocr_markdown_text.strip():
        raise ValueError(f"OCR markdown artifact is empty for {listing.name}")
    ocr_markdown_path.write_text(ocr_markdown_text, encoding="utf-8")

    return ProfessorOCRBundleResult(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=slug,
        page_count=len(image_urls),
        image_urls=list(image_urls),
        page_markdowns=list(page_markdowns),
        ocr_markdown_path=str(ocr_markdown_path.relative_to(project_root)),
        ocr_markdown_text=ocr_markdown_text,
        page_notes_path=str(ocr_markdown_path.relative_to(project_root)),
    )


def prepare_professor_page_notes(
    *,
    listing: ProfessorListing,
    ocr_model: ChatOpenAI,
    session: requests.Session,
    project_root: Path,
    artifact_namespace: str,
) -> ProfessorPageNotesResult:
    return prepare_professor_ocr_bundle(
        listing=listing,
        ocr_model=ocr_model,
        session=session,
        project_root=project_root,
        artifact_namespace=artifact_namespace,
    )


def _artifact_relative_path(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _prune_non_ocr_review_artifacts(artifact_dir: Path) -> None:
    for artifact_path in artifact_dir.iterdir():
        if artifact_path.is_file() and not artifact_path.name.endswith("-ocr.md"):
            artifact_path.unlink()


def _build_professor_preinsertion_result(
    *,
    listing: ProfessorListing,
    slug: str,
    image_urls: Sequence[str],
    stage_statuses: dict[str, bool],
    artifact_paths: dict[str, str],
    error: str | None = None,
    failure_stage: str | None = None,
) -> ProfessorPreInsertionResult:
    return ProfessorPreInsertionResult(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=slug,
        page_count=len(image_urls),
        status="ok" if error is None else "error",
        failure_stage=failure_stage,
        error=error,
        stage_statuses=dict(stage_statuses),
        artifact_paths=dict(artifact_paths),
    )


def _prepare_professor_preinsertion_bundle(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    project_root: Path,
    artifact_namespace: str,
    user_agent: str,
) -> ProfessorPreInsertionResult:
    artifact_dir = project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_name(listing.name)
    ocr_markdown_path = artifact_dir / f"{slug}-ocr.md"

    stage_statuses = {
        "crawl": False,
        "ocr": False,
    }
    artifact_paths: dict[str, str] = {}
    current_stage = "crawl"
    image_urls: list[str] = []

    session = build_requests_session(user_agent)
    try:
        image_urls = extract_image_urls(listing.detail_url, session)
        if not image_urls:
            raise ValueError(f"No image URLs found for {listing.name}")
        stage_statuses["crawl"] = True

        current_stage = "ocr"
        ocr_model = build_ocr_model(settings)
        page_markdowns = extract_professor_poster_page_markdowns(
            listing=listing,
            image_urls=image_urls,
            model=ocr_model,
            session=session,
        )
        ocr_markdown_text = "\n\n".join(page_markdowns).strip()
        if not ocr_markdown_text:
            raise ValueError(f"OCR markdown artifact is empty for {listing.name}")
        ocr_markdown_path.write_text(ocr_markdown_text, encoding="utf-8")
        artifact_paths["ocr_markdown_path"] = _artifact_relative_path(project_root, ocr_markdown_path)
        stage_statuses["ocr"] = True

        return _build_professor_preinsertion_result(
            listing=listing,
            slug=slug,
            image_urls=image_urls,
            stage_statuses=stage_statuses,
            artifact_paths=artifact_paths,
        )
    except Exception as exc:
        return _build_professor_preinsertion_result(
            listing=listing,
            slug=slug,
            image_urls=image_urls,
            stage_statuses=stage_statuses,
            artifact_paths=artifact_paths,
            error=str(exc),
            failure_stage=current_stage,
        )
    finally:
        session.close()


def prepare_professor_preinsertion_artifacts(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    project_root: Path,
    artifact_namespace: str,
    user_agent: str = "agents-tutorial-preinsert-worker/0.1",
) -> ProfessorPreInsertionResult:
    return _prepare_professor_preinsertion_bundle(
        listing=listing,
        settings=settings,
        project_root=project_root,
        artifact_namespace=artifact_namespace,
        user_agent=user_agent,
    )


def _build_stage_counts(results: Sequence[ProfessorPreInsertionResult]) -> dict[str, int]:
    stages = ("crawl", "ocr")
    return {
        stage: sum(1 for result in results if result.stage_statuses.get(stage, False))
        for stage in stages
    }


def refresh_professors_to_artifacts(
    *,
    settings: TutorSettings,
    project_root: Path | None = None,
    limit: int | None = None,
    only_slugs: Sequence[str] | None = None,
    max_concurrency: int = 8,
    artifact_namespace: str = "lab1-pre-insertion-review",
    user_agent_prefix: str = "agents-tutorial-preinsert",
) -> ProfessorPreInsertionBatchResult:
    resolved_project_root = project_root or settings.project_root
    listing_pages, listings = _fetch_live_professor_listings(f"{user_agent_prefix}/listing")
    selected_listings = _select_refresh_listings(
        listings,
        only_slugs=only_slugs,
        limit=limit,
    )
    artifact_dir = resolved_project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _prune_non_ocr_review_artifacts(artifact_dir)

    results: list[ProfessorPreInsertionResult] = []

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_listing = {
            executor.submit(
                _prepare_professor_preinsertion_bundle,
                listing=listing,
                settings=settings,
                project_root=resolved_project_root,
                artifact_namespace=artifact_namespace,
                user_agent=f"{user_agent_prefix}/{index + 1}",
            ): listing
            for index, listing in enumerate(selected_listings)
        }
        for future in as_completed(future_to_listing):
            result = future.result()
            results.append(result)

    results.sort(key=lambda result: result.slug)

    failures = [
        {
            "name": result.name,
            "slug": result.slug,
            "detail_url": result.detail_url,
            "failure_stage": result.failure_stage,
            "error": result.error,
        }
        for result in results
        if result.status != "ok"
    ]

    batch_result = ProfessorPreInsertionBatchResult(
        artifact_namespace=artifact_namespace,
        artifact_dir=str(artifact_dir.relative_to(resolved_project_root)),
        listing_pages=list(listing_pages),
        professor_count=len(selected_listings),
        success_count=sum(1 for result in results if result.status == "ok"),
        failure_count=sum(1 for result in results if result.status != "ok"),
        stage_counts=_build_stage_counts(results),
        failures=failures,
        results=results,
    )
    return batch_result
def _strip_graph_input_metadata(markdown_text: str) -> str:
    filtered_lines: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("- detail_url:"):
            continue
        if line.startswith("- page_count:"):
            continue
        filtered_lines.append(line)

    return GRAPH_INPUT_SOURCE_SECTION_PATTERN.sub("", "\n".join(filtered_lines)).strip()


def build_graph_input_text_from_page_notes(
    *,
    page_notes_result: ProfessorPageNotesResult,
    settings: TutorSettings | None = None,
) -> str:
    _raise_raw_ocr_markdown_mode_error("build_graph_input_text_from_page_notes")


def build_graph_input_text_from_ocr_bundle(
    *,
    ocr_bundle_result: ProfessorOCRBundleResult,
    settings: TutorSettings | None = None,
) -> str:
    _raise_raw_ocr_markdown_mode_error("build_graph_input_text_from_ocr_bundle")


def prepare_professor_markdown(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    ocr_model: ChatOpenAI,
    session: requests.Session,
    project_root: Path,
    artifact_namespace: str,
) -> ProfessorIngestionResult:
    _raise_raw_ocr_markdown_mode_error("prepare_professor_markdown")


def rebuild_markdown_corpus_from_results(
    *,
    project_root: Path,
    settings: TutorSettings,
    results: Iterable[ProfessorIngestionResult],
) -> CorpusBuildResult:
    _raise_raw_ocr_markdown_mode_error("rebuild_markdown_corpus_from_results")


def _coerce_listing(value: ProfessorListing | dict[str, Any]) -> ProfessorListing:
    if isinstance(value, ProfessorListing):
        return value
    return ProfessorListing(**value)


def _build_professor_graph_chain(
    *,
    settings: TutorSettings,
    ocr_model: ChatOpenAI,
    project_root: Path,
    artifact_namespace: str,
    graph_name: str,
    user_agent: str,
):
    _raise_raw_ocr_markdown_mode_error("_build_professor_graph_chain")


def ingest_professor_to_graph(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    ocr_model: ChatOpenAI,
    project_root: Path,
    artifact_namespace: str,
    graph_name: str = "",
    user_agent: str = "agents-tutorial-lab1-worker/0.1",
    max_attempts: int = 2,
) -> ProfessorGraphIngestionResult:
    _raise_raw_ocr_markdown_mode_error("ingest_professor_to_graph")


def ingest_professors_to_graph(
    *,
    listings: Sequence[ProfessorListing],
    settings: TutorSettings,
    ocr_model: ChatOpenAI,
    project_root: Path,
    artifact_namespace: str,
    graph_name: str = "",
    max_concurrency: int = 4,
    user_agent_prefix: str = "agents-tutorial-lab1-worker",
) -> list[ProfessorGraphIngestionResult]:
    _raise_raw_ocr_markdown_mode_error("ingest_professors_to_graph")


def ingest_professor(
    listing: ProfessorListing,
    *,
    settings: TutorSettings | None = None,
    ocr_model: ChatOpenAI | None = None,
    session: requests.Session | None = None,
    project_root: Path | None = None,
    artifact_namespace: str = "lab1-ingestion",
    graph_name: str = "",
    build_graph: bool = True,
) -> ProfessorIngestionResult:
    _raise_raw_ocr_markdown_mode_error("ingest_professor")


def refresh_professors_to_graph(
    *,
    settings: TutorSettings,
    project_root: Path | None = None,
    limit: int | None = None,
    only_slugs: Sequence[str] | None = None,
    max_concurrency: int = 2,
    reset_database: bool = True,
    wipe_first: bool = False,
    artifact_namespace: str = "lab1-full-refresh",
    graph_name: str = "BIT_CSAT_FULL",
    user_agent_prefix: str = "agents-tutorial-lab1-refresh",
) -> dict[str, Any]:
    _raise_raw_ocr_markdown_mode_error("refresh_professors_to_graph")


__all__ = [
    "LISTING_URL",
    "ProfessorArtifactCacheRecord",
    "ProfessorCorpusPartition",
    "ProfessorGraphIngestionResult",
    "ProfessorIngestionPartition",
    "ProfessorIngestionResult",
    "ProfessorListing",
    "ProfessorOCRBundleResult",
    "ProfessorPageNotesResult",
    "RAW_OCR_MARKDOWN_MODE_MESSAGE",
    "RawOCRMarkdownModeError",
    "build_cached_markdown_result",
    "build_cached_summary_line",
    "build_graph_input_text_from_ocr_bundle",
    "build_graph_input_text_from_page_notes",
    "build_ocr_model",
    "build_professor_cache_index",
    "build_requests_session",
    "collect_professor_links",
    "collect_professor_links_from_page",
    "discover_listing_pages",
    "extract_image_urls",
    "fetch_soup",
    "ingest_professor",
    "ingest_professor_to_graph",
    "ingest_professors_to_graph",
    "refresh_professors_to_graph",
    "refresh_professors_to_artifacts",
    "load_neo4j_detail_url_index",
    "partition_professors_for_corpus",
    "partition_professors_for_ingestion",
    "prepare_professor_markdown",
    "prepare_professor_ocr_bundle",
    "prepare_professor_page_notes",
    "prepare_professor_preinsertion_artifacts",
    "rebuild_markdown_corpus_from_results",
    "rebuild_professor_summary",
    "reset_neo4j_database",
    "restore_professor_from_cache",
    "summarize_corpus_partition",
    "verify_neo4j_graph",
]
