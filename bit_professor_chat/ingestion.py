from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import requests
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .config import TutorSettings
from .graph_ingestion import (
    aggregate_graphs,
    build_kg_generator,
    cluster_graph,
    generate_professor_graph,
    generate_professor_graph_with_generator,
    graph_to_payload,
    insert_professor_graph,
    require_kg_gen,
    reset_neo4j_database,
    save_graph_html,
    save_graph_json,
    verify_neo4j_graph,
)
from .ingestion_models import (
    ProfessorArtifactCacheRecord,
    ProfessorCorpusPartition,
    ProfessorDossier,
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
    build_graph_seed_index,
    build_cached_markdown_result,
    build_cached_summary_line,
    build_professor_cache_index,
    load_cached_graph,
    load_neo4j_detail_url_index,
    partition_professors_for_corpus,
    partition_professors_for_ingestion,
    read_professor_markdown_metadata,
    rebuild_professor_summary,
    resolve_graph_seed_dir,
    restore_professor_from_cache,
    summarize_corpus_partition,
    sync_graph_seed_directory,
)
from .markdown_corpus import (
    CorpusBuildResult,
    build_dossier_metadata,
    rebuild_markdown_corpus,
    slugify_name,
    validate_professor_dossier,
)
from .markdown_render import render_professor_markdown
from .model_factory import build_ocr_model
from .ocr_transcript import extract_professor_poster_page_markdowns, extract_professor_poster_markdown
from .source_discovery import (
    LISTING_URL,
    build_requests_session,
    collect_professor_links,
    collect_professor_links_from_page,
    discover_listing_pages,
    extract_image_urls,
    fetch_soup,
)
from .structured_review import build_professor_structured_review
from .synthesis import synthesize_professor_dossier


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


class ProfessorMarkdownValidationError(ValueError):
    def __init__(self, validation: Any):
        self.validation = validation
        super().__init__("; ".join(validation.notes) or "OCR validation failed")


RAW_OCR_MARKDOWN_MODE_MESSAGE = (
    "Raw OCR markdown mode is enabled. Downstream dossier synthesis, professors/*.md generation, "
    "corpus rebuilds, and graph generation are intentionally paused until a follow-up "
    "compatibility pass. Use prepare_professor_ocr_bundle(...) to produce "
    "artifacts/<namespace>/<slug>-ocr.md."
)

KG_GEN_PROFESSOR_CONTEXT = """Beijing Institute of Technology professor profile.

The OCR usually follows a professor dossier structure. Prefer extracting entities and relations for:
- professor identity and aliases
- affiliations and organizations
- research interests
- education experiences
- employment roles
- academic service roles
- awards
- publications
- contact information

When relation wording is supported by the text, prefer predicates such as:
affiliated_with, has_research_interest, studied_at, held_role_at, served_in_role_at,
received_award, authored, has_email, has_phone.
Keep relations faithful to the OCR text. Do not invent unsupported facts.
"""

KG_GEN_CLUSTER_CONTEXT = """Cluster this professor-graph corpus while preserving the professor-profile ontology.

Prefer canonical entities for:
- professor names and aliases
- organizations
- research topics
- awards
- publications

When predicates are semantically equivalent, prefer the canonical forms:
affiliated_with, has_research_interest, studied_at, held_role_at, served_in_role_at,
received_award, authored, has_email, has_phone.
Do not merge entities unless the OCR evidence strongly supports equivalence.
"""

KG_GEN_CLUSTER_ENTITY_LIMIT = 60
KG_GEN_CLUSTER_EDGE_LIMIT = 15


class RawOCRMarkdownModeError(RuntimeError):
    pass


def _raise_raw_ocr_markdown_mode_error(entrypoint: str) -> None:
    raise RawOCRMarkdownModeError(f"{entrypoint}: {RAW_OCR_MARKDOWN_MODE_MESSAGE}")


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_cluster_graph(
    *,
    kg: Any,
    graph: Any,
    context: str,
) -> tuple[Any, bool, str | None]:
    entity_count = len(getattr(graph, "entities", []) or [])
    edge_count = len(getattr(graph, "edges", []) or [])
    if entity_count > KG_GEN_CLUSTER_ENTITY_LIMIT or edge_count > KG_GEN_CLUSTER_EDGE_LIMIT:
        return (
            graph,
            False,
            (
                "skipped kg-gen clustering because graph exceeded the interactive prep "
                f"threshold ({entity_count} entities, {edge_count} edges)"
            ),
        )
    try:
        return cluster_graph(kg=kg, graph=graph, context=context), True, None
    except Exception as exc:
        return graph, False, f"kg-gen clustering failed: {exc}"


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


def _wipe_refresh_outputs(
    *,
    project_root: Path,
    artifact_namespace: str,
    settings: TutorSettings,
    reset_database: bool,
) -> dict[str, Any]:
    deleted_markdown_files = 0
    professor_dir = project_root / "professors"
    if professor_dir.exists():
        for markdown_path in professor_dir.glob("*.md"):
            markdown_path.unlink()
            deleted_markdown_files += 1

    professors_index_path = project_root / "professors.md"
    if professors_index_path.exists():
        professors_index_path.unlink()

    artifact_dir = project_root / "artifacts" / artifact_namespace
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    seed_dir = resolve_graph_seed_dir(project_root)
    deleted_seed_files = 0
    if seed_dir.exists():
        for graph_json_path in seed_dir.glob("*-graph.json"):
            graph_json_path.unlink()
            deleted_seed_files += 1

    if reset_database:
        reset_neo4j_database(settings)

    return {
        "deleted_markdown_files": deleted_markdown_files,
        "deleted_professors_index": not professors_index_path.exists(),
        "deleted_seed_files": deleted_seed_files,
        "artifact_dir": str(artifact_dir.relative_to(project_root)),
        "seed_dir": str(seed_dir.relative_to(project_root)),
        "neo4j_reset": reset_database,
    }


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


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _finalize_dossier(
    *,
    dossier: ProfessorDossier,
    listing: ProfessorListing,
    image_urls: Sequence[str],
    page_notes_result: ProfessorPageNotesResult,
) -> ProfessorDossier:
    basic_information = dossier.basic_information.model_copy(
        update={
            "heading": "Basic Information",
            "bullets": _dedupe_strings(dossier.basic_information.bullets),
        }
    )
    sections = [
        section.model_copy(
            update={
                "heading": section.heading.strip(),
                "heading_original": section.heading_original.strip()
                if section.heading_original
                else None,
                "bullets": _dedupe_strings(section.bullets),
            }
        )
        for section in dossier.sections
        if section.heading.strip() and _dedupe_strings(section.bullets)
    ]
    warnings = list(dossier.warnings)
    for page in getattr(page_notes_result, "ocr_pages", []):
        if not any(block.lines for block in page.blocks) and not page.uncertain_lines:
            warnings.append(f"page {page.page_number} OCR empty")

    return dossier.model_copy(
        update={
            "detail_url": listing.detail_url,
            "source_page_urls": list(image_urls),
            "basic_information": basic_information,
            "sections": sections,
            "uncertain_lines": _dedupe_strings(dossier.uncertain_lines),
            "warnings": _dedupe_strings(warnings),
        }
    )


def _build_synthesis_retry_hint(listing: ProfessorListing) -> str:
    return (
        "The previous synthesis was invalid. Rebuild the dossier using only OCR text. "
        f"Ensure the title matches the professor listing name '{listing.name}', keep "
        "Basic Information populated only with OCR-supported identity/contact lines, "
        "and keep at least one non-empty section when OCR blocks provide section text."
    )


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


def _build_graph_artifact_metadata(
    *,
    listing: ProfessorListing,
    slug: str,
    page_count: int,
    image_urls: Sequence[str],
    artifact_namespace: str,
    source_file: str,
) -> dict[str, Any]:
    return {
        "name": listing.name,
        "detail_url": listing.detail_url,
        "slug": slug,
        "page_count": page_count,
        "image_urls": list(image_urls),
        "artifact_namespace": artifact_namespace,
        "source_file": source_file,
    }


def _build_page_graphs_payload(
    *,
    listing: ProfessorListing,
    slug: str,
    image_urls: Sequence[str],
    page_markdowns: Sequence[str],
    page_graphs: Sequence[Any],
    artifact_namespace: str,
    source_file: str,
) -> dict[str, Any]:
    return {
        "metadata": _build_graph_artifact_metadata(
            listing=listing,
            slug=slug,
            page_count=len(image_urls),
            image_urls=image_urls,
            artifact_namespace=artifact_namespace,
            source_file=source_file,
        ),
        "pages": [
            {
                "page_number": page_number,
                "image_url": image_url,
                "ocr_markdown": page_markdown,
                "graph": graph_to_payload(page_graph),
            }
            for page_number, (image_url, page_markdown, page_graph) in enumerate(
                zip(image_urls, page_markdowns, page_graphs),
                start=1,
            )
        ],
    }


def _build_single_graph_payload(
    *,
    listing: ProfessorListing,
    slug: str,
    image_urls: Sequence[str],
    artifact_namespace: str,
    source_file: str,
    graph: Any,
    stage: str,
    page_graph_count: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _build_graph_artifact_metadata(
        listing=listing,
        slug=slug,
        page_count=len(image_urls),
        image_urls=image_urls,
        artifact_namespace=artifact_namespace,
        source_file=source_file,
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "metadata": metadata,
        "stage": stage,
        "page_graph_count": page_graph_count,
        "graph": graph_to_payload(graph),
    }


def _artifact_relative_path(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _build_professor_preinsertion_result(
    *,
    listing: ProfessorListing,
    slug: str,
    image_urls: Sequence[str],
    stage_statuses: dict[str, bool],
    artifact_paths: dict[str, str],
    aggregate_graph: Any | None = None,
    clustered_graph: Any | None = None,
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
        page_graph_count=len(image_urls) if stage_statuses.get("kg_generate") else 0,
        aggregate_entity_count=len(getattr(aggregate_graph, "entities", []) or []),
        aggregate_relation_count=len(getattr(aggregate_graph, "relations", []) or []),
        clustered_entity_count=len(getattr(clustered_graph, "entities", []) or []),
        clustered_relation_count=len(getattr(clustered_graph, "relations", []) or []),
    )


def _prepare_professor_preinsertion_bundle(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    project_root: Path,
    artifact_namespace: str,
    user_agent: str,
    kg_context: str,
    cluster_context: str,
) -> tuple[ProfessorPreInsertionResult, Any | None]:
    artifact_dir = project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_name(listing.name)
    ocr_markdown_path = artifact_dir / f"{slug}-ocr.md"
    page_graphs_path = artifact_dir / f"{slug}-page-graphs.json"
    professor_aggregate_path = artifact_dir / f"{slug}-professor-aggregate.json"
    professor_aggregate_html_path = artifact_dir / f"{slug}-professor-aggregate.html"
    professor_clustered_path = artifact_dir / f"{slug}-professor-clustered.json"
    professor_clustered_html_path = artifact_dir / f"{slug}-professor-clustered.html"
    structured_json_path = artifact_dir / f"{slug}-structured.json"

    stage_statuses = {
        "crawl": False,
        "ocr": False,
        "kg_generate": False,
        "aggregate": False,
        "cluster": False,
        "structured_json": False,
    }
    artifact_paths: dict[str, str] = {}
    current_stage = "crawl"
    image_urls: list[str] = []
    aggregate_graph: Any | None = None
    clustered_graph: Any | None = None

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

        ocr_bundle_result = ProfessorOCRBundleResult(
            name=listing.name,
            detail_url=listing.detail_url,
            slug=slug,
            page_count=len(image_urls),
            image_urls=list(image_urls),
            page_markdowns=list(page_markdowns),
            ocr_markdown_path=artifact_paths["ocr_markdown_path"],
            ocr_markdown_text=ocr_markdown_text,
            page_notes_path=artifact_paths["ocr_markdown_path"],
        )

        current_stage = "kg_generate"
        kg = build_kg_generator(settings)
        page_graphs = [
            generate_professor_graph_with_generator(
                markdown_text=page_markdown,
                kg=kg,
                context=kg_context,
            )
            for page_markdown in page_markdowns
        ]
        _write_json_file(
            page_graphs_path,
            _build_page_graphs_payload(
                listing=listing,
                slug=slug,
                image_urls=image_urls,
                page_markdowns=page_markdowns,
                page_graphs=page_graphs,
                artifact_namespace=artifact_namespace,
                source_file=artifact_paths["ocr_markdown_path"],
            ),
        )
        artifact_paths["page_graphs_path"] = _artifact_relative_path(project_root, page_graphs_path)
        stage_statuses["kg_generate"] = True

        current_stage = "aggregate"
        aggregate_graph = aggregate_graphs(kg=kg, graphs=page_graphs)
        _write_json_file(
            professor_aggregate_path,
            _build_single_graph_payload(
                listing=listing,
                slug=slug,
                image_urls=image_urls,
                artifact_namespace=artifact_namespace,
                source_file=artifact_paths["ocr_markdown_path"],
                graph=aggregate_graph,
                stage="professor_aggregate",
                page_graph_count=len(page_graphs),
            ),
        )
        artifact_paths["professor_aggregate_path"] = _artifact_relative_path(
            project_root, professor_aggregate_path
        )
        save_graph_html(aggregate_graph, professor_aggregate_html_path)
        artifact_paths["professor_aggregate_html_path"] = _artifact_relative_path(
            project_root, professor_aggregate_html_path
        )
        stage_statuses["aggregate"] = True

        current_stage = "cluster"
        clustered_graph, cluster_applied, cluster_reason = _maybe_cluster_graph(
            kg=kg,
            graph=aggregate_graph,
            context=cluster_context,
        )
        _write_json_file(
            professor_clustered_path,
            _build_single_graph_payload(
                listing=listing,
                slug=slug,
                image_urls=image_urls,
                artifact_namespace=artifact_namespace,
                source_file=artifact_paths["ocr_markdown_path"],
                graph=clustered_graph,
                stage="professor_clustered",
                page_graph_count=len(page_graphs),
                extra_metadata={
                    "cluster_applied": cluster_applied,
                    "cluster_reason": cluster_reason,
                },
            ),
        )
        artifact_paths["professor_clustered_path"] = _artifact_relative_path(
            project_root, professor_clustered_path
        )
        save_graph_html(clustered_graph, professor_clustered_html_path)
        artifact_paths["professor_clustered_html_path"] = _artifact_relative_path(
            project_root, professor_clustered_html_path
        )
        stage_statuses["cluster"] = True

        current_stage = "structured_json"
        structured_review = build_professor_structured_review(
            listing=listing,
            ocr_bundle_result=ocr_bundle_result,
            artifact_namespace=artifact_namespace,
            settings=settings,
        )
        _write_json_file(
            structured_json_path,
            structured_review.model_dump(mode="json"),
        )
        artifact_paths["structured_json_path"] = _artifact_relative_path(
            project_root, structured_json_path
        )
        stage_statuses["structured_json"] = True

        return (
            _build_professor_preinsertion_result(
                listing=listing,
                slug=slug,
                image_urls=image_urls,
                stage_statuses=stage_statuses,
                artifact_paths=artifact_paths,
                aggregate_graph=aggregate_graph,
                clustered_graph=clustered_graph,
            ),
            aggregate_graph,
        )
    except Exception as exc:
        return (
            _build_professor_preinsertion_result(
                listing=listing,
                slug=slug,
                image_urls=image_urls,
                stage_statuses=stage_statuses,
                artifact_paths=artifact_paths,
                aggregate_graph=aggregate_graph,
                clustered_graph=clustered_graph,
                error=str(exc),
                failure_stage=current_stage,
            ),
            None,
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
    kg_context: str = KG_GEN_PROFESSOR_CONTEXT,
    cluster_context: str = KG_GEN_CLUSTER_CONTEXT,
) -> ProfessorPreInsertionResult:
    result, _ = _prepare_professor_preinsertion_bundle(
        listing=listing,
        settings=settings,
        project_root=project_root,
        artifact_namespace=artifact_namespace,
        user_agent=user_agent,
        kg_context=kg_context,
        cluster_context=cluster_context,
    )
    return result


def _build_stage_counts(results: Sequence[ProfessorPreInsertionResult]) -> dict[str, int]:
    stages = ("crawl", "ocr", "kg_generate", "aggregate", "cluster", "structured_json")
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
    kg_context: str = KG_GEN_PROFESSOR_CONTEXT,
    cluster_context: str = KG_GEN_CLUSTER_CONTEXT,
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

    results: list[ProfessorPreInsertionResult] = []
    successful_professor_graphs: list[Any] = []

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_listing = {
            executor.submit(
                _prepare_professor_preinsertion_bundle,
                listing=listing,
                settings=settings,
                project_root=resolved_project_root,
                artifact_namespace=artifact_namespace,
                user_agent=f"{user_agent_prefix}/{index + 1}",
                kg_context=kg_context,
                cluster_context=cluster_context,
            ): listing
            for index, listing in enumerate(selected_listings)
        }
        for future in as_completed(future_to_listing):
            result, professor_graph = future.result()
            results.append(result)
            if result.status == "ok" and professor_graph is not None:
                successful_professor_graphs.append(professor_graph)

    results.sort(key=lambda result: result.slug)

    corpus_artifact_paths: dict[str, str] = {}
    if successful_professor_graphs:
        kg = build_kg_generator(settings)
        corpus_aggregate_graph = aggregate_graphs(kg=kg, graphs=successful_professor_graphs)
        corpus_aggregate_path = artifact_dir / "all-professors-aggregate.json"
        corpus_aggregate_html_path = artifact_dir / "all-professors-aggregate.html"
        _write_json_file(
            corpus_aggregate_path,
            {
                "metadata": {
                    "artifact_namespace": artifact_namespace,
                    "professor_count": len(successful_professor_graphs),
                    "listing_pages": list(listing_pages),
                    "source_professors": [result.name for result in results if result.status == "ok"],
                },
                "stage": "corpus_aggregate",
                "graph": graph_to_payload(corpus_aggregate_graph),
            },
        )
        corpus_artifact_paths["corpus_aggregate_path"] = _artifact_relative_path(
            resolved_project_root,
            corpus_aggregate_path,
        )
        save_graph_html(corpus_aggregate_graph, corpus_aggregate_html_path)
        corpus_artifact_paths["corpus_aggregate_html_path"] = _artifact_relative_path(
            resolved_project_root,
            corpus_aggregate_html_path,
        )

        corpus_clustered_graph, corpus_cluster_applied, corpus_cluster_reason = _maybe_cluster_graph(
            kg=kg,
            graph=corpus_aggregate_graph,
            context=cluster_context,
        )
        corpus_clustered_path = artifact_dir / "all-professors-clustered.json"
        corpus_clustered_html_path = artifact_dir / "all-professors-clustered.html"
        _write_json_file(
            corpus_clustered_path,
            {
                "metadata": {
                    "artifact_namespace": artifact_namespace,
                    "professor_count": len(successful_professor_graphs),
                    "listing_pages": list(listing_pages),
                    "source_professors": [result.name for result in results if result.status == "ok"],
                    "cluster_applied": corpus_cluster_applied,
                    "cluster_reason": corpus_cluster_reason,
                },
                "stage": "corpus_clustered",
                "graph": graph_to_payload(corpus_clustered_graph),
            },
        )
        corpus_artifact_paths["corpus_clustered_path"] = _artifact_relative_path(
            resolved_project_root,
            corpus_clustered_path,
        )
        save_graph_html(corpus_clustered_graph, corpus_clustered_html_path)
        corpus_artifact_paths["corpus_clustered_html_path"] = _artifact_relative_path(
            resolved_project_root,
            corpus_clustered_html_path,
        )

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
        corpus_artifact_paths=corpus_artifact_paths,
    )
    report_path = artifact_dir / "ingestion-report.json"
    _write_json_file(report_path, batch_result.to_dict())
    batch_result.corpus_artifact_paths["report_path"] = _artifact_relative_path(
        resolved_project_root,
        report_path,
    )
    _write_json_file(report_path, batch_result.to_dict())
    return batch_result


def _build_validated_dossier_and_markdown(
    *,
    page_notes_result: ProfessorPageNotesResult,
    settings: TutorSettings,
    retry_hint: str | None = None,
) -> tuple[ProfessorDossier, str, Any]:
    _raise_raw_ocr_markdown_mode_error("_build_validated_dossier_and_markdown")
    listing = ProfessorListing(
        name=page_notes_result.name,
        detail_url=page_notes_result.detail_url,
    )
    dossier = synthesize_professor_dossier(
        listing=listing,
        pages=page_notes_result.ocr_pages,
        settings=settings,
        retry_hint=retry_hint,
    )
    finalized_dossier = _finalize_dossier(
        dossier=dossier,
        listing=listing,
        image_urls=page_notes_result.image_urls,
        page_notes_result=page_notes_result,
    )
    canonical_markdown = render_professor_markdown(finalized_dossier).strip()
    validation = validate_professor_dossier(
        dossier=finalized_dossier,
        listing=listing,
        rendered_markdown=canonical_markdown,
    )
    if validation.status != "valid":
        raise ProfessorMarkdownValidationError(validation)
    return finalized_dossier, canonical_markdown, validation


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


def restore_seeded_professors_to_graph(
    *,
    settings: TutorSettings,
    project_root: Path | None = None,
    seed_dir: Path | None = None,
    limit: int | None = None,
    reset_database: bool = True,
    graph_name: str = "BIT_CSAT_PREPARED",
) -> dict[str, Any]:
    resolved_project_root = project_root or settings.project_root
    settings.require_neo4j()

    professor_dir = resolved_project_root / "professors"
    if not professor_dir.exists():
        raise FileNotFoundError(f"Professor dossier directory does not exist: {professor_dir}")

    seed_index = build_graph_seed_index(resolved_project_root, seed_dir=seed_dir)
    resolved_seed_dir = resolve_graph_seed_dir(resolved_project_root, seed_dir)
    if not seed_index:
        raise FileNotFoundError(
            f"No Lab 3 graph seed files were found under {resolved_seed_dir}."
        )

    markdown_paths = sorted(professor_dir.glob("*.md"))
    if limit is not None:
        markdown_paths = markdown_paths[:limit]

    if reset_database:
        reset_neo4j_database(settings)

    successes: list[dict[str, Any]] = []
    missing_seeds: list[str] = []
    failures: list[dict[str, str]] = []
    for markdown_path in markdown_paths:
        metadata = read_professor_markdown_metadata(markdown_path)
        professor_name = str(metadata.get("markdown_title") or markdown_path.stem).strip()
        detail_url = str(metadata.get("detail_url") or "").strip()
        graph_json_path = seed_index.get(markdown_path.stem)

        if not detail_url:
            failures.append(
                {
                    "professor_name": professor_name,
                    "error": f"{markdown_path.name} is missing detail_url metadata.",
                }
            )
            continue

        if graph_json_path is None:
            missing_seeds.append(professor_name)
            continue

        try:
            markdown_text = markdown_path.read_text(encoding="utf-8")
            graph = load_cached_graph(graph_json_path)
            insert_professor_graph(
                graph=graph,
                settings=settings,
                graph_name=graph_name,
                professor_name=professor_name,
                detail_url=detail_url,
            )
        except Exception as exc:
            failures.append(
                {
                    "professor_name": professor_name,
                    "error": str(exc),
                }
            )
            continue

        successes.append(
            {
                "professor_name": professor_name,
                "slug": markdown_path.stem,
                "graph_json_path": str(graph_json_path.relative_to(resolved_project_root)),
                "summary_line": build_cached_summary_line(markdown_text, professor_name),
            }
        )

    verification = verify_neo4j_graph(settings)
    return {
        "professor_count": len(markdown_paths),
        "limit_applied": limit,
        "graph_name": graph_name,
        "reset_database": reset_database,
        "seed_dir": str(resolved_seed_dir.relative_to(resolved_project_root)),
        "success_count": len(successes),
        "missing_seed_count": len(missing_seeds),
        "error_count": len(failures),
        "missing_professors": missing_seeds[:10],
        "failed_professors": failures[:10],
        "neo4j": {
            "node_count": verification["node_count"],
            "relationship_count": verification["relationship_count"],
            "professor_count": len(verification["distinct_professors"]),
            "sample_professors": verification["distinct_professors"][:10],
        },
    }


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
    "load_cached_graph",
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
    "restore_seeded_professors_to_graph",
    "restore_professor_from_cache",
    "summarize_corpus_partition",
    "verify_neo4j_graph",
]
