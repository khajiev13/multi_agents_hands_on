from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import requests
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .config import TutorSettings
from .graph_ingestion import (
    generate_professor_graph,
    insert_professor_graph,
    require_kg_gen,
    reset_neo4j_database,
    save_graph_json,
    verify_neo4j_graph,
)
from .ingestion_models import (
    OCRPageExtraction,
    OCRVisibleBlock,
    ProfessorArtifactCacheRecord,
    ProfessorCorpusPartition,
    ProfessorGraphIngestionResult,
    ProfessorIngestionPartition,
    ProfessorIngestionResult,
    ProfessorListing,
    ProfessorPageNotesResult,
)
from .legacy_cache import (
    build_cached_markdown_result,
    build_cached_summary_line,
    build_professor_cache_index,
    load_cached_graph,
    load_neo4j_detail_url_index,
    partition_professors_for_corpus,
    partition_professors_for_ingestion,
    rebuild_professor_summary,
    restore_professor_from_cache,
    summarize_corpus_partition,
)
from .markdown_corpus import (
    CorpusBuildResult,
    build_dossier_metadata,
    rebuild_markdown_corpus,
    slugify_name,
    validate_professor_markdown,
)
from .model_factory import build_model, build_ocr_model
from .ocr_transcript import (
    build_professor_markdown_from_page_notes,
    cleanup_markdown_artifact,
    extract_header_identity_lines,
    extract_identity_candidates,
    extract_professor_page_markdown,
    extract_professor_poster_notes,
    needs_top_block_fallback,
    normalize_page_ocr_markdown,
    page_markdown_to_page_extraction,
    parse_header_identity_lines,
    parse_ocr_page_notes,
    render_page_notes,
)
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


def prepare_professor_page_notes(
    *,
    listing: ProfessorListing,
    ocr_model: ChatOpenAI,
    session: requests.Session,
    project_root: Path,
    artifact_namespace: str,
) -> ProfessorPageNotesResult:
    artifact_dir = project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_name(listing.name)
    raw_page_notes_path = artifact_dir / f"{slug}-page-notes.md"
    image_urls = extract_image_urls(listing.detail_url, session)
    if not image_urls:
        raise ValueError(f"No image URLs found for {listing.name}")

    page_notes_markdown = extract_professor_poster_notes(
        model=ocr_model,
        image_urls=image_urls,
        session=session,
    )
    raw_page_notes_path.write_text(page_notes_markdown, encoding="utf-8")

    pages = parse_ocr_page_notes(page_notes_markdown)
    supplemental_header_lines: list[str] = []
    if needs_top_block_fallback(pages=pages, expected_name=listing.name):
        supplemental_header_lines = extract_header_identity_lines(
            model=ocr_model,
            image_url=image_urls[0],
            session=session,
        )

    return ProfessorPageNotesResult(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=slug,
        page_count=len(image_urls),
        image_urls=list(image_urls),
        page_notes_path=str(raw_page_notes_path.relative_to(project_root)),
        page_notes_markdown=page_notes_markdown,
        supplemental_header_lines=supplemental_header_lines,
    )


def _build_validated_canonical_markdown(
    *,
    page_notes_result: ProfessorPageNotesResult,
) -> tuple[str, Any]:
    listing = ProfessorListing(
        name=page_notes_result.name,
        detail_url=page_notes_result.detail_url,
    )
    canonical_markdown = cleanup_markdown_artifact(
        build_professor_markdown_from_page_notes(
            listing=listing,
            image_urls=page_notes_result.image_urls,
            page_notes_markdown=page_notes_result.page_notes_markdown,
            supplemental_header_lines=page_notes_result.supplemental_header_lines,
        )
    )
    validation = validate_professor_markdown(
        markdown_text=canonical_markdown,
        expected_name=listing.name,
        expected_detail_url=listing.detail_url,
    )
    if validation.status != "valid":
        raise ValueError("; ".join(validation.notes) or "OCR validation failed")
    return canonical_markdown, validation


def _strip_graph_input_metadata(markdown_text: str) -> str:
    filtered_lines: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("- detail_url:"):
            continue
        if line.startswith("- page_count:"):
            continue
        filtered_lines.append(line)

    stripped = GRAPH_INPUT_SOURCE_SECTION_PATTERN.sub("", "\n".join(filtered_lines)).strip()
    return cleanup_markdown_artifact(stripped)


def build_graph_input_text_from_page_notes(
    *,
    page_notes_result: ProfessorPageNotesResult,
) -> str:
    canonical_markdown, _validation = _build_validated_canonical_markdown(
        page_notes_result=page_notes_result,
    )
    graph_input_text = _strip_graph_input_metadata(canonical_markdown)
    if not graph_input_text:
        raise ValueError(f"Graph input text is empty for {page_notes_result.name}")
    return graph_input_text


def prepare_professor_markdown(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    ocr_model: ChatOpenAI,
    session: requests.Session,
    project_root: Path,
    artifact_namespace: str,
) -> ProfessorIngestionResult:
    professor_dir = project_root / "professors"
    professor_dir.mkdir(parents=True, exist_ok=True)

    page_notes_result = prepare_professor_page_notes(
        listing=listing,
        ocr_model=ocr_model,
        session=session,
        project_root=project_root,
        artifact_namespace=artifact_namespace,
    )
    canonical_markdown, validation = _build_validated_canonical_markdown(
        page_notes_result=page_notes_result,
    )

    professor_markdown_path = professor_dir / f"{page_notes_result.slug}.md"
    professor_markdown_path.write_text(canonical_markdown, encoding="utf-8")
    summary_line = build_cached_summary_line(canonical_markdown, listing.name)

    return ProfessorIngestionResult(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=page_notes_result.slug,
        page_count=page_notes_result.page_count,
        markdown_path=str(professor_markdown_path.relative_to(project_root)),
        page_notes_path=page_notes_result.page_notes_path,
        graph_json_path="",
        graph_html_path="",
        entity_count=0,
        relation_count=0,
        summary_line=summary_line,
        validation_status=validation.status,
        validation_notes=validation.notes,
    )


def rebuild_markdown_corpus_from_results(
    *,
    project_root: Path,
    settings: TutorSettings,
    results: Iterable[ProfessorIngestionResult],
) -> CorpusBuildResult:
    dossier_entries = [
        build_dossier_metadata(
            professor_name=result.name,
            detail_url=result.detail_url,
            markdown_path=project_root / result.markdown_path,
            project_root=project_root,
        )
        for result in sorted(results, key=lambda item: item.name)
        if result.status == "ok"
        and result.markdown_path
        and (project_root / result.markdown_path).exists()
    ]
    return rebuild_markdown_corpus(
        project_root=project_root,
        dossier_entries=dossier_entries,
        settings=settings,
    )


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
    artifact_dir = project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def prepare_step(listing_input: ProfessorListing | dict[str, Any]) -> dict[str, Any]:
        listing = _coerce_listing(listing_input)
        session = build_requests_session(user_agent)
        page_notes_result = prepare_professor_page_notes(
            listing=listing,
            ocr_model=ocr_model,
            session=session,
            project_root=project_root,
            artifact_namespace=artifact_namespace,
        )
        return {
            "listing": listing,
            "page_notes_result": page_notes_result,
        }

    def build_graph_input_step(payload: dict[str, Any]) -> dict[str, Any]:
        graph_input_text = build_graph_input_text_from_page_notes(
            page_notes_result=payload["page_notes_result"],
        )
        return {**payload, "graph_input_text": graph_input_text}

    def generate_graph_step(payload: dict[str, Any]) -> dict[str, Any]:
        page_notes_result: ProfessorPageNotesResult = payload["page_notes_result"]
        graph_input_text = payload["graph_input_text"]
        graph_json_path = artifact_dir / f"{page_notes_result.slug}-graph.json"
        graph_html_path = artifact_dir / f"{page_notes_result.slug}-graph.html"
        graph = generate_professor_graph(graph_input_text, settings)
        save_graph_json(graph, graph_json_path)
        require_kg_gen().visualize(graph, str(graph_html_path), open_in_browser=False)
        return {
            **payload,
            "graph": graph,
            "graph_json_path": str(graph_json_path.relative_to(project_root)),
            "graph_html_path": str(graph_html_path.relative_to(project_root)),
        }

    def insert_graph_step(payload: dict[str, Any]) -> ProfessorGraphIngestionResult:
        listing: ProfessorListing = payload["listing"]
        page_notes_result: ProfessorPageNotesResult = payload["page_notes_result"]
        graph_input_text: str = payload["graph_input_text"]
        graph = payload["graph"]
        insert_professor_graph(
            graph=graph,
            settings=settings,
            graph_name=graph_name,
            professor_name=listing.name,
            detail_url=listing.detail_url,
        )
        summary_line = build_cached_summary_line(graph_input_text, listing.name)
        return ProfessorGraphIngestionResult(
            name=listing.name,
            detail_url=listing.detail_url,
            slug=page_notes_result.slug,
            page_count=page_notes_result.page_count,
            page_notes_path=page_notes_result.page_notes_path,
            graph_json_path=payload["graph_json_path"],
            graph_html_path=payload["graph_html_path"],
            entity_count=len(graph.entities),
            relation_count=len(graph.relations),
            summary_line=summary_line,
            graph_input_char_count=len(graph_input_text),
            validation_status="valid",
            validation_notes=[],
        )

    return (
        RunnableLambda(prepare_step)
        | RunnableLambda(build_graph_input_step)
        | RunnableLambda(generate_graph_step)
        | RunnableLambda(insert_graph_step)
    )


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
    professor_chain = _build_professor_graph_chain(
        settings=settings,
        ocr_model=ocr_model,
        project_root=project_root,
        artifact_namespace=artifact_namespace,
        graph_name=graph_name,
        user_agent=user_agent,
    )
    slug = slugify_name(listing.name)
    artifact_dir = project_root / "artifacts" / artifact_namespace
    graph_json_path = artifact_dir / f"{slug}-graph.json"
    graph_html_path = artifact_dir / f"{slug}-graph.html"

    for attempt in range(max_attempts):
        try:
            result = professor_chain.invoke(listing)
            if attempt == 0:
                return result
            return ProfessorGraphIngestionResult(
                **{
                    **result.to_dict(),
                    "retry_count": attempt,
                }
            )
        except Exception as exc:
            if attempt + 1 < max_attempts:
                continue
            return ProfessorGraphIngestionResult(
                name=listing.name,
                detail_url=listing.detail_url,
                slug=slug,
                page_count=0,
                page_notes_path=str(
                    (artifact_dir / f"{slug}-page-notes.md").relative_to(project_root)
                ),
                graph_json_path=str(graph_json_path.relative_to(project_root)),
                graph_html_path=str(graph_html_path.relative_to(project_root)),
                entity_count=0,
                relation_count=0,
                summary_line=f"{listing.name} could not be summarized because ingestion failed.",
                retry_count=max_attempts - 1,
                graph_input_char_count=0,
                status="error",
                error=str(exc),
                validation_status="error",
                validation_notes=[str(exc)],
            )

    raise RuntimeError(f"Unreachable ingest_professor_to_graph state for {listing.name}")


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
    professor_runner = RunnableLambda(
        lambda listing: ingest_professor_to_graph(
            listing=_coerce_listing(listing),
            settings=settings,
            ocr_model=ocr_model,
            project_root=project_root,
            artifact_namespace=artifact_namespace,
            graph_name=graph_name,
            user_agent=f"{user_agent_prefix}/0.1",
        )
    )
    return professor_runner.batch(
        list(listings),
        config={"max_concurrency": max_concurrency},
    )


def ingest_professor(
    *,
    listing: ProfessorListing,
    settings: TutorSettings,
    ocr_model: ChatOpenAI,
    session: requests.Session,
    project_root: Path,
    artifact_namespace: str,
    graph_name: str = "",
    build_graph: bool = True,
) -> ProfessorIngestionResult:
    artifact_dir = project_root / "artifacts" / artifact_namespace
    artifact_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_name(listing.name)
    graph_json_path = artifact_dir / f"{slug}-graph.json"
    graph_html_path = artifact_dir / f"{slug}-graph.html"

    try:
        prepared = prepare_professor_markdown(
            listing=listing,
            settings=settings,
            ocr_model=ocr_model,
            session=session,
            project_root=project_root,
            artifact_namespace=artifact_namespace,
        )

        if not build_graph:
            return prepared

        markdown_text = (project_root / prepared.markdown_path).read_text(encoding="utf-8")
        graph = generate_professor_graph(markdown_text, settings)
        save_graph_json(graph, graph_json_path)
        from .graph_ingestion import require_kg_gen

        require_kg_gen().visualize(graph, str(graph_html_path), open_in_browser=False)
        insert_professor_graph(
            graph=graph,
            settings=settings,
            graph_name=graph_name,
            professor_name=listing.name,
            detail_url=listing.detail_url,
        )

        return ProfessorIngestionResult(
            **{
                **prepared.to_dict(),
                "graph_json_path": str(graph_json_path.relative_to(project_root)),
                "graph_html_path": str(graph_html_path.relative_to(project_root)),
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            }
        )
    except Exception as exc:
        return ProfessorIngestionResult(
            name=listing.name,
            detail_url=listing.detail_url,
            slug=slug,
            page_count=0,
            markdown_path=str((project_root / "professors" / f"{slug}.md").relative_to(project_root)),
            page_notes_path=str((project_root / "artifacts" / artifact_namespace / f"{slug}-page-notes.md").relative_to(project_root)),
            graph_json_path=str(graph_json_path.relative_to(project_root)),
            graph_html_path=str(graph_html_path.relative_to(project_root)),
            entity_count=0,
            relation_count=0,
            summary_line=f"{listing.name} could not be summarized because ingestion failed.",
            status="error",
            error=str(exc),
            validation_status="error",
            validation_notes=[str(exc)],
        )


__all__ = [
    "LISTING_URL",
    "OCRPageExtraction",
    "OCRVisibleBlock",
    "ProfessorArtifactCacheRecord",
    "ProfessorCorpusPartition",
    "ProfessorGraphIngestionResult",
    "ProfessorIngestionPartition",
    "ProfessorIngestionResult",
    "ProfessorListing",
    "ProfessorPageNotesResult",
    "build_cached_markdown_result",
    "build_cached_summary_line",
    "build_graph_input_text_from_page_notes",
    "build_model",
    "build_ocr_model",
    "build_professor_cache_index",
    "build_requests_session",
    "cleanup_markdown_artifact",
    "collect_professor_links",
    "collect_professor_links_from_page",
    "discover_listing_pages",
    "extract_header_identity_lines",
    "extract_identity_candidates",
    "extract_image_urls",
    "extract_professor_page_markdown",
    "extract_professor_poster_notes",
    "fetch_soup",
    "ingest_professor",
    "ingest_professor_to_graph",
    "ingest_professors_to_graph",
    "load_cached_graph",
    "load_neo4j_detail_url_index",
    "needs_top_block_fallback",
    "normalize_page_ocr_markdown",
    "page_markdown_to_page_extraction",
    "parse_header_identity_lines",
    "parse_ocr_page_notes",
    "partition_professors_for_corpus",
    "partition_professors_for_ingestion",
    "prepare_professor_markdown",
    "prepare_professor_page_notes",
    "rebuild_markdown_corpus_from_results",
    "rebuild_professor_summary",
    "render_page_notes",
    "reset_neo4j_database",
    "restore_professor_from_cache",
    "summarize_corpus_partition",
    "verify_neo4j_graph",
]
