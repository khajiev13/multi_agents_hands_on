from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

from .config import TutorSettings
from .graph_ingestion import insert_professor_graph, require_kg_gen, save_graph_json
from .ingestion_models import (
    ProfessorArtifactCacheRecord,
    ProfessorCorpusPartition,
    ProfessorIngestionPartition,
    ProfessorIngestionResult,
    ProfessorListing,
)
from .markdown_corpus import validate_professor_markdown


def extract_markdown_section_lines(
    markdown_text: str, section_titles: Sequence[str]
) -> list[str]:
    targets = {title.strip().lower() for title in section_titles}
    lines = markdown_text.splitlines()
    collected: list[str] = []
    capturing = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip().lower()
            if heading_text in targets:
                capturing = True
                continue
            if capturing:
                break
            continue

        if not capturing:
            continue

        if stripped.startswith(("- ", "* ")):
            collected.append(stripped[2:].strip())
            continue

        numbered_match = re.match(r"\d+\.\s+(.*)", stripped)
        if numbered_match:
            collected.append(numbered_match.group(1).strip())
            continue

        collected.append(stripped)

    return [line for line in collected if line]


def build_cached_summary_line(markdown_text: str, fallback_name: str) -> str:
    research_lines = extract_markdown_section_lines(
        markdown_text, ["Research Interests", "research interests"]
    )
    if research_lines:
        return f"{fallback_name}: {', '.join(research_lines[:3])}"

    basic_lines = extract_markdown_section_lines(
        markdown_text, ["Basic Information", "basic information"]
    )
    if basic_lines:
        return f"{fallback_name}: {', '.join(basic_lines[:3])}"

    return f"{fallback_name}: restored from cached graph data"


def load_cached_graph(graph_json_path: Path) -> Any:
    payload = json.loads(graph_json_path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        entities=payload.get("entities", []),
        relations=payload.get("relations", []),
    )


def read_professor_markdown_metadata(markdown_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "markdown_title": markdown_path.stem,
        "detail_url": None,
        "page_count": None,
    }
    try:
        lines = markdown_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return metadata

    for line in lines[:40]:
        stripped = line.strip()
        if not stripped:
            continue

        heading_match = re.match(r"^#\s+(.+)$", stripped)
        if heading_match and metadata["markdown_title"] == markdown_path.stem:
            metadata["markdown_title"] = heading_match.group(1).strip()
            continue

        detail_match = re.match(r"^-\s*detail_url:\s*(.+?)\s*$", stripped, re.I)
        if detail_match:
            metadata["detail_url"] = detail_match.group(1).strip()
            continue

        page_count_match = re.match(r"^-\s*page_count:\s*(\d+)\s*$", stripped, re.I)
        if page_count_match:
            metadata["page_count"] = int(page_count_match.group(1))
            continue

    return metadata


def _latest_path(paths: Sequence[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, str(path)))


def find_professor_artifact_path(project_root: Path, slug: str, suffix: str) -> Path | None:
    if not slug:
        return None
    artifact_root = project_root / "artifacts"
    if not artifact_root.exists():
        return None
    return _latest_path(list(artifact_root.rglob(f"{slug}{suffix}")))


def build_professor_cache_index(
    project_root: Path,
) -> dict[str, ProfessorArtifactCacheRecord]:
    professor_dir = project_root / "professors"
    if not professor_dir.exists():
        return {}

    cache_index: dict[str, ProfessorArtifactCacheRecord] = {}
    for markdown_path in sorted(professor_dir.glob("*.md")):
        metadata = read_professor_markdown_metadata(markdown_path)
        detail_url = metadata.get("detail_url")
        if not detail_url:
            continue

        slug = markdown_path.stem
        graph_json_path = find_professor_artifact_path(project_root, slug, "-graph.json")
        graph_html_path = find_professor_artifact_path(project_root, slug, "-graph.html")
        page_notes_path = find_professor_artifact_path(project_root, slug, "-page-notes.md")

        record = ProfessorArtifactCacheRecord(
            detail_url=detail_url,
            slug=slug,
            markdown_title=str(metadata.get("markdown_title") or markdown_path.stem),
            markdown_path=str(markdown_path),
            graph_json_path=str(graph_json_path) if graph_json_path else None,
            graph_html_path=str(graph_html_path) if graph_html_path else None,
            page_notes_path=str(page_notes_path) if page_notes_path else None,
            page_count=metadata.get("page_count"),
            markdown_mtime=markdown_path.stat().st_mtime,
            graph_json_mtime=graph_json_path.stat().st_mtime if graph_json_path else None,
        )

        current = cache_index.get(detail_url)
        if current is None or record.markdown_mtime >= current.markdown_mtime:
            cache_index[detail_url] = record

    return cache_index


def build_cached_markdown_result(
    *,
    listing: ProfessorListing,
    cache_entry: ProfessorArtifactCacheRecord,
    project_root: Path,
    note: str | None = None,
) -> ProfessorIngestionResult:
    markdown_path = Path(cache_entry.markdown_path)
    cached_markdown = markdown_path.read_text(encoding="utf-8")
    summary_line = build_cached_summary_line(cached_markdown, listing.name)
    if note:
        summary_line = f"{summary_line} ({note})"

    validation = validate_professor_markdown(
        markdown_text=cached_markdown,
        expected_name=listing.name,
        expected_detail_url=listing.detail_url,
    )
    return ProfessorIngestionResult(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=cache_entry.slug,
        page_count=cache_entry.page_count or 0,
        markdown_path=str(markdown_path.relative_to(project_root)),
        page_notes_path=(
            str(Path(cache_entry.page_notes_path).relative_to(project_root))
            if cache_entry.page_notes_path
            else ""
        ),
        graph_json_path="",
        graph_html_path="",
        entity_count=0,
        relation_count=0,
        summary_line=summary_line,
        validation_status=validation.status,
        validation_notes=validation.notes,
    )


def summarize_corpus_partition(partition: ProfessorCorpusPartition) -> dict[str, int]:
    return {
        "live_professors": partition.ready_count + partition.needs_rebuild_count,
        "ready_count": partition.ready_count,
        "needs_rebuild_count": partition.needs_rebuild_count,
        "invalid_cached_count": partition.invalid_cached_count,
    }


def partition_professors_for_corpus(
    listings: Sequence[ProfessorListing],
    *,
    project_root: Path,
) -> ProfessorCorpusPartition:
    cache_index = build_professor_cache_index(project_root)
    ready: list[ProfessorListing] = []
    needs_rebuild: list[ProfessorListing] = []
    invalid_cached: list[ProfessorListing] = []

    for listing in listings:
        cache_entry = cache_index.get(listing.detail_url)
        if cache_entry is None or not cache_entry.has_markdown:
            needs_rebuild.append(listing)
            continue

        markdown_path = Path(cache_entry.markdown_path)
        cached_markdown = markdown_path.read_text(encoding="utf-8")
        validation = validate_professor_markdown(
            markdown_text=cached_markdown,
            expected_name=listing.name,
            expected_detail_url=listing.detail_url,
        )
        if validation.status == "valid":
            ready.append(listing)
            continue

        invalid_cached.append(listing)
        needs_rebuild.append(listing)

    return ProfessorCorpusPartition(
        ready=ready,
        needs_rebuild=needs_rebuild,
        cache_index=cache_index,
        invalid_cached=invalid_cached,
    )


def load_neo4j_detail_url_index(settings: TutorSettings) -> set[str]:
    from .graph_ingestion import require_neo4j_driver

    settings.require_neo4j()
    neo4j_driver = require_neo4j_driver()
    driver = neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session(database=settings.neo4j_database) as neo4j_session:
            return {
                record["detail_url"]
                for record in neo4j_session.run(
                    """
                    MATCH (n:Entity)
                    WHERE n.detail_url IS NOT NULL AND n.professor_name IS NOT NULL
                    RETURN DISTINCT n.detail_url AS detail_url
                    ORDER BY detail_url
                    """
                )
            }
    finally:
        driver.close()


def partition_professors_for_ingestion(
    listings: Sequence[ProfessorListing],
    *,
    settings: TutorSettings,
    project_root: Path,
) -> ProfessorIngestionPartition:
    cache_index = build_professor_cache_index(project_root)
    neo4j_detail_urls = load_neo4j_detail_url_index(settings)

    ready: list[ProfessorListing] = []
    needs_restore: list[ProfessorListing] = []
    needs_crawl: list[ProfessorListing] = []

    for listing in listings:
        cache_entry = cache_index.get(listing.detail_url)
        if cache_entry is None or not cache_entry.has_graph_json:
            needs_crawl.append(listing)
            continue
        if listing.detail_url in neo4j_detail_urls:
            ready.append(listing)
            continue
        needs_restore.append(listing)

    return ProfessorIngestionPartition(
        ready=ready,
        needs_restore=needs_restore,
        needs_crawl=needs_crawl,
        cache_index=cache_index,
        neo4j_detail_urls=sorted(neo4j_detail_urls),
    )


def restore_professor_from_cache(
    *,
    listing: ProfessorListing,
    cache_entry: ProfessorArtifactCacheRecord,
    settings: TutorSettings,
    graph_name: str,
) -> ProfessorIngestionResult:
    if not cache_entry.graph_json_path:
        raise ValueError(f"No cached graph JSON found for {listing.detail_url}")

    project_root = settings.project_root
    artifact_dir = project_root / "artifacts" / graph_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = Path(cache_entry.markdown_path)
    graph_json_path = Path(cache_entry.graph_json_path)
    graph_html_path = (
        Path(cache_entry.graph_html_path)
        if cache_entry.graph_html_path
        else artifact_dir / f"{cache_entry.slug}-graph.html"
    )
    page_notes_path = (
        Path(cache_entry.page_notes_path)
        if cache_entry.page_notes_path
        else artifact_dir / f"{cache_entry.slug}-page-notes.md"
    )

    cached_markdown = markdown_path.read_text(encoding="utf-8")
    summary_line = build_cached_summary_line(cached_markdown, listing.name)
    graph = load_cached_graph(graph_json_path)
    validation = validate_professor_markdown(
        markdown_text=cached_markdown,
        expected_name=listing.name,
        expected_detail_url=listing.detail_url,
    )

    save_graph_json(graph, graph_json_path)
    require_kg_gen().visualize(graph, str(graph_html_path), open_in_browser=False)
    insert_professor_graph(
        graph=graph,
        settings=settings,
        graph_name=graph_name,
        professor_name=listing.name,
        detail_url=listing.detail_url,
    )

    return ProfessorIngestionResult(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=cache_entry.slug,
        page_count=cache_entry.page_count or 0,
        markdown_path=str(markdown_path.relative_to(project_root)),
        page_notes_path=str(page_notes_path.relative_to(project_root)),
        graph_json_path=str(graph_json_path.relative_to(project_root)),
        graph_html_path=str(graph_html_path.relative_to(project_root)),
        entity_count=len(graph.entities),
        relation_count=len(graph.relations),
        summary_line=summary_line,
        validation_status=validation.status,
        validation_notes=validation.notes,
    )


def rebuild_professor_summary(project_root: Path, entries: Iterable[ProfessorIngestionResult]) -> Path:
    lines = ["# BIT CSAT Professors", ""]
    for entry in sorted(entries, key=lambda item: item.name):
        lines.append(f"- {entry.summary_line}")
    lines.append("")
    output_path = project_root / "professors.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
