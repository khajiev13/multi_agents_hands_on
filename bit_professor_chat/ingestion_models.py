from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OCRVisibleBlock:
    heading_text: str
    block_role: str
    content_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OCRPageExtraction:
    page_number: int
    image_url: str
    blocks: list[OCRVisibleBlock] = field(default_factory=list)
    uncertain_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProfessorListing:
    name: str
    detail_url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ProfessorIngestionResult:
    name: str
    detail_url: str
    slug: str
    page_count: int
    markdown_path: str
    page_notes_path: str
    graph_json_path: str
    graph_html_path: str
    entity_count: int
    relation_count: int
    summary_line: str
    status: str = "ok"
    error: str | None = None
    validation_status: str = "unknown"
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_markdown_path(self, project_root: Path) -> Path:
        return project_root / self.markdown_path


@dataclass(frozen=True)
class ProfessorPageNotesResult:
    name: str
    detail_url: str
    slug: str
    page_count: int
    image_urls: list[str] = field(default_factory=list)
    page_notes_path: str = ""
    page_notes_markdown: str = ""
    supplemental_header_lines: list[str] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_page_notes_path(self, project_root: Path) -> Path:
        return project_root / self.page_notes_path


@dataclass(frozen=True)
class ProfessorGraphIngestionResult:
    name: str
    detail_url: str
    slug: str
    page_count: int
    page_notes_path: str
    graph_json_path: str
    graph_html_path: str
    entity_count: int
    relation_count: int
    summary_line: str
    retry_count: int = 0
    graph_input_char_count: int = 0
    status: str = "ok"
    error: str | None = None
    validation_status: str = "unknown"
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfessorArtifactCacheRecord:
    detail_url: str
    slug: str
    markdown_title: str
    markdown_path: str
    graph_json_path: str | None
    graph_html_path: str | None
    page_notes_path: str | None
    page_count: int | None
    markdown_mtime: float
    graph_json_mtime: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_graph_json(self) -> bool:
        return self.graph_json_path is not None and Path(self.graph_json_path).exists()

    @property
    def has_graph_html(self) -> bool:
        return self.graph_html_path is not None and Path(self.graph_html_path).exists()

    @property
    def has_markdown(self) -> bool:
        return Path(self.markdown_path).exists()


@dataclass(frozen=True)
class ProfessorIngestionPartition:
    ready: list[ProfessorListing]
    needs_restore: list[ProfessorListing]
    needs_crawl: list[ProfessorListing]
    cache_index: dict[str, ProfessorArtifactCacheRecord]
    neo4j_detail_urls: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": [listing.to_dict() for listing in self.ready],
            "needs_restore": [listing.to_dict() for listing in self.needs_restore],
            "needs_crawl": [listing.to_dict() for listing in self.needs_crawl],
            "cache_index": {
                detail_url: record.to_dict() for detail_url, record in self.cache_index.items()
            },
            "neo4j_detail_urls": list(self.neo4j_detail_urls),
        }

    @property
    def ready_count(self) -> int:
        return len(self.ready)

    @property
    def needs_restore_count(self) -> int:
        return len(self.needs_restore)

    @property
    def needs_crawl_count(self) -> int:
        return len(self.needs_crawl)


@dataclass(frozen=True)
class ProfessorCorpusPartition:
    ready: list[ProfessorListing]
    needs_rebuild: list[ProfessorListing]
    cache_index: dict[str, ProfessorArtifactCacheRecord]
    invalid_cached: list[ProfessorListing]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": [listing.to_dict() for listing in self.ready],
            "needs_rebuild": [listing.to_dict() for listing in self.needs_rebuild],
            "cache_index": {
                detail_url: record.to_dict() for detail_url, record in self.cache_index.items()
            },
            "invalid_cached": [listing.to_dict() for listing in self.invalid_cached],
        }

    @property
    def ready_count(self) -> int:
        return len(self.ready)

    @property
    def needs_rebuild_count(self) -> int:
        return len(self.needs_rebuild)

    @property
    def invalid_cached_count(self) -> int:
        return len(self.invalid_cached)
