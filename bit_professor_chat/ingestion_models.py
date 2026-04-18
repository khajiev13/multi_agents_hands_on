from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OCRBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading_original: str | None = None
    heading_english: str | None = None
    role: Literal["identity", "section", "continuation", "footer", "uncertain"]
    lines: list[str] = Field(default_factory=list)


class OCRPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    image_url: str
    blocks: list[OCRBlock] = Field(default_factory=list)
    uncertain_lines: list[str] = Field(default_factory=list)


class DossierSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    heading_original: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ProfessorDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    detail_url: str
    source_page_urls: list[str] = Field(default_factory=list)
    basic_information: DossierSection
    sections: list[DossierSection] = Field(default_factory=list)
    uncertain_lines: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    ocr_json_path: str = ""
    dossier_json_path: str = ""
    status: str = "ok"
    error: str | None = None
    validation_status: str = "unknown"
    validation_notes: list[str] = field(default_factory=list)
    validation_checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_markdown_path(self, project_root: Path) -> Path:
        return project_root / self.markdown_path


@dataclass(frozen=True)
class ProfessorOCRBundleResult:
    name: str
    detail_url: str
    slug: str
    page_count: int
    image_urls: list[str] = field(default_factory=list)
    page_markdowns: list[str] = field(default_factory=list)
    ocr_markdown_path: str = ""
    ocr_markdown_text: str = ""
    page_notes_path: str = ""
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["page_notes_markdown"] = self.page_notes_markdown
        return payload

    def resolved_page_notes_path(self, project_root: Path) -> Path:
        relative_path = self.page_notes_path or self.ocr_markdown_path
        return project_root / relative_path

    def resolved_ocr_markdown_path(self, project_root: Path) -> Path:
        return project_root / self.ocr_markdown_path

    @property
    def page_notes_markdown(self) -> str:
        return self.ocr_markdown_text


ProfessorPageNotesResult = ProfessorOCRBundleResult


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
    ocr_json_path: str = ""
    dossier_json_path: str = ""
    status: str = "ok"
    error: str | None = None
    validation_status: str = "unknown"
    validation_notes: list[str] = field(default_factory=list)
    validation_checks: dict[str, bool] = field(default_factory=dict)

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
    dossier_json_path: str | None
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


class ProfessorArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    detail_url: str
    slug: str
    page_count: int
    image_urls: list[str] = Field(default_factory=list)
    artifact_namespace: str
    source_file: str


class StructuredProfessorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    professor_id: str = ""
    name_local: str = ""
    name_english: str = ""
    aliases: list[str] = Field(default_factory=list)
    title: str = ""
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    biography_text: str = ""
    source_file: str = ""


class StructuredOrganizationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    org_id: str = ""
    name: str
    aliases: list[str] = Field(default_factory=list)
    org_type: Literal[
        "school",
        "university",
        "association",
        "conference",
        "journal",
        "funder",
        "company",
        "unknown",
    ] = "unknown"


class StructuredResearchTopicRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str = ""
    name: str
    normalized_name: str = ""
    language: str = ""


class StructuredEducationExperienceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience_id: str = ""
    degree: str = ""
    field: str = ""
    organization_name_raw: str = ""
    start_text: str = ""
    end_text: str = ""
    is_current: bool = False
    order: int = 0
    raw_text: str = ""


class StructuredEmploymentExperienceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience_id: str = ""
    role_title: str = ""
    organization_name_raw: str = ""
    location: str = ""
    start_text: str = ""
    end_text: str = ""
    is_current: bool = False
    order: int = 0
    raw_text: str = ""


class StructuredAcademicServiceRoleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_id: str = ""
    role_title: str = ""
    organization_name_raw: str = ""
    service_type: Literal[
        "reviewer",
        "member",
        "committee",
        "chair",
        "vice_president",
        "expert",
        "unknown",
    ] = "unknown"
    raw_text: str = ""


class StructuredAwardRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    award_id: str = ""
    name: str
    category: str = ""
    year: str = ""
    granting_org_name_raw: str = ""
    level: str = ""
    raw_text: str = ""


class StructuredPublicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_id: str = ""
    title: str
    authors_raw: str = ""
    year: str = ""
    venue: str = ""
    publication_type: Literal["journal", "conference", "book", "unknown"] = "unknown"
    doi_or_isbn: str = ""
    raw_text: str = ""


class StructuredProfessorReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: ProfessorArtifactMetadata
    professor: StructuredProfessorRecord
    organizations: list[StructuredOrganizationRecord] = Field(default_factory=list)
    affiliation_organization_names: list[str] = Field(default_factory=list)
    research_topics: list[StructuredResearchTopicRecord] = Field(default_factory=list)
    education_experiences: list[StructuredEducationExperienceRecord] = Field(
        default_factory=list
    )
    employment_experiences: list[StructuredEmploymentExperienceRecord] = Field(
        default_factory=list
    )
    academic_service_roles: list[StructuredAcademicServiceRoleRecord] = Field(
        default_factory=list
    )
    awards: list[StructuredAwardRecord] = Field(default_factory=list)
    publications: list[StructuredPublicationRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ProfessorPreInsertionResult:
    name: str
    detail_url: str
    slug: str
    page_count: int
    status: str = "ok"
    failure_stage: str | None = None
    error: str | None = None
    stage_statuses: dict[str, bool] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    page_graph_count: int = 0
    aggregate_entity_count: int = 0
    aggregate_relation_count: int = 0
    clustered_entity_count: int = 0
    clustered_relation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfessorPreInsertionBatchResult:
    artifact_namespace: str
    artifact_dir: str
    listing_pages: list[str] = field(default_factory=list)
    professor_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    stage_counts: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    results: list[ProfessorPreInsertionResult] = field(default_factory=list)
    corpus_artifact_paths: dict[str, str] = field(default_factory=dict)
    insertion_status: str = "to_be_implemented"
    next_stage: str = "structured_json -> typed Cypher -> Neo4j schema insertion"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [result.to_dict() for result in self.results]
        return payload


@dataclass(frozen=True)
class StructuredSeedExportResult:
    artifact_namespace: str
    seed_dir: str
    professor_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    exported_professors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredSeedInsertionResult:
    seed_dir: str
    professor_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    inserted_professors: list[str] = field(default_factory=list)
    node_count: int = 0
    relationship_count: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
