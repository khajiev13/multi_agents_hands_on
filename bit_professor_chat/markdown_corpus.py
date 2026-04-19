from __future__ import annotations

import json
import re
import shutil
from hashlib import sha1
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import TutorSettings
from .corpus_paths import resolve_professor_corpus_dir
from .ingestion_models import ProfessorDossier, ProfessorListing
from .rerank_client import JinaStyleReranker


SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.M)
TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.M)
DETAIL_URL_PATTERN = re.compile(r"^-\s*detail_url:\s*(.+?)\s*$", re.M)
PAGE_COUNT_PATTERN = re.compile(r"^-\s*page_count:\s*(\d+)\s*$", re.M)
HTML_TAG_PATTERN = re.compile(
    r"</?(?:"
    r"html|body|head|title|meta|link|style|script|div|span|p|br|hr|img|a|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|dl|dt|dd|h[1-6]|strong|em|b|i|u|sup|sub|section|article|header|footer|main|nav|aside"
    r")(?:\s+[^>]*)?\s*/?>",
    re.I,
)
OCR_TOKEN_PATTERN = re.compile(r"\b\d+text\b", re.I)
BRACE_RUN_PATTERN = re.compile(r"[{}]{4,}")
FLOAT_REPETITION_PATTERN = re.compile(r"\b(?:1\.0(?:\s+|$)){3,}", re.I)
ALNUM_OR_CJK_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")

VALIDATION_CHECK_KEYS = (
    "detail_url_matches",
    "name_matches",
    "has_basic_information",
    "has_any_section",
    "minimum_clean_content",
    "no_blocked_artifacts",
)

BLOCKED_ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ocr_token", OCR_TOKEN_PATTERN),
    ("html_tag", HTML_TAG_PATTERN),
    ("brace_run", BRACE_RUN_PATTERN),
    ("float_repetition", FLOAT_REPETITION_PATTERN),
)


def slugify_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", name.lower())


def _roman_name_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z]+", name.lower())


def names_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False

    left_compact = compact_name(left)
    right_compact = compact_name(right)
    if not left_compact or not right_compact:
        return False
    if left_compact == right_compact:
        return True
    if left_compact in right_compact or right_compact in left_compact:
        return True
    left_tokens = _roman_name_tokens(left)
    right_tokens = _roman_name_tokens(right)
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        if left_tokens == right_tokens:
            return True
        if sorted(left_tokens) == sorted(right_tokens):
            return True
        if set(left_tokens).issubset(set(right_tokens)) or set(right_tokens).issubset(set(left_tokens)):
            return True
    return SequenceMatcher(a=left_compact, b=right_compact).ratio() >= 0.9


def normalize_section_request(sections: Sequence[str] | None) -> list[str]:
    if not sections:
        return []
    seen: list[str] = []
    for section in sections:
        cleaned = section.strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def extract_title(markdown_text: str) -> str:
    match = TITLE_PATTERN.search(markdown_text)
    return match.group(1).strip() if match else ""


def extract_detail_url(markdown_text: str) -> str | None:
    match = DETAIL_URL_PATTERN.search(markdown_text)
    return match.group(1).strip() if match else None


def extract_page_count(markdown_text: str) -> int | None:
    match = PAGE_COUNT_PATTERN.search(markdown_text)
    return int(match.group(1)) if match else None


def parse_markdown_sections(markdown_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        section_match = re.match(r"^##\s+(.+)$", line)
        if section_match:
            current_section = section_match.group(1).strip()
            sections[current_section] = []
            continue
        if current_section is None:
            continue
        if line.startswith("### "):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ")):
            sections[current_section].append(stripped[2:].strip())
            continue
        numbered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered_match:
            sections[current_section].append(numbered_match.group(1).strip())
            continue
        sections[current_section].append(stripped)
    return sections


def clean_quality_line(line: str) -> str:
    cleaned = HTML_TAG_PATTERN.sub(" ", line)
    cleaned = OCR_TOKEN_PATTERN.sub(" ", cleaned)
    cleaned = BRACE_RUN_PATTERN.sub(" ", cleaned)
    cleaned = FLOAT_REPETITION_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"[*_`]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.|")
    if not cleaned or not ALNUM_OR_CJK_PATTERN.search(cleaned):
        return ""
    return cleaned


def detect_blocked_artifacts(text: str) -> list[str]:
    detected: list[str] = []
    for label, pattern in BLOCKED_ARTIFACT_PATTERNS:
        if pattern.search(text):
            detected.append(label)
    return detected


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_name_candidates(markdown_text: str, sections: dict[str, list[str]]) -> list[str]:
    candidates: list[str] = []
    title = extract_title(markdown_text)
    if title:
        candidates.append(title)

    for line in sections.get("Basic Information", []):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("name:"):
            candidates.append(stripped.split(":", 1)[1].strip())
        if stripped.startswith("姓名："):
            candidates.append(stripped.split("：", 1)[1].strip())
        if stripped.startswith("姓名:"):
            candidates.append(stripped.split(":", 1)[1].strip())
        if stripped.startswith("英文名："):
            candidates.append(stripped.split("：", 1)[1].strip())

    seen: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


@dataclass(frozen=True)
class MarkdownValidation:
    status: str
    notes: list[str] = field(default_factory=list)
    discovered_names: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    blocked_artifacts: list[str] = field(default_factory=list)


def _finalize_validation(
    *,
    notes: list[str],
    quality_notes: list[str],
    discovered_names: list[str],
    checks: dict[str, bool],
    blocked_artifacts: list[str],
) -> MarkdownValidation:
    combined_notes = [*notes, *quality_notes]
    status = "valid" if not combined_notes else "invalid"
    return MarkdownValidation(
        status=status,
        notes=combined_notes,
        discovered_names=discovered_names,
        quality_notes=quality_notes,
        checks=checks,
        blocked_artifacts=blocked_artifacts,
    )


def validate_professor_dossier(
    *,
    dossier: ProfessorDossier,
    listing: ProfessorListing,
    rendered_markdown: str | None = None,
) -> MarkdownValidation:
    from .markdown_render import render_professor_markdown

    notes: list[str] = []
    quality_notes: list[str] = []
    discovered_names = [dossier.title.strip()] if dossier.title.strip() else []
    actual_detail_url = dossier.detail_url.strip()
    basic_information_lines = [
        clean_quality_line(line) for line in dossier.basic_information.bullets
    ]
    basic_information_lines = [line for line in basic_information_lines if line]
    section_lines = [
        clean_quality_line(line)
        for section in dossier.sections
        for line in section.bullets
    ]
    section_lines = [line for line in section_lines if line]
    blocked_artifacts = detect_blocked_artifacts(
        rendered_markdown or render_professor_markdown(dossier)
    )
    has_matching_name = bool(discovered_names) and names_similar(
        listing.name, discovered_names[0]
    )

    if actual_detail_url != listing.detail_url:
        notes.append(
            f"detail_url mismatch: expected {listing.detail_url}, found {actual_detail_url or 'missing'}"
        )

    if discovered_names and not has_matching_name:
        notes.append(
            f"name mismatch: expected {listing.name}, found {', '.join(discovered_names[:3])}"
        )
    elif not discovered_names:
        notes.append("no readable professor title found in the synthesized dossier")

    if not basic_information_lines:
        quality_notes.append("missing Basic Information section content")

    if not dossier.sections or not section_lines:
        quality_notes.append("missing at least one non-empty dossier section")

    if len([*basic_information_lines, *section_lines]) < 3:
        quality_notes.append("not enough clean dossier bullets: found fewer than 3")

    if blocked_artifacts:
        quality_notes.append(
            "blocked artifact patterns detected: " + ", ".join(blocked_artifacts)
        )

    checks = {
        "detail_url_matches": actual_detail_url == listing.detail_url,
        "name_matches": has_matching_name,
        "has_basic_information": bool(basic_information_lines),
        "has_any_section": bool(dossier.sections and section_lines),
        "minimum_clean_content": len([*basic_information_lines, *section_lines]) >= 3,
        "no_blocked_artifacts": not blocked_artifacts,
    }
    return _finalize_validation(
        notes=notes,
        quality_notes=quality_notes,
        discovered_names=discovered_names,
        checks=checks,
        blocked_artifacts=blocked_artifacts,
    )


def validate_professor_markdown(
    *,
    markdown_text: str,
    expected_name: str,
    expected_detail_url: str,
) -> MarkdownValidation:
    notes: list[str] = []
    quality_notes: list[str] = []
    sections = parse_markdown_sections(markdown_text)
    discovered_names = extract_name_candidates(markdown_text, sections)
    actual_detail_url = extract_detail_url(markdown_text)
    basic_information_lines = [
        clean_quality_line(line) for line in sections.get("Basic Information", [])
    ]
    basic_information_lines = [line for line in basic_information_lines if line]
    section_lines = [
        clean_quality_line(line)
        for section_name, lines in sections.items()
        if section_name not in {"Basic Information", "Source Pages", "Uncertain or Illegible Text"}
        for line in lines
    ]
    section_lines = [line for line in section_lines if line]
    populated_sections = [
        section_name
        for section_name, lines in sections.items()
        if section_name not in {"Basic Information", "Source Pages", "Uncertain or Illegible Text"}
        and any(clean_quality_line(line) for line in lines)
    ]
    clean_content_lines = [
        cleaned
        for section_name, lines in sections.items()
        if section_name not in {"Source Pages", "Uncertain or Illegible Text"}
        for cleaned in (clean_quality_line(line) for line in lines)
        if cleaned
    ]
    blocked_artifacts = detect_blocked_artifacts(markdown_text)
    has_matching_name = bool(discovered_names) and any(
        names_similar(expected_name, candidate) for candidate in discovered_names
    )

    if actual_detail_url != expected_detail_url:
        notes.append(
            f"detail_url mismatch: expected {expected_detail_url}, found {actual_detail_url or 'missing'}"
        )

    if discovered_names and not has_matching_name:
        notes.append(
            f"name mismatch: expected {expected_name}, found {', '.join(discovered_names[:3])}"
        )
    elif not discovered_names:
        notes.append("no readable professor name found in the markdown header or basic information")

    if not basic_information_lines:
        quality_notes.append("missing Basic Information section content")

    if not populated_sections or not section_lines:
        quality_notes.append("missing at least one non-empty dossier section")

    if len(clean_content_lines) < 3:
        quality_notes.append("not enough clean content bullets outside Source Pages: found fewer than 3")

    if blocked_artifacts:
        quality_notes.append(
            "blocked artifact patterns detected: " + ", ".join(blocked_artifacts)
        )

    checks = {
        "detail_url_matches": actual_detail_url == expected_detail_url,
        "name_matches": has_matching_name,
        "has_basic_information": bool(basic_information_lines),
        "has_any_section": bool(populated_sections and section_lines),
        "minimum_clean_content": len(clean_content_lines) >= 3,
        "no_blocked_artifacts": not blocked_artifacts,
    }
    return _finalize_validation(
        notes=notes,
        quality_notes=quality_notes,
        discovered_names=discovered_names,
        checks=checks,
        blocked_artifacts=blocked_artifacts,
    )


@dataclass(frozen=True)
class ProfessorDossierMetadata:
    professor_name: str
    slug: str
    detail_url: str
    markdown_path: str
    page_count: int
    research_interests: list[str]
    validation_status: str
    validation_notes: list[str]
    validation_checks: dict[str, bool]
    aliases: list[str]
    available_sections: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalProfessorMarkdownTask:
    professor_name: str
    detail_url: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TopicChunkMatch:
    professor_name: str
    detail_url: str
    markdown_path: str
    section: str
    snippet: str
    vector_score: float
    rerank_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopicProfessorMatch:
    professor_name: str
    detail_url: str
    research_interests: list[str]
    rerank_score: float
    matched_sections: list[str]
    snippets: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusBuildResult:
    valid_professor_count: int
    invalid_dossier_count: int
    chunk_count: int
    index_path: str
    retrieval_manifest_path: str
    vector_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dossier_metadata(
    *,
    professor_name: str,
    detail_url: str,
    markdown_path: Path,
    project_root: Path,
) -> ProfessorDossierMetadata:
    markdown_text = read_markdown(markdown_path)
    sections = parse_markdown_sections(markdown_text)
    validation = validate_professor_markdown(
        markdown_text=markdown_text,
        expected_name=professor_name,
        expected_detail_url=detail_url,
    )
    title = extract_title(markdown_text)
    aliases = [professor_name]
    for candidate in [title, *validation.discovered_names]:
        cleaned = candidate.strip()
        if cleaned and cleaned not in aliases:
            aliases.append(cleaned)

    return ProfessorDossierMetadata(
        professor_name=professor_name,
        slug=markdown_path.stem,
        detail_url=detail_url,
        markdown_path=str(markdown_path.relative_to(project_root)),
        page_count=extract_page_count(markdown_text) or 0,
        research_interests=sections.get("Research Interests", []),
        validation_status=validation.status,
        validation_notes=validation.notes,
        validation_checks=validation.checks,
        aliases=aliases,
        available_sections=list(sections.keys()),
    )


def deslugify_professor_slug(slug: str) -> str:
    tokens = [token for token in slug.replace("_", "-").split("-") if token]
    if not tokens:
        return slug
    return " ".join(token.capitalize() for token in tokens)


def discover_professor_markdown_paths(
    *,
    project_root: Path,
    professor_dir: Path | None = None,
) -> list[str]:
    search_dir = resolve_professor_corpus_dir(project_root, professor_dir)
    if not search_dir.exists():
        raise ValueError(
            f"Professor markdown directory not found at {search_dir}. "
            "Run the instructor prep workflow first."
        )

    markdown_paths = sorted(search_dir.glob("*.md"))
    if not markdown_paths:
        raise ValueError(
            f"No professor markdown files found in {search_dir}. "
            "Run the instructor prep workflow first."
        )
    return [str(path.relative_to(project_root)) for path in markdown_paths]


def build_local_professor_markdown_tasks(
    *,
    project_root: Path,
    markdown_paths: Sequence[str],
) -> list[LocalProfessorMarkdownTask]:
    tasks: list[LocalProfessorMarkdownTask] = []
    for raw_path in markdown_paths:
        markdown_path = project_root / raw_path
        if not markdown_path.exists():
            raise ValueError(f"Markdown file not found: {markdown_path}")

        markdown_text = read_markdown(markdown_path)
        detail_url = extract_detail_url(markdown_text)
        if not detail_url:
            raise ValueError(
                f"Missing `detail_url` in {markdown_path}. "
                "Run the instructor prep workflow first."
            )

        tasks.append(
            LocalProfessorMarkdownTask(
                professor_name=deslugify_professor_slug(markdown_path.stem),
                detail_url=detail_url,
                markdown_path=str(markdown_path.relative_to(project_root)),
            )
        )

    return tasks


def build_embedding_model(settings: TutorSettings) -> OpenAIEmbeddings:
    config = settings.require_embeddings()
    return OpenAIEmbeddings(
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        dimensions=config["dimensions"],
        chunk_size=config["batch_size"],
        max_retries=2,
        timeout=60,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def chunk_professor_markdown(
    *,
    metadata: ProfessorDossierMetadata,
    markdown_text: str,
    max_chars: int = 1200,
    overlap: int = 160,
) -> list[Document]:
    sections = parse_markdown_sections(markdown_text)
    documents: list[Document] = []
    for section_title, lines in sections.items():
        if section_title == "Source Pages":
            continue
        if metadata.validation_status != "valid":
            continue
        if not lines:
            continue
        section_body = "\n".join(lines).strip()
        if not section_body:
            continue

        chunks: list[str] = []
        if len(section_body) <= max_chars:
            chunks = [section_body]
        else:
            start = 0
            while start < len(section_body):
                end = min(len(section_body), start + max_chars)
                chunks.append(section_body[start:end].strip())
                if end >= len(section_body):
                    break
                start = max(0, end - overlap)

        for chunk_index, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "professor_name": metadata.professor_name,
                        "detail_url": metadata.detail_url,
                        "markdown_path": metadata.markdown_path,
                        "section": section_title,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return documents


def build_document_ids(documents: Sequence[Document]) -> list[str]:
    ids: list[str] = []
    for document in documents:
        metadata = document.metadata or {}
        seed = "|".join(
            [
                str(metadata.get("professor_name", "")),
                str(metadata.get("detail_url", "")),
                str(metadata.get("section", "")),
                str(metadata.get("chunk_index", "")),
            ]
        )
        ids.append(sha1(seed.encode("utf-8")).hexdigest())
    return ids


def build_markdown_corpus(
    *,
    project_root: Path,
    dossier_entries: Iterable[ProfessorDossierMetadata],
    settings: TutorSettings,
) -> CorpusBuildResult:
    entries = sorted(dossier_entries, key=lambda item: item.professor_name)
    valid_entries = [entry for entry in entries if entry.validation_status == "valid"]
    invalid_entries = [entry for entry in entries if entry.validation_status != "valid"]

    settings.corpus_artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    settings.corpus_index_path.write_text(
        json.dumps([entry.to_dict() for entry in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    documents: list[Document] = []
    manifest_rows: list[dict[str, Any]] = []
    for entry in valid_entries:
        markdown_path = project_root / entry.markdown_path
        markdown_text = read_markdown(markdown_path)
        for document in chunk_professor_markdown(metadata=entry, markdown_text=markdown_text):
            documents.append(document)
            manifest_rows.append(
                {
                    **document.metadata,
                    "text": document.page_content,
                }
            )

    settings.retrieval_manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    if documents:
        vector_store = Chroma(
            collection_name="bit-prof-lab2",
            persist_directory=str(settings.chroma_dir),
            embedding_function=build_embedding_model(settings),
            collection_metadata={"corpus": "bit-professor-markdown"},
        )
        vector_store.add_documents(documents=documents, ids=build_document_ids(documents))

    return CorpusBuildResult(
        valid_professor_count=len(valid_entries),
        invalid_dossier_count=len(invalid_entries),
        chunk_count=len(documents),
        index_path=str(settings.corpus_index_path.relative_to(project_root)),
        retrieval_manifest_path=str(settings.retrieval_manifest_path.relative_to(project_root)),
        vector_dir=str(settings.chroma_dir.relative_to(project_root)),
    )


def rebuild_markdown_corpus(
    *,
    project_root: Path,
    dossier_entries: Iterable[ProfessorDossierMetadata],
    settings: TutorSettings,
) -> CorpusBuildResult:
    return build_markdown_corpus(
        project_root=project_root,
        dossier_entries=dossier_entries,
        settings=settings,
    )


class ProfessorMarkdownRepository:
    def __init__(
        self,
        *,
        project_root: Path,
        settings: TutorSettings,
        dossiers: Sequence[ProfessorDossierMetadata],
    ) -> None:
        self.project_root = project_root
        self.settings = settings
        self._dossiers = list(dossiers)
        self._dossiers_by_name = {item.professor_name: item for item in self._dossiers}

    @classmethod
    def from_index(
        cls,
        *,
        settings: TutorSettings,
        project_root: Path | None = None,
    ) -> "ProfessorMarkdownRepository":
        project_root = project_root or settings.project_root
        if not settings.corpus_index_path.exists():
            raise ValueError(
                f"Corpus index not found at {settings.corpus_index_path}. Run the Lab 2 "
                "ingestion workflow first."
            )
        payload = json.loads(settings.corpus_index_path.read_text(encoding="utf-8"))
        dossiers = [
            ProfessorDossierMetadata(
                **{
                    **row,
                    "validation_checks": row.get("validation_checks", {}),
                }
            )
            for row in payload
        ]
        return cls(project_root=project_root, settings=settings, dossiers=dossiers)

    def list_professors(
        self,
        *,
        limit: int = 10,
        include_invalid: bool = False,
    ) -> list[ProfessorDossierMetadata]:
        rows = [
            item
            for item in sorted(self._dossiers, key=lambda dossier: dossier.professor_name)
            if include_invalid or item.validation_status == "valid"
        ]
        return rows[:limit]

    def resolve_professor(
        self,
        *,
        name_hint: str,
        limit: int = 5,
        include_invalid: bool = False,
    ) -> list[tuple[ProfessorDossierMetadata, int]]:
        needle = name_hint.strip()
        if not needle:
            return []

        resolved: list[tuple[ProfessorDossierMetadata, int]] = []
        for dossier in self._dossiers:
            if dossier.validation_status != "valid" and not include_invalid:
                continue

            haystacks = [dossier.professor_name, *dossier.aliases]
            best_score: int | None = None
            for haystack in haystacks:
                if not haystack:
                    continue
                if names_similar(needle, haystack):
                    candidate_score = 0
                elif compact_name(haystack).startswith(compact_name(needle)):
                    candidate_score = 1
                elif compact_name(needle) in compact_name(haystack):
                    candidate_score = 2
                else:
                    continue
                best_score = candidate_score if best_score is None else min(best_score, candidate_score)
            if best_score is not None:
                resolved.append((dossier, best_score))

        resolved.sort(key=lambda item: (item[1], item[0].professor_name))
        return resolved[:limit]

    def read_professor(
        self,
        *,
        professor_name: str,
        sections: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        dossier = self._dossiers_by_name.get(professor_name)
        if dossier is None:
            raise ValueError(f"Unknown professor: {professor_name}")

        markdown_path = self.project_root / dossier.markdown_path
        markdown_text = read_markdown(markdown_path)
        section_map = parse_markdown_sections(markdown_text)
        requested_sections = normalize_section_request(sections)
        selected_sections = []
        for section_title, lines in section_map.items():
            if requested_sections and section_title.lower() not in requested_sections:
                continue
            selected_sections.append({"title": section_title, "lines": lines})

        return {
            "professor": dossier.to_dict(),
            "sections": selected_sections,
        }

    def dataset_overview(self, *, limit: int = 8) -> dict[str, Any]:
        valid_dossiers = [item for item in self._dossiers if item.validation_status == "valid"]

        interest_counter: Counter[str] = Counter()
        section_counter: Counter[str] = Counter()
        for dossier in valid_dossiers:
            for interest in dossier.research_interests:
                interest_counter[interest] += 1
            for section_name in dossier.available_sections:
                section_counter[section_name] += 1

        return {
            "total_professors": len(valid_dossiers),
            "sample_professors": [item.professor_name for item in valid_dossiers[:limit]],
            "common_research_interests": [
                {"topic": topic, "count": count}
                for topic, count in interest_counter.most_common(limit)
            ],
            "section_coverage": [
                {"section": section, "count": count}
                for section, count in section_counter.most_common(limit)
            ],
        }

    def _load_vector_store(self) -> Chroma:
        if not self.settings.chroma_dir.exists():
            raise ValueError(
                f"Vector index not found at {self.settings.chroma_dir}. Run the Lab 2 "
                "ingestion workflow first."
            )
        return Chroma(
            collection_name="bit-prof-lab2",
            persist_directory=str(self.settings.chroma_dir),
            embedding_function=build_embedding_model(self.settings),
        )

    def search_professors_by_topic(
        self,
        *,
        query: str,
        limit: int = 5,
    ) -> list[TopicProfessorMatch]:
        if not query.strip():
            return []

        vector_store = self._load_vector_store()
        candidate_k = max(limit * 6, 12)
        raw_matches = vector_store.similarity_search_with_score(query, k=candidate_k)
        if not raw_matches:
            return []

        candidate_texts = [document.page_content for document, _ in raw_matches]
        reranker_config = self.settings.require_reranker()
        reranker = JinaStyleReranker(
            base_url=self.settings.rerank_endpoint(),
            api_key=reranker_config["api_key"],
            model=reranker_config["model"],
        )
        reranked = reranker.rerank(
            query=query,
            documents=candidate_texts,
            top_n=min(candidate_k, len(candidate_texts)),
        )

        chunk_matches: list[TopicChunkMatch] = []
        for item in reranked:
            document, raw_vector_score = raw_matches[item.index]
            # Chroma returns backend-specific distance/score values here, so we
            # normalize them into a simple 0..1 proxy for debugging output.
            vector_score = 1.0 / (1.0 + max(float(raw_vector_score), 0.0))
            chunk_matches.append(
                TopicChunkMatch(
                    professor_name=document.metadata["professor_name"],
                    detail_url=document.metadata["detail_url"],
                    markdown_path=document.metadata["markdown_path"],
                    section=document.metadata["section"],
                    snippet=document.page_content[:320],
                    vector_score=float(vector_score),
                    rerank_score=item.relevance_score,
                )
            )

        by_professor: dict[str, list[TopicChunkMatch]] = defaultdict(list)
        for match in chunk_matches:
            by_professor[match.professor_name].append(match)

        aggregated: list[TopicProfessorMatch] = []
        for professor_name, matches in by_professor.items():
            dossier = self._dossiers_by_name.get(professor_name)
            if dossier is None:
                continue
            matches.sort(key=lambda item: item.rerank_score, reverse=True)
            snippets: list[str] = []
            for match in matches:
                snippet = match.snippet
                if snippet not in snippets:
                    snippets.append(snippet)
            sections: list[str] = []
            for match in matches:
                if match.section not in sections:
                    sections.append(match.section)
            aggregated.append(
                TopicProfessorMatch(
                    professor_name=professor_name,
                    detail_url=dossier.detail_url,
                    research_interests=dossier.research_interests,
                    rerank_score=matches[0].rerank_score,
                    matched_sections=sections[:5],
                    snippets=snippets[:3],
                )
            )

        aggregated.sort(key=lambda item: item.rerank_score, reverse=True)
        return aggregated[:limit]


__all__ = [
    "CorpusBuildResult",
    "LocalProfessorMarkdownTask",
    "MarkdownValidation",
    "ProfessorDossierMetadata",
    "ProfessorMarkdownRepository",
    "TopicProfessorMatch",
    "build_markdown_corpus",
    "build_local_professor_markdown_tasks",
    "build_dossier_metadata",
    "build_document_ids",
    "build_embedding_model",
    "clean_quality_line",
    "detect_blocked_artifacts",
    "deslugify_professor_slug",
    "discover_professor_markdown_paths",
    "extract_detail_url",
    "extract_name_candidates",
    "extract_page_count",
    "extract_title",
    "names_similar",
    "parse_markdown_sections",
    "rebuild_markdown_corpus",
    "slugify_name",
    "validate_professor_dossier",
    "validate_professor_markdown",
]
