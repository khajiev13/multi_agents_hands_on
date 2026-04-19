from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from .config import TutorSettings
from .ingestion_models import DossierSection, ProfessorDossier, ProfessorListing
from .markdown_corpus import deslugify_professor_slug, slugify_name
from .model_factory import build_model


ExperienceKind = Literal["education", "employment", "academic_service"]
OrganizationType = Literal[
    "school",
    "university",
    "association",
    "conference",
    "journal",
    "funder",
    "company",
    "government",
    "laboratory",
    "unknown",
]
PublicationType = Literal["journal", "conference", "book", "thesis", "report", "unknown"]


class ProfessorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    aliases: list[str] = Field(default_factory=list)
    title: str = ""
    school_name: str = ""
    discipline: str = ""
    biography_text: str = ""
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    websites: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


class OrganizationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    org_type: OrganizationType = "unknown"
    normalized_name: str = ""


class ResearchTopicRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    normalized_name: str = ""


class ExperienceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ExperienceKind = "employment"
    title: str = ""
    degree: str = ""
    field: str = ""
    organization_name: str = ""
    location: str = ""
    start_text: str | int = ""
    end_text: str | int = ""
    is_current: bool = False
    order: int = 1
    raw_text: str = ""


class PublicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    authors_raw: str = ""
    year: str | int = ""
    venue: str = ""
    publication_type: PublicationType = "unknown"
    doi_or_url: str = ""
    raw_text: str = ""
    order: int = 1


class AwardRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    year: str | int = ""
    granting_org_name: str = ""
    level: str = ""
    raw_text: str = ""
    order: int = 1


class StructuredProfessorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    professor: ProfessorRecord
    organizations: list[OrganizationRecord] = Field(default_factory=list)
    research_topics: list[ResearchTopicRecord] = Field(default_factory=list)
    experiences: list[ExperienceRecord] = Field(default_factory=list)
    publications: list[PublicationRecord] = Field(default_factory=list)
    awards: list[AwardRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StructuredProfileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    slug: str
    detail_url: str
    source_ocr_markdown_path: str
    source_page_urls: list[str] = Field(default_factory=list)


class StructuredProfessorProfileArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: StructuredProfileMetadata
    profile: StructuredProfessorProfile


def normalize_lookup_text(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", value.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value).strip())
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _normalize_organization(record: OrganizationRecord) -> OrganizationRecord | None:
    name = _clean_text(record.name)
    if not name:
        return None
    normalized_name = normalize_lookup_text(record.normalized_name or name)
    return record.model_copy(
        update={
            "name": name,
            "org_type": _clean_text(record.org_type or "unknown") or "unknown",
            "normalized_name": normalized_name,
        }
    )


def _normalize_topic(record: ResearchTopicRecord) -> ResearchTopicRecord | None:
    name = _clean_text(record.name)
    if not name:
        return None
    return record.model_copy(
        update={
            "name": name,
            "normalized_name": normalize_lookup_text(record.normalized_name or name),
        }
    )


def _normalize_experience(record: ExperienceRecord, order: int) -> ExperienceRecord | None:
    raw_text = _clean_text(record.raw_text)
    title = _clean_text(record.title)
    degree = _clean_text(record.degree)
    field = _clean_text(record.field)
    organization_name = _clean_text(record.organization_name)
    location = _clean_text(record.location)
    start_text = _clean_text(record.start_text)
    end_text = _clean_text(record.end_text)
    if not any([raw_text, title, degree, field, organization_name, start_text, end_text]):
        return None
    return record.model_copy(
        update={
            "kind": _clean_text(record.kind or "employment") or "employment",
            "title": title,
            "degree": degree,
            "field": field,
            "organization_name": organization_name,
            "location": location,
            "start_text": start_text,
            "end_text": end_text,
            "order": order,
            "raw_text": raw_text,
        }
    )


def _normalize_publication(record: PublicationRecord, order: int) -> PublicationRecord | None:
    title = _clean_text(record.title)
    raw_text = _clean_text(record.raw_text)
    if not title and not raw_text:
        return None
    return record.model_copy(
        update={
            "title": title,
            "authors_raw": _clean_text(record.authors_raw),
            "year": _clean_text(record.year),
            "venue": _clean_text(record.venue),
            "publication_type": _clean_text(record.publication_type or "unknown") or "unknown",
            "doi_or_url": _clean_text(record.doi_or_url),
            "raw_text": raw_text,
            "order": order,
        }
    )


def _normalize_award(record: AwardRecord, order: int) -> AwardRecord | None:
    name = _clean_text(record.name)
    raw_text = _clean_text(record.raw_text)
    if not name and not raw_text:
        return None
    return record.model_copy(
        update={
            "name": name,
            "year": _clean_text(record.year),
            "granting_org_name": _clean_text(record.granting_org_name),
            "level": _clean_text(record.level),
            "raw_text": raw_text,
            "order": order,
        }
    )


def finalize_structured_profile(
    *,
    profile: StructuredProfessorProfile,
    listing: ProfessorListing,
) -> StructuredProfessorProfile:
    professor = profile.professor.model_copy(
        update={
            "name": _clean_text(profile.professor.name) or listing.name,
            "aliases": _dedupe_strings(profile.professor.aliases),
            "title": _clean_text(profile.professor.title),
            "school_name": _clean_text(profile.professor.school_name),
            "discipline": _clean_text(profile.professor.discipline),
            "biography_text": _clean_text(profile.professor.biography_text),
            "emails": _dedupe_strings(profile.professor.emails),
            "phones": _dedupe_strings(profile.professor.phones),
            "websites": _dedupe_strings(profile.professor.websites),
            "locations": _dedupe_strings(profile.professor.locations),
        }
    )

    aliases = professor.aliases
    if professor.name and professor.name not in aliases:
        aliases = [professor.name, *aliases]
    if listing.name not in aliases:
        aliases.append(listing.name)
    professor = professor.model_copy(update={"aliases": _dedupe_strings(aliases)})

    organizations: list[OrganizationRecord] = []
    seen_organizations: set[str] = set()
    for record in profile.organizations:
        normalized = _normalize_organization(record)
        if normalized is None or normalized.normalized_name in seen_organizations:
            continue
        seen_organizations.add(normalized.normalized_name)
        organizations.append(normalized)

    research_topics: list[ResearchTopicRecord] = []
    seen_topics: set[str] = set()
    for record in profile.research_topics:
        normalized = _normalize_topic(record)
        if normalized is None or normalized.normalized_name in seen_topics:
            continue
        seen_topics.add(normalized.normalized_name)
        research_topics.append(normalized)

    experiences: list[ExperienceRecord] = []
    for index, record in enumerate(profile.experiences, start=1):
        normalized = _normalize_experience(record, index)
        if normalized is not None:
            experiences.append(normalized)

    publications: list[PublicationRecord] = []
    for index, record in enumerate(profile.publications, start=1):
        normalized = _normalize_publication(record, index)
        if normalized is not None:
            publications.append(normalized)

    awards: list[AwardRecord] = []
    for index, record in enumerate(profile.awards, start=1):
        normalized = _normalize_award(record, index)
        if normalized is not None:
            awards.append(normalized)

    warnings = _dedupe_strings(profile.warnings)
    return profile.model_copy(
        update={
            "professor": professor,
            "organizations": organizations,
            "research_topics": research_topics,
            "experiences": experiences,
            "publications": publications,
            "awards": awards,
            "warnings": warnings,
        }
    )


def _append_labeled_line(lines: list[str], label: str, values: Sequence[str]) -> None:
    cleaned_values = _dedupe_strings(values)
    if cleaned_values:
        lines.append(f"{label}: {', '.join(cleaned_values)}")


def _format_time_range(start_text: str, end_text: str, is_current: bool) -> str:
    start = _clean_text(start_text)
    end = _clean_text(end_text)
    if is_current and not end:
        end = "Present"
    if start and end:
        return f"{start} - {end}"
    return start or end


def _format_experience(record: ExperienceRecord) -> str:
    title = _clean_text(record.title)
    degree = _clean_text(record.degree)
    field = _clean_text(record.field)
    organization = _clean_text(record.organization_name)
    location = _clean_text(record.location)
    time_range = _format_time_range(record.start_text, record.end_text, record.is_current)
    raw_text = _clean_text(record.raw_text)

    parts: list[str] = []
    if record.kind == "education":
        if degree and field:
            parts.append(f"{degree} in {field}")
        elif degree:
            parts.append(degree)
        elif field:
            parts.append(field)
    else:
        if title:
            parts.append(title)
        if degree:
            parts.append(degree)
        if field:
            parts.append(field)

    if organization:
        parts.append(organization)
    if location:
        parts.append(location)

    body = ", ".join(part for part in parts if part)
    if time_range and body:
        return f"{time_range}: {body}"
    if body:
        return body
    if time_range and raw_text:
        return f"{time_range}: {raw_text}"
    return raw_text or time_range


def _format_publication(record: PublicationRecord) -> str:
    parts: list[str] = []
    if record.title:
        parts.append(_clean_text(record.title))
    if record.authors_raw:
        parts.append(f"Authors: {_clean_text(record.authors_raw)}")
    if record.venue:
        parts.append(f"Venue: {_clean_text(record.venue)}")
    if record.year:
        parts.append(f"Year: {_clean_text(record.year)}")
    publication_type = _clean_text(record.publication_type)
    if publication_type and publication_type != "unknown":
        parts.append(f"Type: {publication_type}")
    if record.doi_or_url:
        parts.append(_clean_text(record.doi_or_url))
    return "; ".join(part for part in parts if part) or _clean_text(record.raw_text)


def _format_award(record: AwardRecord) -> str:
    parts: list[str] = []
    if record.name:
        parts.append(_clean_text(record.name))
    if record.year:
        parts.append(f"Year: {_clean_text(record.year)}")
    if record.granting_org_name:
        parts.append(f"Granting Organization: {_clean_text(record.granting_org_name)}")
    if record.level:
        parts.append(f"Level: {_clean_text(record.level)}")
    return "; ".join(part for part in parts if part) or _clean_text(record.raw_text)


def _build_basic_information_section(profile: StructuredProfessorProfile) -> DossierSection:
    professor = profile.professor
    lines: list[str] = []
    _append_labeled_line(lines, "Name", [professor.name])
    alias_values = [
        alias for alias in professor.aliases if _clean_text(alias) != _clean_text(professor.name)
    ]
    _append_labeled_line(lines, "Aliases", alias_values)
    _append_labeled_line(lines, "Title", [professor.title])
    _append_labeled_line(lines, "School", [professor.school_name])
    _append_labeled_line(lines, "Discipline", [professor.discipline])
    _append_labeled_line(lines, "Email", professor.emails)
    _append_labeled_line(lines, "Phone", professor.phones)
    _append_labeled_line(lines, "Website", professor.websites)
    _append_labeled_line(lines, "Location", professor.locations)
    return DossierSection(heading="Basic Information", bullets=lines)


def _build_dossier_sections(profile: StructuredProfessorProfile) -> list[DossierSection]:
    sections: list[DossierSection] = []

    topic_lines = _dedupe_strings(topic.name for topic in profile.research_topics)
    if topic_lines:
        sections.append(DossierSection(heading="Research Interests", bullets=topic_lines))

    biography_text = _clean_text(profile.professor.biography_text)
    if biography_text:
        sections.append(DossierSection(heading="Biography", bullets=[biography_text]))

    education_lines = _dedupe_strings(
        _format_experience(record) for record in profile.experiences if record.kind == "education"
    )
    if education_lines:
        sections.append(DossierSection(heading="Education", bullets=education_lines))

    employment_lines = _dedupe_strings(
        _format_experience(record) for record in profile.experiences if record.kind == "employment"
    )
    if employment_lines:
        sections.append(DossierSection(heading="Work Experience", bullets=employment_lines))

    service_lines = _dedupe_strings(
        _format_experience(record)
        for record in profile.experiences
        if record.kind == "academic_service"
    )
    if service_lines:
        sections.append(
            DossierSection(
                heading="Academic Service and Memberships",
                bullets=service_lines,
            )
        )

    publication_lines = _dedupe_strings(_format_publication(record) for record in profile.publications)
    if publication_lines:
        sections.append(DossierSection(heading="Publications", bullets=publication_lines))

    award_lines = _dedupe_strings(_format_award(record) for record in profile.awards)
    if award_lines:
        sections.append(DossierSection(heading="Awards", bullets=award_lines))

    return sections


def structured_profile_to_dossier(
    *,
    profile: StructuredProfessorProfile,
    listing: ProfessorListing,
    source_page_urls: Sequence[str] | None = None,
) -> ProfessorDossier:
    title_slug = slugify_name(listing.name)
    title = (
        deslugify_professor_slug(title_slug)
        if title_slug
        else _clean_text(listing.name) or _clean_text(profile.professor.name)
    )
    return ProfessorDossier(
        title=title,
        detail_url=listing.detail_url,
        source_page_urls=_dedupe_strings(source_page_urls or []),
        basic_information=_build_basic_information_section(profile),
        sections=_build_dossier_sections(profile),
        uncertain_lines=[],
        warnings=_dedupe_strings(profile.warnings),
    )


def structured_profile_artifact_to_dossier(
    artifact: StructuredProfessorProfileArtifact,
) -> ProfessorDossier:
    return structured_profile_to_dossier(
        profile=artifact.profile,
        listing=ProfessorListing(
            name=artifact.metadata.name,
            detail_url=artifact.metadata.detail_url,
        ),
        source_page_urls=artifact.metadata.source_page_urls,
    )


_WORKED_EXAMPLE = json.dumps(
    {
        "ocr_excerpt": {
            "listing": {
                "name": "Example Professor",
                "detail_url": "https://example.edu/professor",
            },
            "ocr_markdown": (
                "姓名(Name): Example Professor - 职称(Title): Associate Professor\n"
                "学院(School): School of Computer Science - 专业(Discipline): Software Engineering\n"
                "教育背景(Educational Background): 2010-2014, Example University, Ph.D. in Computer Science\n"
                "工作经历(Working Experience): 2020-Present, Associate Professor, Example University\n"
                "发表文章(Publications): Please list 5 representative publications\n"
                "邮箱(Email): example@bit.edu.cn"
            ),
        },
        "output": {
            "professor": {
                "name": "Example Professor",
                "aliases": ["Example Professor"],
                "title": "Associate Professor",
                "school_name": "School of Computer Science",
                "discipline": "Software Engineering",
                "biography_text": "",
                "emails": ["example@bit.edu.cn"],
                "phones": [],
                "websites": [],
                "locations": [],
            },
            "organizations": [
                {
                    "name": "School of Computer Science",
                    "org_type": "school",
                    "normalized_name": "school of computer science",
                },
                {
                    "name": "Example University",
                    "org_type": "university",
                    "normalized_name": "example university",
                },
            ],
            "research_topics": [],
            "experiences": [
                {
                    "kind": "education",
                    "title": "",
                    "degree": "Ph.D.",
                    "field": "Computer Science",
                    "organization_name": "Example University",
                    "location": "",
                    "start_text": "2010",
                    "end_text": "2014",
                    "is_current": False,
                    "order": 1,
                    "raw_text": "2010-2014, Example University, Ph.D. in Computer Science",
                },
                {
                    "kind": "employment",
                    "title": "Associate Professor",
                    "degree": "",
                    "field": "",
                    "organization_name": "Example University",
                    "location": "",
                    "start_text": "2020",
                    "end_text": "Present",
                    "is_current": True,
                    "order": 2,
                    "raw_text": "2020-Present, Associate Professor, Example University",
                },
            ],
            "publications": [],
            "awards": [],
            "warnings": [
                "ignored the instruction-only publication heading because it did not list concrete items"
            ],
        },
    },
    ensure_ascii=False,
)


STRUCTURED_PROFILE_SYSTEM_PROMPT = (
    "You extract a typed professor profile from OCR markdown for a Beijing Institute of "
    "Technology professor page.\n\n"
    "Return one JSON object only. It must match the structured schema exactly.\n\n"
    "Non-negotiable rules:\n"
    "- Use only facts explicitly supported by the OCR markdown or the provided listing metadata.\n"
    "- Never infer missing years, titles, organizations, degrees, venues, contact details, or current status.\n"
    "- Never reconstruct or complete a partial DOI, URL, identifier, or code from pattern guesses.\n"
    "- If a field is unsupported, return an empty string, false, or an empty list.\n"
    "- Deduplicate repeated facts.\n"
    "- Ignore layout junk, form instructions, OCR noise, and empty headings unless they contain a concrete professor fact.\n"
    "- Prefer the clearest canonical value when the same fact appears in both Chinese and English.\n"
    "- Preserve ambiguity in warnings instead of normalizing it away.\n\n"
    "Section mapping:\n"
    "- biography or personal profile text -> professor.biography_text\n"
    "- school / discipline / current affiliation -> professor.school_name, professor.discipline, organizations\n"
    "- research interests -> research_topics\n"
    "- educational background -> experiences with kind='education'\n"
    "- working experience -> experiences with kind='employment'\n"
    "- part-time academic jobs / editor / chair / committee / federation roles -> experiences with kind='academic_service'\n"
    "- awards / honors -> awards\n"
    "- publications / representative papers / books -> publications\n\n"
    "Critical distinction rules for experiences:\n"
    "- Education items may populate degree and field, but title must stay empty unless the education line itself explicitly states a title.\n"
    "- Employment items may populate title, but degree must stay empty unless the employment line itself explicitly states a degree.\n"
    "- Academic service items should use title for the role and organization_name for the host organization when available.\n"
    "- Never copy a title from a neighboring section into an education record.\n"
    "- Never create an experience item from a section heading alone.\n\n"
    "Normalization rules:\n"
    "- organizations.org_type must be one of school, university, association, conference, journal, funder, company, government, laboratory, unknown.\n"
    "- publications.publication_type must be one of journal, conference, book, thesis, report, unknown.\n"
    "- order starts at 1 and follows the source order within each list.\n"
    "- start_text and end_text should stay close to the source wording instead of forcing a normalized date.\n"
    "- is_current is true only when the source clearly says present, current, now, or an equivalent.\n\n"
    "Top-level schema:\n"
    "- professor: {name, aliases, title, school_name, discipline, biography_text, emails, phones, websites, locations}\n"
    "- organizations: [{name, org_type, normalized_name}]\n"
    "- research_topics: [{name, normalized_name}]\n"
    "- experiences: [{kind, title, degree, field, organization_name, location, start_text, end_text, is_current, order, raw_text}]\n"
    "- publications: [{title, authors_raw, year, venue, publication_type, doi_or_url, raw_text, order}]\n"
    "- awards: [{name, year, granting_org_name, level, raw_text, order}]\n"
    "- warnings: [string]\n\n"
    "Compact worked example:\n"
    f"{_WORKED_EXAMPLE}\n"
)


def extract_structured_professor_profile(
    *,
    listing: ProfessorListing,
    ocr_markdown: str,
    settings: TutorSettings,
) -> StructuredProfessorProfile:
    if not ocr_markdown.strip():
        raise ValueError(f"OCR markdown is empty for {listing.name}")

    structured_model = build_model(settings).with_structured_output(
        StructuredProfessorProfile,
        method="json_schema",
        strict=True,
    )
    profile = structured_model.invoke(
        [
            SystemMessage(content=STRUCTURED_PROFILE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Listing metadata:\n"
                    f"- name: {listing.name}\n"
                    f"- detail_url: {listing.detail_url}\n"
                    f"- slug: {slugify_name(listing.name)}\n\n"
                    "OCR markdown:\n"
                    f"{ocr_markdown}"
                )
            ),
        ]
    )
    return finalize_structured_profile(profile=profile, listing=listing)


__all__ = [
    "AwardRecord",
    "ExperienceRecord",
    "OrganizationRecord",
    "ProfessorRecord",
    "PublicationRecord",
    "ResearchTopicRecord",
    "StructuredProfessorProfile",
    "StructuredProfessorProfileArtifact",
    "StructuredProfileMetadata",
    "STRUCTURED_PROFILE_SYSTEM_PROMPT",
    "extract_structured_professor_profile",
    "finalize_structured_profile",
    "normalize_lookup_text",
    "structured_profile_artifact_to_dossier",
    "structured_profile_to_dossier",
]
