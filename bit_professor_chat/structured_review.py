from __future__ import annotations

import hashlib
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .config import TutorSettings
from .ingestion_models import (
    ProfessorArtifactMetadata,
    ProfessorListing,
    ProfessorOCRBundleResult,
    StructuredAcademicServiceRoleRecord,
    StructuredAwardRecord,
    StructuredEducationExperienceRecord,
    StructuredEmploymentExperienceRecord,
    StructuredOrganizationRecord,
    StructuredProfessorRecord,
    StructuredProfessorReview,
    StructuredPublicationRecord,
    StructuredResearchTopicRecord,
)
from .model_factory import build_model


STRUCTURED_REVIEW_SYSTEM_PROMPT = """You convert OCR markdown into a structured professor review draft.

Use only OCR-supported facts from the input.
Do not use outside knowledge.
Do not invent, normalize, repair, or infer unsupported details.
If a field is not explicitly supported by the OCR markdown, leave it empty.
Preserve bilingual strings when they appear in the OCR.
Merge obvious cross-page continuations when the OCR markdown clearly shows one item split across pages.
The OCR in this dataset usually follows a professor-profile pattern. Prefer extracting:
- professor identity and contact information
- current affiliations and organizations
- research interests
- education experiences
- employment experiences
- academic service roles
- awards
- publications

Populate affiliation_organization_names only when the OCR explicitly supports that the professor is affiliated with the organization now.
Map facts into the provided schema only when the OCR supports them.
Return only the structured schema.
"""

WHITESPACE_PATTERN = re.compile(r"\s+")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_CLEAN_PATTERN = re.compile(r"[^\d+()-]+")
PAREN_CONTENT_PATTERN = re.compile(r"\(([^()]+)\)")
CHINESE_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
LATIN_SEGMENT_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9&.,'’/ -]*[A-Za-z0-9)]")
TOPIC_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
ALIAS_KEY_PATTERN = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


def _normalize_whitespace(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = _normalize_whitespace(value)
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _normalize_topic_name(value: str) -> str:
    return TOPIC_NORMALIZE_PATTERN.sub("-", value.lower()).strip("-")


def _normalize_alias_key(value: str) -> str:
    return ALIAS_KEY_PATTERN.sub("", value.lower())


def _clean_phone(value: str) -> str:
    return PHONE_CLEAN_PATTERN.sub("", value).strip()


def _strip_parenthetical(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", value).strip()


def _expand_aliases(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        cleaned = _normalize_whitespace(value)
        if not cleaned:
            continue
        expanded.append(cleaned)
        stripped = _strip_parenthetical(cleaned)
        if stripped and stripped != cleaned:
            expanded.append(stripped)
        for match in PAREN_CONTENT_PATTERN.finditer(cleaned):
            expanded.append(match.group(1))
        expanded.extend(CHINESE_SEGMENT_PATTERN.findall(cleaned))
        expanded.extend(
            segment.strip()
            for segment in LATIN_SEGMENT_PATTERN.findall(cleaned)
            if segment.strip()
        )
        expanded.extend(
            segment.strip()
            for segment in re.split(r"[;/|]+", cleaned)
            if segment.strip()
        )
    return _dedupe_strings(expanded)


def _pick_org_type(*values: str) -> str:
    priority = {
        "university": 0,
        "school": 1,
        "company": 2,
        "association": 3,
        "conference": 4,
        "journal": 5,
        "funder": 6,
        "unknown": 7,
    }
    candidates = [value for value in values if value]
    return min(candidates, key=lambda value: priority.get(value, 99), default="unknown")


def _finalize_organizations(
    organizations: list[StructuredOrganizationRecord],
) -> list[StructuredOrganizationRecord]:
    deduped: dict[str, StructuredOrganizationRecord] = {}
    for organization in organizations:
        cleaned_name = _normalize_whitespace(organization.name)
        if not cleaned_name:
            continue
        aliases = _expand_aliases([cleaned_name, *organization.aliases])
        alias_keys = sorted(
            {
                alias_key
                for alias_key in (_normalize_alias_key(alias) for alias in aliases)
                if alias_key
            }
        )
        canonical_key = alias_keys[0] if alias_keys else _normalize_alias_key(cleaned_name)
        if not canonical_key:
            continue
        current = deduped.get(canonical_key)
        if current is not None:
            deduped[canonical_key] = current.model_copy(
                update={
                    "aliases": _dedupe_strings([*current.aliases, *aliases]),
                    "org_type": _pick_org_type(current.org_type, organization.org_type),
                }
            )
            continue
        deduped[canonical_key] = organization.model_copy(
            update={
                "org_id": _stable_id("org", canonical_key),
                "name": cleaned_name,
                "aliases": aliases,
                "org_type": _pick_org_type(organization.org_type),
            }
        )
    return list(deduped.values())


def _finalize_topics(
    topics: list[StructuredResearchTopicRecord],
) -> list[StructuredResearchTopicRecord]:
    finalized: list[StructuredResearchTopicRecord] = []
    seen: set[str] = set()
    for topic in topics:
        name = _normalize_whitespace(topic.name)
        if not name:
            continue
        normalized_name = _normalize_topic_name(
            topic.normalized_name or name,
        )
        if not normalized_name or normalized_name in seen:
            continue
        seen.add(normalized_name)
        finalized.append(
            topic.model_copy(
                update={
                    "topic_id": _stable_id("topic", normalized_name),
                    "name": name,
                    "normalized_name": normalized_name,
                    "language": _normalize_whitespace(topic.language),
                }
            )
        )
    return finalized


def _finalize_education_records(
    listing: ProfessorListing,
    records: list[StructuredEducationExperienceRecord],
) -> list[StructuredEducationExperienceRecord]:
    finalized: list[StructuredEducationExperienceRecord] = []
    for index, record in enumerate(records, start=1):
        raw_text = _normalize_whitespace(record.raw_text)
        organization_name_raw = _normalize_whitespace(record.organization_name_raw)
        payload = record.model_copy(
            update={
                "experience_id": _stable_id(
                    "education",
                    listing.detail_url,
                    str(index),
                    raw_text or organization_name_raw or record.degree,
                ),
                "degree": _normalize_whitespace(record.degree),
                "field": _normalize_whitespace(record.field),
                "organization_name_raw": organization_name_raw,
                "start_text": _normalize_whitespace(record.start_text),
                "end_text": _normalize_whitespace(record.end_text),
                "order": index,
                "raw_text": raw_text,
            }
        )
        if payload.degree or payload.field or payload.organization_name_raw or payload.raw_text:
            finalized.append(payload)
    return finalized


def _finalize_employment_records(
    listing: ProfessorListing,
    records: list[StructuredEmploymentExperienceRecord],
) -> list[StructuredEmploymentExperienceRecord]:
    finalized: list[StructuredEmploymentExperienceRecord] = []
    for index, record in enumerate(records, start=1):
        raw_text = _normalize_whitespace(record.raw_text)
        organization_name_raw = _normalize_whitespace(record.organization_name_raw)
        payload = record.model_copy(
            update={
                "experience_id": _stable_id(
                    "employment",
                    listing.detail_url,
                    str(index),
                    raw_text or organization_name_raw or record.role_title,
                ),
                "role_title": _normalize_whitespace(record.role_title),
                "organization_name_raw": organization_name_raw,
                "location": _normalize_whitespace(record.location),
                "start_text": _normalize_whitespace(record.start_text),
                "end_text": _normalize_whitespace(record.end_text),
                "order": index,
                "raw_text": raw_text,
            }
        )
        if payload.role_title or payload.organization_name_raw or payload.raw_text:
            finalized.append(payload)
    return finalized


def _finalize_service_roles(
    listing: ProfessorListing,
    records: list[StructuredAcademicServiceRoleRecord],
) -> list[StructuredAcademicServiceRoleRecord]:
    finalized: list[StructuredAcademicServiceRoleRecord] = []
    for index, record in enumerate(records, start=1):
        raw_text = _normalize_whitespace(record.raw_text)
        organization_name_raw = _normalize_whitespace(record.organization_name_raw)
        payload = record.model_copy(
            update={
                "service_id": _stable_id(
                    "service",
                    listing.detail_url,
                    str(index),
                    raw_text or organization_name_raw or record.role_title,
                ),
                "role_title": _normalize_whitespace(record.role_title),
                "organization_name_raw": organization_name_raw,
                "raw_text": raw_text,
            }
        )
        if payload.role_title or payload.organization_name_raw or payload.raw_text:
            finalized.append(payload)
    return finalized


def _finalize_awards(
    listing: ProfessorListing,
    records: list[StructuredAwardRecord],
) -> list[StructuredAwardRecord]:
    finalized: list[StructuredAwardRecord] = []
    for index, record in enumerate(records, start=1):
        raw_text = _normalize_whitespace(record.raw_text)
        name = _normalize_whitespace(record.name)
        payload = record.model_copy(
            update={
                "award_id": _stable_id(
                    "award",
                    listing.detail_url,
                    str(index),
                    raw_text or name,
                ),
                "name": name,
                "category": _normalize_whitespace(record.category),
                "year": _normalize_whitespace(record.year),
                "granting_org_name_raw": _normalize_whitespace(record.granting_org_name_raw),
                "level": _normalize_whitespace(record.level),
                "raw_text": raw_text,
            }
        )
        if payload.name or payload.raw_text:
            finalized.append(payload)
    return finalized


def _finalize_publications(
    listing: ProfessorListing,
    records: list[StructuredPublicationRecord],
) -> list[StructuredPublicationRecord]:
    finalized: list[StructuredPublicationRecord] = []
    for index, record in enumerate(records, start=1):
        raw_text = _normalize_whitespace(record.raw_text)
        title = _normalize_whitespace(record.title)
        payload = record.model_copy(
            update={
                "publication_id": _stable_id(
                    "publication",
                    listing.detail_url,
                    str(index),
                    raw_text or title,
                ),
                "title": title,
                "authors_raw": _normalize_whitespace(record.authors_raw),
                "year": _normalize_whitespace(record.year),
                "venue": _normalize_whitespace(record.venue),
                "doi_or_isbn": _normalize_whitespace(record.doi_or_isbn),
                "raw_text": raw_text,
            }
        )
        if payload.title or payload.raw_text:
            finalized.append(payload)
    return finalized


def _finalize_professor(
    review: StructuredProfessorReview,
    listing: ProfessorListing,
    source_file: str,
) -> StructuredProfessorRecord:
    aliases = _expand_aliases(
        [review.professor.name_local, review.professor.name_english, *review.professor.aliases]
    )
    emails = _dedupe_strings(
        [match.group(0) for match in EMAIL_PATTERN.finditer(" ".join(review.professor.emails))]
        or review.professor.emails
    )
    phones = _dedupe_strings([_clean_phone(value) for value in review.professor.phones])
    return review.professor.model_copy(
        update={
            "professor_id": _stable_id("professor", listing.detail_url),
            "name_local": _normalize_whitespace(review.professor.name_local),
            "name_english": _normalize_whitespace(review.professor.name_english),
            "aliases": aliases,
            "title": _normalize_whitespace(review.professor.title),
            "emails": emails,
            "phones": phones,
            "biography_text": _normalize_whitespace(review.professor.biography_text),
            "source_file": source_file,
        }
    )


def _infer_affiliations(
    *,
    review: StructuredProfessorReview,
    organizations: list[StructuredOrganizationRecord],
    employment_experiences: list[StructuredEmploymentExperienceRecord],
) -> list[str]:
    explicit = _dedupe_strings(review.affiliation_organization_names)
    if explicit:
        return explicit

    employment_candidates = [
        record.organization_name_raw
        for record in employment_experiences
        if record.is_current and record.organization_name_raw
    ]
    if not employment_candidates:
        employment_candidates = [
            record.organization_name_raw
            for record in employment_experiences
            if record.organization_name_raw
        ]

    affiliations = _dedupe_strings(employment_candidates)
    candidate_org_names = [
        organization.name
        for organization in organizations
        if organization.org_type in {"school", "university", "company", "unknown"}
    ]
    if affiliations and len(candidate_org_names) <= 2:
        affiliations = _dedupe_strings([*affiliations, *candidate_org_names])
    elif not affiliations:
        affiliations = _dedupe_strings(candidate_org_names[:2])
    return affiliations


def finalize_structured_review(
    *,
    review: StructuredProfessorReview,
    listing: ProfessorListing,
    metadata: ProfessorArtifactMetadata,
) -> StructuredProfessorReview:
    organizations = _finalize_organizations(review.organizations)
    research_topics = _finalize_topics(review.research_topics)
    education_experiences = _finalize_education_records(listing, review.education_experiences)
    employment_experiences = _finalize_employment_records(
        listing, review.employment_experiences
    )
    academic_service_roles = _finalize_service_roles(listing, review.academic_service_roles)
    awards = _finalize_awards(listing, review.awards)
    publications = _finalize_publications(listing, review.publications)
    professor = _finalize_professor(review, listing, metadata.source_file)
    affiliation_organization_names = _infer_affiliations(
        review=review,
        organizations=organizations,
        employment_experiences=employment_experiences,
    )
    return review.model_copy(
        update={
            "metadata": metadata,
            "professor": professor,
            "organizations": organizations,
            "affiliation_organization_names": affiliation_organization_names,
            "research_topics": research_topics,
            "education_experiences": education_experiences,
            "employment_experiences": employment_experiences,
            "academic_service_roles": academic_service_roles,
            "awards": awards,
            "publications": publications,
            "warnings": _dedupe_strings(review.warnings),
        }
    )


def build_professor_structured_review(
    *,
    listing: ProfessorListing,
    ocr_bundle_result: ProfessorOCRBundleResult,
    artifact_namespace: str,
    settings: TutorSettings,
) -> StructuredProfessorReview:
    metadata = ProfessorArtifactMetadata(
        name=listing.name,
        detail_url=listing.detail_url,
        slug=ocr_bundle_result.slug,
        page_count=ocr_bundle_result.page_count,
        image_urls=list(ocr_bundle_result.image_urls),
        artifact_namespace=artifact_namespace,
        source_file=ocr_bundle_result.ocr_markdown_path,
    )
    structured_model = build_model(settings).with_structured_output(
        StructuredProfessorReview,
        method="json_schema",
        strict=True,
    )
    review = structured_model.invoke(
        [
            SystemMessage(content=STRUCTURED_REVIEW_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "metadata": metadata.model_dump(mode="json"),
                        "listing": listing.to_dict(),
                        "ocr_markdown": ocr_bundle_result.ocr_markdown_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
    )
    return finalize_structured_review(
        review=review,
        listing=listing,
        metadata=metadata,
    )


__all__ = [
    "STRUCTURED_REVIEW_SYSTEM_PROMPT",
    "StructuredProfessorReview",
    "build_professor_structured_review",
    "finalize_structured_review",
]
