from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from .config import TutorSettings
from .ingestion_models import OCRPage, ProfessorDossier, ProfessorListing
from .model_factory import build_model


SYNTHESIS_SYSTEM_PROMPT = """You reorganize OCR text into one professor dossier.

You only see listing metadata and OCR text. You do not see images.
Use only text present in the input.
Do not use outside knowledge.
Do not invent, infer, translate, normalize, repair, or expand facts.
If a field is not present in the OCR text, leave it empty.
Preserve bilingual strings exactly as they appear in OCR.
For each section, keep the original printed heading in heading_original when present.
Stitch cross-page continuations by following page order and OCR block flow.
Basic Information should contain only OCR-supported identity, title, affiliation, and contact lines.
Warnings may include issues like empty OCR pages, but do not fabricate missing content.
Return only the structured schema."""


def synthesize_professor_dossier(
    *,
    listing: ProfessorListing,
    pages: Sequence[OCRPage],
    settings: TutorSettings,
    retry_hint: str | None = None,
) -> ProfessorDossier:
    if not pages:
        raise ValueError(f"No OCR pages provided for {listing.name}")

    prompt = SYNTHESIS_SYSTEM_PROMPT
    if retry_hint:
        prompt = f"{prompt}\n\nAdditional correction hint:\n{retry_hint}"

    structured_model = build_model(settings).with_structured_output(
        ProfessorDossier,
        method="json_schema",
        strict=True,
    )
    dossier = structured_model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=json.dumps(
                    {
                        "listing": listing.to_dict(),
                        "ocr_pages": [page.model_dump(mode="json") for page in pages],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
    )
    return dossier.model_copy(
        update={
            "detail_url": listing.detail_url,
            "source_page_urls": [page.image_url for page in pages],
        }
    )


def build_professor_dossier(
    *,
    listing: ProfessorListing,
    pages: Sequence[OCRPage],
    settings: TutorSettings,
    retry_hint: str | None = None,
) -> ProfessorDossier:
    return synthesize_professor_dossier(
        listing=listing,
        pages=pages,
        settings=settings,
        retry_hint=retry_hint,
    )


__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "build_professor_dossier",
    "synthesize_professor_dossier",
]
