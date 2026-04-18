from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .ingestion_models import ProfessorListing


RAW_OCR_MARKDOWN_PROMPT = """You perform faithful OCR transcription for one professor poster image.

Return only the visible page content as markdown.
Preserve printed order and text exactly as seen.
Use markdown headings, lists, tables, and line breaks only when they match the visible page.
Do not translate, summarize, explain, normalize, clean up, or invent content.
If text is cut off, include only the visible portion.
If text is unreadable, omit it or mark it inline as [unreadable]. Do not guess.
Do not wrap the answer in code fences.
Do not add metadata, page markers, or commentary that is not visibly present in the image."""


def extract_professor_page_markdown(
    *,
    model: ChatOpenAI,
    image_url: str,
    session: requests.Session | None = None,
) -> str:
    response = (session or requests).get(image_url, timeout=60)
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").split(";", maxsplit=1)[
        0
    ].strip()
    if not content_type.startswith("image/"):
        guessed_type, _ = mimetypes.guess_type(image_url)
        content_type = guessed_type or "image/png"
    data_url = f"data:{content_type};base64,{base64.b64encode(response.content).decode('ascii')}"

    llm_response = model.invoke(
        [
            SystemMessage(content=RAW_OCR_MARKDOWN_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this single professor-poster page into faithful "
                            "markdown. Return page markdown only."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            ),
        ]
    )
    page_markdown = (
        llm_response.text if isinstance(llm_response.text, str) else str(llm_response.text)
    )
    if not page_markdown.strip():
        raise ValueError("OCR returned empty markdown for page image")
    return page_markdown


def extract_professor_poster_markdown(
    *,
    listing: ProfessorListing,
    image_urls: Sequence[str],
    model: ChatOpenAI,
    session: requests.Session | None = None,
) -> str:
    rendered_pages = extract_professor_poster_page_markdowns(
        listing=listing,
        image_urls=image_urls,
        model=model,
        session=session,
    )
    return "\n\n".join(rendered_pages)


def extract_professor_poster_page_markdowns(
    *,
    listing: ProfessorListing,
    image_urls: Sequence[str],
    model: ChatOpenAI,
    session: requests.Session | None = None,
) -> list[str]:
    rendered_pages: list[str] = []
    for page_number, image_url in enumerate(image_urls, start=1):
        try:
            page_markdown = extract_professor_page_markdown(
                model=model,
                image_url=image_url,
                session=session,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Raw markdown OCR failed for {listing.name} page {page_number}: {exc}"
            ) from exc
        rendered_pages.append(page_markdown)
    return rendered_pages


__all__ = [
    "RAW_OCR_MARKDOWN_PROMPT",
    "extract_professor_page_markdown",
    "extract_professor_poster_page_markdowns",
    "extract_professor_poster_markdown",
]
