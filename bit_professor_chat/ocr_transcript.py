from __future__ import annotations

import base64
import mimetypes
import re
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

MARKDOWN_FENCE_LINE_PATTERN = re.compile(r"^\s*```(?:markdown|md)?\s*$", re.IGNORECASE)
MARKDOWN_TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
MARKDOWN_TABLE_SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{3,}:?$")
INLINE_PIPE_SEPARATOR_PATTERN = re.compile(r"\s+\|\s+")


def clean_ocr_markdown(markdown_text: str) -> str:
    normalized_lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: list[str] = []
    table_block: list[str] = []

    def flush_table_block() -> None:
        nonlocal table_block
        if not table_block:
            return
        parsed_rows = [
            [cell.strip() for cell in row.strip().strip("|").split("|")]
            for row in table_block
        ]
        has_separator_row = any(
            nonempty_cells
            and all(
                MARKDOWN_TABLE_SEPARATOR_CELL_PATTERN.fullmatch(cell)
                for cell in nonempty_cells
            )
            for nonempty_cells in ([cell for cell in row if cell] for row in parsed_rows)
        )
        if has_separator_row:
            for row in parsed_rows:
                nonempty_cells = [cell for cell in row if cell]
                if not nonempty_cells:
                    continue
                if all(
                    MARKDOWN_TABLE_SEPARATOR_CELL_PATTERN.fullmatch(cell)
                    for cell in nonempty_cells
                ):
                    continue
                cleaned_lines.append(" - ".join(nonempty_cells))
        else:
            cleaned_lines.extend(table_block)
        table_block = []

    for line in normalized_lines:
        if MARKDOWN_FENCE_LINE_PATTERN.match(line):
            continue
        if MARKDOWN_TABLE_LINE_PATTERN.match(line):
            table_block.append(line)
            continue
        flush_table_block()
        cleaned_lines.append(line)

    flush_table_block()
    return "\n".join(
        INLINE_PIPE_SEPARATOR_PATTERN.sub(" - ", line) for line in cleaned_lines
    ).strip()


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
    page_markdown = clean_ocr_markdown(page_markdown)
    if not page_markdown:
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
    "clean_ocr_markdown",
    "extract_professor_page_markdown",
    "extract_professor_poster_page_markdowns",
    "extract_professor_poster_markdown",
]
