from __future__ import annotations

import base64
import mimetypes
import re
from collections.abc import Sequence
from typing import Any

import requests
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from .ingestion_models import OCRPageExtraction, OCRVisibleBlock, ProfessorListing
from .markdown_corpus import names_similar


PLACEHOLDER_VALUES = {
    "none",
    "n/a",
    "na",
    "null",
    "not applicable",
    "not available",
    "no additional content",
    "no additional content on this page",
    "nothing to add",
}
PAGE_PATTERN = re.compile(r"<!-- PAGE (\d+) START -->(.*?)<!-- PAGE \1 END -->", re.S)
NAME_LINE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'\-]{1,60}$")
CHINESE_NAME_PATTERN = re.compile(r"^[\u4e00-\u9fff]{2,6}$")
ADJACENT_MIXED_SCRIPT_PATTERN = re.compile(
    r"(?<=[A-Za-z])[\u4e00-\u9fff]|(?<=[\u4e00-\u9fff])[A-Za-z]"
)
MODEL_ARTIFACT_PATTERN = re.compile(
    r"<\|[^|>]+?\|>|#+\s*|^\}\s*|^\d+text",
)

PAGE_SLICE_OCR_PROMPT = """OCR only. This is one image slice from a professor CV poster.

Preserve all visible text in reading order.
Use markdown headings for visible block titles and bullet lines for block content.
Never use tables, columns, code fences, LaTeX, prose, summaries, or explanations.
Do not infer missing text.
Keep cut-off lines exactly as visible.

Use this structure:
### <visible heading text or "(top identity block)" or "(continuation with no visible heading)">
- <transcribed line>
- <transcribed line>
"""

TOP_BLOCK_OCR_PROMPT = """OCR only. Focus only on the compact top identity/contact block of this poster image.

Ignore lower sections like biography, education, work experience, publications, awards, and service.
Output markdown bullets only, one bullet per visible text line, in reading order.
Do not translate. Do not summarize. Do not normalize labels.
Do not output tables, code fences, LaTeX, or prose.
Keep the exact visible wording and line breaks as closely as possible.
"""

INLINE_BASIC_INFO_LABELS = (
    "title",
    "职称",
    "school",
    "学院",
    "discipline",
    "专业",
    "department",
    "系所",
    "office",
    "办公地址",
)
CONTACT_LABELS = (
    "tel",
    "phone",
    "email",
    "邮箱",
    "电话",
    "联系",
    "联系方式",
    "fax",
    "地址",
    "office",
    "homepage",
    "主页",
)
RESEARCH_LABELS = ("research interests", "research directions", "研究方向", "研究兴趣")
NAME_LABELS = ("name", "姓名", "英文名", "english name", "chinese name", "中文名")
TITLE_SUFFIXES = (
    "associate professor",
    "assistant professor",
    "professor",
    "lecturer",
    "副教授",
    "教授",
    "讲师",
)
INLINE_IDENTITY_FIELD_PATTERN = re.compile(
    "|".join(
        re.escape(label)
        for label in sorted(
            set(INLINE_BASIC_INFO_LABELS + CONTACT_LABELS + RESEARCH_LABELS + NAME_LABELS),
            key=len,
            reverse=True,
        )
    ),
    re.I,
)
SECTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Research Interests", ("research interests", "research directions", "研究方向", "研究兴趣")),
    ("Basic Information", ("basic information", "基本信息")),
    ("Biography", ("biography", "个人简历", "个人简介")),
    ("Education", ("education", "educational background", "教育背景")),
    ("Work Experience", ("work experience", "working experience", "工作经历")),
    (
        "Academic Service and Memberships",
        ("academic service", "memberships", "part-time academic jobs", "学术兼职", "社会兼职"),
    ),
    ("Research Projects", ("research projects", "科研项目")),
    ("Teaching", ("teaching", "教学")),
    ("Publications", ("publications", "发表文章", "发表论文", "论文", "著作")),
    ("Awards", ("awards", "所获荣誉", "荣誉", "获奖")),
    ("Patents", ("patents", "patent", "专利")),
    ("Academic Achievements", ("academic achievements", "学术成果")),
    ("Social Service", ("social service", "社会服务")),
    ("Contact", ("contact", "联系方式")),
)
SECTION_ORDER = [
    "Basic Information",
    "Research Interests",
    "Biography",
    "Education",
    "Work Experience",
    "Research Projects",
    "Teaching",
    "Academic Service and Memberships",
    "Academic Achievements",
    "Publications",
    "Patents",
    "Awards",
    "Social Service",
    "Contact",
]


def text_from_response(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(block.get("text", ""))
            else:
                chunks.append(str(block))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return str(content).strip()


def clean_markdown_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def is_placeholder_markdown_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    candidate = None
    if stripped.startswith(("- ", "* ")):
        candidate = stripped[2:].strip()
    else:
        match = re.match(r"\d+\.\s+(.*)", stripped)
        if match:
            candidate = match.group(1).strip()

    if candidate is None:
        return False

    candidate = re.sub(r"[`*_]+", "", candidate).strip().lower().rstrip(".")
    return candidate in PLACEHOLDER_VALUES or candidate == "<transcribed line>"


def collapse_blank_lines(lines: Sequence[str]) -> list[str]:
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return collapsed


def remove_empty_markdown_sections(text: str) -> str:
    lines = text.splitlines()

    def is_non_content_line(line: str) -> bool:
        stripped = line.strip()
        return not stripped or (stripped.startswith("<!--") and stripped.endswith("-->"))

    def heading_level(line: str) -> int | None:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            return None
        return len(stripped) - len(stripped.lstrip("#"))

    changed = True
    while changed:
        changed = False
        new_lines: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            current_level = heading_level(line)
            if current_level is None:
                new_lines.append(line)
                index += 1
                continue

            end = index + 1
            while end < len(lines):
                next_level = heading_level(lines[end])
                if next_level is not None and next_level <= current_level:
                    break
                end += 1

            body = lines[index + 1 : end]
            if any(not is_non_content_line(entry) for entry in body):
                new_lines.append(line)
                new_lines.extend(body)
            else:
                changed = True
            index = end
        lines = new_lines

    return "\n".join(collapse_blank_lines(lines)).strip()


def cleanup_markdown_artifact(text: str) -> str:
    cleaned = clean_markdown_text(text)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    lines = [line for line in lines if not is_placeholder_markdown_line(line)]
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return ""
    cleaned = remove_empty_markdown_sections(cleaned)
    cleaned = re.sub(
        r"(?ms)^## (Continuations Resolved|Uncertain or Illegible Text)\n(?=(?:\s*<!--|\s*#|\Z))",
        "",
        cleaned,
    ).strip()
    return "\n".join(collapse_blank_lines(cleaned.splitlines())).strip()


def _build_single_image_message(
    *,
    prompt_text: str,
    image_url: str,
) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


def _is_multimodal_download_error(error: BadRequestError) -> bool:
    error_text = str(error).lower()
    return (
        "failed to download multimodal content" in error_text
        or "invalidparameter" in error_text
        or "invalid_parameter_error" in error_text
    )


def _download_image_as_data_url(*, image_url: str, session: requests.Session) -> str:
    response = session.get(image_url, timeout=60)
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").split(";", maxsplit=1)[0].strip()
    if not content_type.startswith("image/"):
        guessed_type, _ = mimetypes.guess_type(image_url)
        content_type = guessed_type or "image/png"

    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _replace_image_url_with_data_url(
    *,
    message_content: Sequence[dict[str, Any]],
    image_url: str,
    session: requests.Session,
) -> list[dict[str, Any]]:
    data_url = _download_image_as_data_url(image_url=image_url, session=session)
    rewritten: list[dict[str, Any]] = []
    for item in message_content:
        if item.get("type") != "image_url":
            rewritten.append(dict(item))
            continue
        image_payload = dict(item.get("image_url", {}))
        if image_payload.get("url") == image_url:
            image_payload["url"] = data_url
        rewritten.append({"type": "image_url", "image_url": image_payload})
    return rewritten


def _invoke_multimodal_message(
    *,
    model: ChatOpenAI,
    message_content: list[dict[str, Any]],
    image_url: str,
    session: requests.Session | None = None,
):
    try:
        return model.invoke([HumanMessage(content=message_content)])
    except BadRequestError as exc:
        if not session or not _is_multimodal_download_error(exc):
            raise

        data_url_content = _replace_image_url_with_data_url(
            message_content=message_content,
            image_url=image_url,
            session=session,
        )
        return model.invoke([HumanMessage(content=data_url_content)])


def extract_professor_page_markdown(
    *,
    model: ChatOpenAI,
    image_url: str,
    session: requests.Session | None = None,
) -> str:
    response = _invoke_multimodal_message(
        model=model,
        message_content=_build_single_image_message(
            prompt_text=PAGE_SLICE_OCR_PROMPT,
            image_url=image_url,
        ),
        image_url=image_url,
        session=session,
    )
    return cleanup_markdown_artifact(text_from_response(response))


def extract_header_identity_lines(
    *,
    model: ChatOpenAI,
    image_url: str,
    session: requests.Session | None = None,
) -> list[str]:
    response = _invoke_multimodal_message(
        model=model,
        message_content=_build_single_image_message(
            prompt_text=TOP_BLOCK_OCR_PROMPT,
            image_url=image_url,
        ),
        image_url=image_url,
        session=session,
    )
    return parse_header_identity_lines(cleanup_markdown_artifact(text_from_response(response)))


def _append_unique(lines: list[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in lines:
        lines.append(cleaned)


def _clean_extracted_line(line: str) -> str:
    cleaned = MODEL_ARTIFACT_PATTERN.sub("", line.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.lower() in {"(header only)", "(section header)", "header only", "section header"}:
        return ""
    return cleaned


def _strip_markdown_formatting(text: str) -> str:
    return re.sub(r"[*_`#]+", "", text).strip()


def _strip_leading_number(line: str) -> str:
    return re.sub(r"^\d+[.)、]\s*", "", line).strip()


def _merge_key_value_lines(lines: Sequence[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = _clean_extracted_line(lines[index])
        if not current:
            index += 1
            continue

        if index + 1 < len(lines):
            following = _clean_extracted_line(lines[index + 1])
            if (
                current.rstrip().endswith((":", "："))
                and following
                and not following.rstrip().endswith((":", "："))
            ):
                merged.append(f"{current} {following}".strip())
                index += 2
                continue

        merged.append(current)
        index += 1
    return merged


def _split_labeled_line(line: str) -> tuple[str, str]:
    normalized = _strip_markdown_formatting(_clean_extracted_line(line))
    if ":" in normalized:
        label, _, value = normalized.partition(":")
        return label.strip(), value.strip()
    if "：" in normalized:
        label, _, value = normalized.partition("：")
        return label.strip(), value.strip()
    return normalized, ""


def _split_numbered_value(value: str) -> list[str]:
    normalized = _strip_markdown_formatting(value)
    matches = [
        match.group(1).strip()
        for match in re.finditer(
            r"(?:^|\s)\d+[.)、]\s*(.+?)(?=(?:\s+\d+[.)、]\s)|$)",
            normalized,
        )
    ]
    if matches:
        return matches
    return [normalized] if normalized else []


def parse_header_identity_lines(markdown_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("- ", "* ")):
            cleaned = _clean_extracted_line(stripped[2:])
            if cleaned:
                lines.append(cleaned)
            continue
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and all(not cell or set(cell) <= {":", "-", " "} for cell in cells):
                continue
            for cell in cells:
                cleaned = _clean_extracted_line(cell)
                if cleaned:
                    lines.append(cleaned)
            continue
        cleaned = _clean_extracted_line(stripped)
        if cleaned:
            lines.append(cleaned)
    return lines


def _looks_like_metadata_table(rows: Sequence[Sequence[str]]) -> bool:
    for row in rows:
        for cell in row:
            lowered = _strip_markdown_formatting(cell).lower()
            if any(label in lowered for label in INLINE_BASIC_INFO_LABELS + CONTACT_LABELS + RESEARCH_LABELS):
                return True
            if CHINESE_NAME_PATTERN.fullmatch(_strip_markdown_formatting(cell)):
                return True
            if NAME_LINE_PATTERN.fullmatch(_strip_markdown_formatting(cell)):
                return True
    return False


def _convert_table_chunk(table_lines: Sequence[str]) -> list[str]:
    rows: list[list[str]] = []
    for raw in table_lines:
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if cells and all(not cell or set(cell) <= {":", "-", " "} for cell in cells):
            continue
        meaningful = [_clean_extracted_line(cell) for cell in cells if _clean_extracted_line(cell)]
        if meaningful:
            rows.append(meaningful)

    if not rows:
        return []

    converted: list[str] = []
    if _looks_like_metadata_table(rows):
        converted.append("### (top identity block)")
        for row in rows:
            for cell in row:
                converted.append(f"- {cell}")
        return converted

    for row in rows:
        for cell in row:
            converted.append(cell)
    return converted


def _looks_like_section_heading(line: str) -> bool:
    normalized = _normalize_heading_token(line)
    if not normalized:
        return False
    return any(alias in normalized for _, aliases in SECTION_ALIASES for alias in aliases)


def _normalize_heading_token(heading_text: str) -> str:
    normalized = heading_text.strip().lower()
    normalized = normalized.replace("(continuation with no visible heading)", "")
    normalized = normalized.replace("（continuation with no visible heading）", "")
    normalized = re.sub(r"[：:()（）]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_page_ocr_markdown(*, raw_text: str, page_number: int) -> str:
    cleaned = clean_markdown_text(raw_text)
    source_lines = cleaned.splitlines()
    converted: list[str] = []
    index = 0
    while index < len(source_lines):
        stripped = source_lines[index].strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines: list[str] = []
            while index < len(source_lines):
                candidate = source_lines[index].strip()
                if not (candidate.startswith("|") and candidate.endswith("|")):
                    break
                table_lines.append(source_lines[index])
                index += 1
            converted.extend(_convert_table_chunk(table_lines))
            continue
        if stripped.startswith("```"):
            index += 1
            continue
        converted.append(source_lines[index].rstrip())
        index += 1

    normalized: list[str] = []
    has_any_heading = False
    for raw_line in converted:
        stripped = raw_line.strip()
        if not stripped or stripped == "---" or is_placeholder_markdown_line(stripped):
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if not heading:
                continue
            normalized.append(f"### {heading}")
            has_any_heading = True
            continue
        if stripped.startswith(("- ", "* ")):
            normalized.append(f"- {_clean_extracted_line(stripped[2:])}")
            continue
        if re.match(r"^\d+\.\s+.+", stripped):
            normalized.append(stripped)
            continue
        if _looks_like_section_heading(stripped):
            normalized.append(f"### {stripped}")
            has_any_heading = True
            continue
        if not has_any_heading:
            normalized.append(
                f"### {'(top identity block)' if page_number == 1 else '(continuation with no visible heading)'}"
            )
            normalized.append(f"- {stripped}")
            has_any_heading = True
            continue
        if normalized and normalized[-1].startswith("### "):
            normalized.append(f"- {stripped}")
            continue
        normalized.append(f"- {stripped}")

    return "\n".join(collapse_blank_lines(normalized)).strip()


def _flush_page_block(
    *,
    blocks: list[OCRVisibleBlock],
    heading_text: str | None,
    content_lines: list[str],
    page_number: int,
) -> None:
    if heading_text is None and not content_lines:
        return
    heading = (heading_text or "(continuation with no visible heading)").strip()
    content = [_clean_extracted_line(line) for line in content_lines if _clean_extracted_line(line)]
    if not heading and not content:
        return

    if heading == "(top identity block)":
        blocks.append(OCRVisibleBlock(heading_text=heading, block_role="standalone", content_lines=content))
        return

    if heading == "(continuation with no visible heading)":
        block_role = "continuation_from_previous_page" if page_number > 1 else "standalone"
        blocks.append(OCRVisibleBlock(heading_text=heading, block_role=block_role, content_lines=content))
        return

    if _looks_like_section_heading(heading):
        blocks.append(OCRVisibleBlock(heading_text=heading, block_role="new_section", content_lines=content))
        return

    if page_number > 1:
        blocks.append(
            OCRVisibleBlock(
                heading_text="(continuation with no visible heading)",
                block_role="continuation_from_previous_page",
                content_lines=[heading, *content],
            )
        )
        return

    blocks.append(OCRVisibleBlock(heading_text="(top identity block)", block_role="standalone", content_lines=[heading, *content]))


def page_markdown_to_page_extraction(
    *,
    normalized_markdown: str,
    page_number: int,
    image_url: str,
) -> OCRPageExtraction:
    blocks: list[OCRVisibleBlock] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in normalized_markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            _flush_page_block(
                blocks=blocks,
                heading_text=current_heading,
                content_lines=current_lines,
                page_number=page_number,
            )
            current_heading = stripped[4:].strip()
            current_lines = []
            continue
        if stripped.startswith(("- ", "* ")):
            current_lines.append(stripped[2:].strip())
            continue
        current_lines.append(stripped)

    _flush_page_block(
        blocks=blocks,
        heading_text=current_heading,
        content_lines=current_lines,
        page_number=page_number,
    )

    if not blocks:
        raise ValueError(f"OCR output did not yield any readable blocks for page {page_number}.")

    return OCRPageExtraction(
        page_number=page_number,
        image_url=image_url,
        blocks=blocks,
        uncertain_lines=[],
    )


def render_page_notes(pages: Sequence[OCRPageExtraction]) -> str:
    rendered_pages: list[str] = []
    for page in pages:
        lines = [
            f"<!-- PAGE {page.page_number} START -->",
            "# Page Extraction",
            "## Source",
            f"- page_number: {page.page_number}",
            f"- image_url: {page.image_url}",
            "## Visible Blocks",
        ]
        for index, block in enumerate(page.blocks, start=1):
            lines.extend(
                [
                    f"### Block {index}",
                    f"- heading_text: {block.heading_text}",
                    f"- block_role: {block.block_role}",
                    "- content:",
                ]
            )
            lines.extend(f"  - {line}" for line in block.content_lines)
        if page.uncertain_lines:
            lines.append("## Uncertain or Illegible Text")
            lines.extend(f"- {line}" for line in page.uncertain_lines)
        lines.append(f"<!-- PAGE {page.page_number} END -->")
        rendered_pages.append("\n".join(lines))
    return "\n\n".join(rendered_pages)


def extract_professor_poster_notes(
    *,
    model: ChatOpenAI,
    image_urls: Sequence[str],
    session: requests.Session | None = None,
) -> str:
    if not image_urls:
        raise ValueError("At least one image URL is required for OCR extraction.")

    pages: list[OCRPageExtraction] = []
    for page_number, image_url in enumerate(image_urls, start=1):
        raw_page_markdown = extract_professor_page_markdown(
            model=model,
            image_url=image_url,
            session=session,
        )
        normalized_page_markdown = normalize_page_ocr_markdown(
            raw_text=raw_page_markdown,
            page_number=page_number,
        )
        pages.append(
            page_markdown_to_page_extraction(
                normalized_markdown=normalized_page_markdown,
                page_number=page_number,
                image_url=image_url,
            )
        )

    return cleanup_markdown_artifact(render_page_notes(pages))


def parse_ocr_page_notes(markdown_text: str) -> list[OCRPageExtraction]:
    pages: list[OCRPageExtraction] = []
    for match in PAGE_PATTERN.finditer(markdown_text):
        page_number = int(match.group(1))
        page_body = match.group(2)
        image_url_match = re.search(r"^-\s*image_url:\s*(.+?)\s*$", page_body, re.M)
        image_url = image_url_match.group(1).strip() if image_url_match else ""

        blocks: list[OCRVisibleBlock] = []
        uncertain_lines: list[str] = []
        current_block: dict[str, Any] | None = None
        in_content = False
        in_uncertain = False

        def flush_block() -> None:
            nonlocal current_block
            if current_block is None:
                return
            blocks.append(
                OCRVisibleBlock(
                    heading_text=current_block["heading_text"],
                    block_role=current_block["block_role"],
                    content_lines=current_block["content_lines"],
                )
            )
            current_block = None

        for raw_line in page_body.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("### Block"):
                flush_block()
                current_block = {
                    "heading_text": "",
                    "block_role": "standalone",
                    "content_lines": [],
                }
                in_content = False
                in_uncertain = False
                continue

            if stripped == "## Uncertain or Illegible Text":
                flush_block()
                in_content = False
                in_uncertain = True
                continue

            if stripped.startswith("## "):
                in_content = False
                if stripped != "## Uncertain or Illegible Text":
                    in_uncertain = False
                continue

            if in_uncertain:
                if stripped.startswith(("- ", "* ")):
                    cleaned = _clean_extracted_line(stripped[2:])
                    if cleaned:
                        uncertain_lines.append(cleaned)
                elif stripped:
                    cleaned = _clean_extracted_line(stripped)
                    if cleaned:
                        uncertain_lines.append(cleaned)
                continue

            if current_block is None:
                continue

            if stripped.startswith("- heading_text:"):
                current_block["heading_text"] = stripped.split(":", 1)[1].strip()
                continue

            if stripped.startswith("- block_role:"):
                current_block["block_role"] = stripped.split(":", 1)[1].strip()
                continue

            if stripped == "- content:":
                in_content = True
                continue

            if in_content:
                if stripped.startswith(("- ", "* ")):
                    cleaned = _clean_extracted_line(stripped[2:])
                    if cleaned:
                        current_block["content_lines"].append(cleaned)
                elif stripped:
                    cleaned = _clean_extracted_line(stripped)
                    if cleaned:
                        current_block["content_lines"].append(cleaned)

        flush_block()
        pages.append(
            OCRPageExtraction(
                page_number=page_number,
                image_url=image_url,
                blocks=blocks,
                uncertain_lines=uncertain_lines,
            )
        )

    if not pages:
        raise ValueError("OCR output did not contain any structured page extraction blocks.")

    return pages


def _is_name_candidate(line: str) -> bool:
    candidate = _strip_trailing_title(line.strip())
    if not candidate or candidate.endswith((":", "：")):
        return False
    lowered = candidate.lower()
    if any(label in lowered for label in INLINE_BASIC_INFO_LABELS + CONTACT_LABELS + RESEARCH_LABELS):
        return False
    if CHINESE_NAME_PATTERN.fullmatch(candidate):
        return True
    if not NAME_LINE_PATTERN.fullmatch(candidate):
        return False
    words = candidate.split()
    if not 1 < len(words) <= 3:
        return False
    banned = {"school", "institute", "university", "professor", "assistant", "associate"}
    return not any(word.lower() in banned for word in words)


def _strip_trailing_title(candidate: str) -> str:
    cleaned = candidate.strip()
    lowered = cleaned.lower()
    for suffix in TITLE_SUFFIXES:
        if lowered.endswith(f" {suffix}"):
            return cleaned[: -len(suffix) - 1].strip()
    return cleaned


def _extract_mixed_script_name_candidate(line: str) -> str | None:
    normalized = _strip_trailing_title(line.strip())
    match = re.search(r"([\u4e00-\u9fff]{2,6})\s+([A-Za-z][A-Za-z .'\-]{1,60})$", normalized)
    if not match:
        return None
    english_candidate = _strip_trailing_title(match.group(2).strip())
    return english_candidate or match.group(1).strip()


def _label_contains_name_field(label: str) -> bool:
    normalized = re.sub(r"[\s()（）]+", " ", label.lower()).strip()
    return any(name_label in normalized for name_label in NAME_LABELS)


def _trim_inline_field_value(value: str) -> str:
    match = INLINE_IDENTITY_FIELD_PATTERN.search(value)
    if match and match.start() > 0:
        return value[: match.start()].strip()
    return value.strip()


def extract_identity_candidates_from_lines(lines: Sequence[str]) -> list[str]:
    candidates: list[str] = []
    saw_field_label = False
    for line in _merge_key_value_lines(lines):
        normalized = _strip_markdown_formatting(line)
        lowered = normalized.lower()
        label, value = _split_labeled_line(line)
        if _label_contains_name_field(label) and value:
            _append_unique(candidates, _trim_inline_field_value(value))
            continue
        mixed_script_candidate = _extract_mixed_script_name_candidate(normalized)
        if mixed_script_candidate:
            _append_unique(candidates, mixed_script_candidate)
            continue
        stripped_title_candidate = _strip_trailing_title(normalized)
        if _is_name_candidate(stripped_title_candidate) and not saw_field_label:
            _append_unique(candidates, stripped_title_candidate)
            continue
        if any(token in lowered for token in INLINE_BASIC_INFO_LABELS + CONTACT_LABELS + RESEARCH_LABELS):
            saw_field_label = True
    return candidates


def extract_identity_candidates(pages: Sequence[OCRPageExtraction]) -> list[str]:
    candidates: list[str] = []
    for page in pages[:2]:
        for block in page.blocks:
            for candidate in extract_identity_candidates_from_lines(block.content_lines):
                _append_unique(candidates, candidate)
    return candidates


def choose_markdown_title(*, expected_name: str, identity_candidates: Sequence[str]) -> str:
    if not identity_candidates:
        raise ValueError("OCR extraction did not yield any readable professor name candidates.")

    for candidate in identity_candidates:
        if names_similar(expected_name, candidate):
            return candidate

    for candidate in identity_candidates:
        if NAME_LINE_PATTERN.fullmatch(candidate):
            return candidate

    return identity_candidates[0]


def _is_contact_line(line: str) -> bool:
    lowered = _strip_markdown_formatting(line).lower()
    return any(label in lowered for label in CONTACT_LABELS)


def _is_research_heading_line(line: str) -> bool:
    lowered = _strip_markdown_formatting(line).lower()
    return any(label in lowered for label in RESEARCH_LABELS)


def _is_basic_info_heading_line(line: str) -> bool:
    lowered = _strip_markdown_formatting(line).lower()
    return any(label in lowered for label in INLINE_BASIC_INFO_LABELS)


def _extract_inline_sections(lines: Sequence[str]) -> dict[str, list[str]]:
    merged_lines = _merge_key_value_lines(lines)
    sections: dict[str, list[str]] = {
        "Basic Information": [],
        "Research Interests": [],
        "Contact": [],
    }
    mode = "Basic Information"

    for raw_line in merged_lines:
        line = _clean_extracted_line(raw_line)
        if not line:
            continue

        label, value = _split_labeled_line(line)

        if _is_research_heading_line(line):
            mode = "Research Interests"
            for item in _split_numbered_value(value):
                _append_unique(sections["Research Interests"], _strip_leading_number(item))
            continue

        if _is_contact_line(line):
            mode = "Contact"
            _append_unique(sections["Contact"], f"{label}: {value}".strip(": ") if value else label)
            continue

        if mode == "Research Interests":
            if _is_basic_info_heading_line(line):
                mode = "Basic Information"
                _append_unique(
                    sections["Basic Information"],
                    f"{label}: {value}".strip(": ") if value else label,
                )
                continue
            if _is_contact_line(line):
                mode = "Contact"
                _append_unique(
                    sections["Contact"],
                    f"{label}: {value}".strip(": ") if value else label,
                )
                continue
            for item in _split_numbered_value(value or line):
                _append_unique(sections["Research Interests"], _strip_leading_number(item))
            continue

        _append_unique(
            sections["Basic Information"],
            f"{label}: {value}".strip(": ") if value else _strip_markdown_formatting(line),
        )

    return {key: value for key, value in sections.items() if value}


def _canonical_section_name(heading_text: str) -> str | None:
    stripped = heading_text.strip()
    lowered = stripped.lower()
    if (
        not stripped
        or "continuation with no visible heading" in lowered
        or "top identity block" in lowered
    ):
        return None

    normalized = _normalize_heading_token(stripped)
    for section_name, aliases in SECTION_ALIASES:
        if any(alias in normalized for alias in aliases):
            return section_name

    return stripped.rstrip("：:")


def _append_section_lines(
    *,
    section_lines: dict[str, list[str]],
    section_name: str,
    lines: Sequence[str],
    block_role: str,
) -> None:
    target = section_lines.setdefault(section_name, [])
    pending = [line for line in lines if line]
    if block_role == "continuation_from_previous_page" and target and pending:
        first_line = pending[0]
        if not re.match(r"^\d+[.)、]\s+", first_line) and not _is_basic_info_heading_line(first_line):
            target[-1] = f"{target[-1].rstrip()} {first_line}".strip()
            pending = pending[1:]

    for line in pending:
        _append_unique(target, line)


def _filter_basic_information_noise(items: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    for item in items:
        normalized = _strip_markdown_formatting(item)
        if ":" in normalized or "：" in normalized:
            filtered.append(item)
            continue
        if ADJACENT_MIXED_SCRIPT_PATTERN.search(normalized):
            continue
        filtered.append(item)
    return filtered or list(items)


def needs_top_block_fallback(
    *,
    pages: Sequence[OCRPageExtraction],
    expected_name: str,
) -> bool:
    identity_candidates = extract_identity_candidates(pages)
    first_page_lines: list[str] = []
    if pages:
        for block in pages[0].blocks:
            first_page_lines.extend(block.content_lines)
    inline_sections = _extract_inline_sections(first_page_lines)
    has_expected_name = any(names_similar(expected_name, candidate) for candidate in identity_candidates)
    has_research_lines = bool(inline_sections.get("Research Interests"))
    return not has_expected_name or not has_research_lines


def build_professor_markdown_from_page_notes(
    *,
    listing: ProfessorListing,
    image_urls: Sequence[str],
    page_notes_markdown: str,
    supplemental_header_lines: Sequence[str] | None = None,
) -> str:
    pages = parse_ocr_page_notes(page_notes_markdown)
    identity_candidates = extract_identity_candidates(pages)
    for candidate in extract_identity_candidates_from_lines(supplemental_header_lines or []):
        _append_unique(identity_candidates, candidate)
    if not identity_candidates:
        _append_unique(identity_candidates, listing.name)
    title = choose_markdown_title(
        expected_name=listing.name,
        identity_candidates=identity_candidates,
    )

    if not any(names_similar(listing.name, candidate) for candidate in identity_candidates):
        raise ValueError(
            f"Extracted professor identity does not match listing {listing.name}. "
            f"Found: {', '.join(identity_candidates)}"
        )

    section_lines: dict[str, list[str]] = {}
    extra_sections: list[str] = []
    uncertain_lines: list[str] = []
    last_section_name: str | None = None

    for inline_section, inline_lines in _extract_inline_sections(supplemental_header_lines or []).items():
        _append_section_lines(
            section_lines=section_lines,
            section_name=inline_section,
            lines=inline_lines,
            block_role="standalone",
        )
        last_section_name = inline_section

    for page in pages:
        for uncertain_line in page.uncertain_lines:
            _append_unique(uncertain_lines, uncertain_line)

        for block in page.blocks:
            merged_lines = _merge_key_value_lines(block.content_lines)
            if not merged_lines:
                continue

            section_name = _canonical_section_name(block.heading_text)
            if section_name is None:
                if (
                    block.block_role in {"continuation_from_previous_page", "continuation_to_next_page"}
                    and last_section_name is not None
                ):
                    _append_section_lines(
                        section_lines=section_lines,
                        section_name=last_section_name,
                        lines=merged_lines,
                        block_role=block.block_role,
                    )
                    continue
                for inline_section, inline_lines in _extract_inline_sections(merged_lines).items():
                    _append_section_lines(
                        section_lines=section_lines,
                        section_name=inline_section,
                        lines=inline_lines,
                        block_role=block.block_role,
                    )
                    last_section_name = inline_section
                continue

            rendered_lines = (
                [_strip_leading_number(line) for line in merged_lines]
                if section_name == "Research Interests"
                else list(merged_lines)
            )
            _append_section_lines(
                section_lines=section_lines,
                section_name=section_name,
                lines=rendered_lines,
                block_role=block.block_role,
            )
            last_section_name = section_name
            if section_name not in SECTION_ORDER and section_name not in extra_sections:
                extra_sections.append(section_name)

    lines: list[str] = [
        f"# {title}",
        f"- detail_url: {listing.detail_url}",
        f"- page_count: {len(image_urls)}",
        "",
    ]

    for section_name in [*SECTION_ORDER, *extra_sections]:
        items = section_lines.get(section_name, [])
        if section_name == "Basic Information":
            items = _filter_basic_information_noise(items)
        if not items:
            continue
        lines.append(f"## {section_name}")
        lines.extend(f"- {item}" for item in items)
        lines.append("")

    lines.append("## Source Pages")
    lines.extend(f"- page {index}: {url}" for index, url in enumerate(image_urls, start=1))
    if uncertain_lines:
        lines.extend(["", "## Uncertain or Illegible Text"])
        lines.extend(f"- {item}" for item in uncertain_lines)

    return "\n".join(lines).strip()
