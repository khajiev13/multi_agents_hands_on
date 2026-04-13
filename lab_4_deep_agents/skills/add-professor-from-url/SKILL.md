---
name: add-professor-from-url
description: Use this skill when the task is to add one BIT professor from an official detail-page URL into the local professor workspace.
---

# Add Professor From URL

## When To Use

Use this skill when:

- the user provides a BIT professor detail URL
- the user asks to add or refresh one professor from the official source page

Do not use this skill for normal question-answering over existing local files.

## Goal

Create exactly one canonical dossier file under `/professors/<slug>.md` from one official detail URL.

The parent agent is responsible for rebuilding `/professors.md`. This skill only covers the dossier-creation workflow.

## Required Workflow

1. Inspect the existing dossier folder first.
   - Check `/professors/*.md` for an existing matching `detail_url`.
   - Also check for a likely matching slug if the same professor may already exist under a different filename.
2. If the professor already exists:
   - do not create a duplicate file
   - return a compact duplicate result to the parent
3. If the professor is new:
   - call `extract_image_urls_tool(detail_url=...)`
   - call `ocr_images_to_professor_markdown_tool(detail_url=..., image_urls=..., professor_name_hint="")`
   - write the returned `canonical_markdown` exactly to `/professors/<slug>.md`
   - stop immediately after the dossier file has been written successfully
4. Return a short completion summary to the parent agent.

## Tool Guidance

- `extract_image_urls_tool` is the page-fetching step. Use it before OCR.
- `ocr_images_to_professor_markdown_tool` returns canonical dossier markdown and metadata. Do not rewrite the markdown by hand.
- Use file tools to inspect existing dossiers and to write the new dossier file.
- After the dossier write succeeds, do not continue exploring or editing other files.
- Do not use `execute` in this workflow.

## File Contract

- The dossier output path must be `/professors/<slug>.md`.
- The dossier body must be the returned `canonical_markdown` with no extra wrapper text.
- Do not modify unrelated dossier files.
- Do not rebuild `/professors.md`.
- Do not inspect unrelated files after the dossier write is complete.

## Final Return To Parent

Return one compact result that includes:

- `status`
- `professor_name`
- `slug`
- `markdown_path`
- `page_count`

Return exactly one final summary line and then stop.

Valid `status` values:

- `added`
- `duplicate_found`
- `failed`
