---
name: add-professor-from-url
description: Use this skill when the task is to add one BIT professor from an official detail-page URL into the sandbox professor workspace.
---

# Add Professor From URL

## When To Use

Use this skill when:

- the user provides a BIT professor detail URL
- the user asks to add or refresh one professor from the official source page

Do not use this skill for normal question-answering over existing sandbox files.

## Goal

Create exactly one canonical dossier file under `professors/<slug>.md` from one official detail URL inside the sandbox workspace.

The parent agent is responsible for rebuilding `professors.md`. This skill only covers the dossier-creation workflow.

## Required Workflow

1. Treat `/` as the sandbox root and inspect `professors/*.md` first so you understand the current workspace.
2. Accept only an official BIT CSAT detail URL shaped like `https://isc.bit.edu.cn/schools/csat/knowingprofessors5/b123456.htm`.
3. Run exactly one command with `execute` from the sandbox root:
   - `python skills/add-professor-from-url/scripts/add_professor_from_url.py --detail-url "<detail-url>" --professors-dir professors`
4. Read the script stdout.
5. If the script returns `status=added` or `status=duplicate_found`, return one compact summary line to the parent and stop.
6. If the script fails, return one compact `status=failed` summary line and stop.

## Execute Guidance

- Use `execute` exactly once for the script entrypoint.
- The Lab 4 backend only allows this documented add-professor command. Do not try any other shell command.
- Do not chain together extra crawl/OCR shell commands around it.
- Do not rewrite the dossier markdown by hand after the script succeeds.
- The script already validates the URL, resolves the professor name, performs crawl + OCR + structured-profile extraction, checks duplicates, and writes the runtime dossier file when appropriate.
- After the script finishes, do not continue exploring or editing unrelated files.

## File Contract

- The dossier output path is still `professors/<slug>.md`.
- The script also saves the cleaned OCR markdown snapshot under `incoming/<slug>-ocr.md`.
- If structured extraction succeeds, the script also saves the typed structured profile snapshot under `incoming/<slug>-profile.json`.
- Do not modify unrelated dossier files yourself.
- Do not rebuild `professors.md`.
- Do not inspect unrelated files after the script is complete.

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
