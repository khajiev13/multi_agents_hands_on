---
name: add-professor-from-incoming
description: Use this skill when the task is to add one prepared incoming professor artifact from /incoming/ into the local professor workspace.
---

# Add Professor From Incoming Artifact

## When To Use

Use this skill when:

- the user asks to add the prepared incoming professor artifact
- the workspace already contains one or more prepared markdown files under `/incoming/`

Do not use this skill for normal question-answering over existing local files.

## Goal

Create exactly one dossier file under `/professors/<slug>.md` from one prepared incoming markdown artifact.

The parent agent is responsible for rebuilding `/professors.md`. This skill only covers the dossier-add workflow.

## Required Workflow

1. Inspect `/incoming/*.md` and `/professors/*.md` first.
2. Choose the prepared incoming artifact the user is referring to.
3. Treat the incoming filename stem as the target slug.
4. Check for duplicates before writing:
   - `/professors/<slug>.md` already exists, or
   - an existing dossier has the same `detail_url` in its markdown header
5. If the professor already exists:
   - do not create a duplicate file
   - return a compact duplicate result to the parent
6. If the professor is new:
   - read the incoming markdown
   - write that markdown exactly to `/professors/<slug>.md`
   - stop immediately after the dossier file has been written successfully
7. Return a short completion summary to the parent agent.

## File Contract

- The dossier output path must be `/professors/<slug>.md`.
- The dossier body must be copied exactly from the chosen incoming markdown file.
- Do not modify the source file under `/incoming/`.
- Do not modify unrelated dossier files.
- Do not rebuild `/professors.md`.

## Final Return To Parent

Return one compact result that includes:

- `status`
- `professor_name`
- `slug`
- `markdown_path`

Return exactly one final summary line and then stop.

Valid `status` values:

- `added`
- `duplicate_found`
- `failed`
