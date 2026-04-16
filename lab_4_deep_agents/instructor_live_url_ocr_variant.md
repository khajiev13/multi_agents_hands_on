# Instructor-Only Live URL + OCR Variant

This note preserves the advanced Lab 4 extension while keeping the student notebook fully file-based.

The student-facing notebook now focuses only on:

- `/professors.md`
- `/professors/*.md`
- `/incoming/*.md`
- `/skills/add-professor-from-incoming/SKILL.md`

Use the live variant only when you explicitly want to demonstrate how that local file workflow can be extended to a live BIT professor URL.

## What Stays Reusable

- `lab_4_deep_agents/skills/add-professor-from-url/SKILL.md`
- `bit_professor_chat/source_discovery.py` for listing/detail-page discovery
- `bit_professor_chat/ocr_transcript.py` for OCR-to-markdown helpers
- `lab_1_langchain_pipeline/prepare_lab1_graph.py` for full instructor refresh of the professor corpus and Lab 3 seed data

## Suggested Instructor Demo Flow

1. Start from the regular Lab 4 runtime workspace so students see the file-based baseline first.
2. Introduce one official BIT professor detail URL as an advanced input.
3. Use the `add-professor-from-url` skill plus OCR helpers to produce one canonical dossier markdown file.
4. Rebuild `/professors.md` after the new dossier is added.
5. Compare the live-add path with the simpler prepared-incoming path and explain why the student notebook omits crawling and OCR.

## Practical Guidance

- Treat this as an optional instructor demo, not part of the required student workflow.
- Keep the live variant separate from the core notebook so network issues or OCR instability cannot derail class.
- If the live path is flaky during class, fall back to the prepared incoming artifact flow in the student notebook.
