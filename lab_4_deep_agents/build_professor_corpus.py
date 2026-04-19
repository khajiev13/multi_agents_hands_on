from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bit_professor_chat.corpus_paths import (
    resolve_professor_corpus_dir,
    resolve_professor_summary_path,
)
from bit_professor_chat.graph_ingestion import load_structured_output_records
from bit_professor_chat.legacy_cache import build_cached_summary_line
from bit_professor_chat.markdown_corpus import deslugify_professor_slug
from bit_professor_chat.markdown_render import render_professor_markdown
from bit_professor_chat.structured_profiles import structured_profile_artifact_to_dossier


def build_professor_corpus() -> dict[str, object]:
    corpus_dir = resolve_professor_corpus_dir(PROJECT_ROOT)
    summary_path = resolve_professor_summary_path(PROJECT_ROOT)

    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_structured_output_records(project_root=PROJECT_ROOT)
    summary_lines: list[str] = []
    written_files: list[str] = []

    for _, artifact in records:
        dossier = structured_profile_artifact_to_dossier(artifact)
        dossier = dossier.model_copy(update={"title": deslugify_professor_slug(artifact.metadata.slug)})
        markdown_text = render_professor_markdown(dossier).strip() + "\n"
        markdown_path = corpus_dir / f"{artifact.metadata.slug}.md"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        written_files.append(str(markdown_path.relative_to(PROJECT_ROOT)))
        summary_lines.append(
            build_cached_summary_line(markdown_text, deslugify_professor_slug(artifact.metadata.slug))
        )

    lines = ["# BIT CSAT Professors", ""]
    lines.extend(f"- {line}" for line in sorted(dict.fromkeys(summary_lines), key=str.casefold))
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "professor_count": len(written_files),
        "corpus_dir": str(corpus_dir.relative_to(PROJECT_ROOT)),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "sample_files": written_files[:8],
    }


def main() -> None:
    print(json.dumps(build_professor_corpus(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
