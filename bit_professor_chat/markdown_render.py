from __future__ import annotations

from .ingestion_models import ProfessorDossier


def _clean_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line and line.strip()]


def render_professor_markdown(dossier: ProfessorDossier) -> str:
    lines = [
        f"# {dossier.title.strip()}",
        f"- detail_url: {dossier.detail_url.strip()}",
    ]
    if dossier.source_page_urls:
        lines.append(f"- page_count: {len(dossier.source_page_urls)}")

    basic_information = _clean_lines(dossier.basic_information.bullets)
    if basic_information:
        lines.extend(["", "## Basic Information"])
        lines.extend(f"- {line}" for line in basic_information)

    for section in dossier.sections:
        section_heading = section.heading.strip()
        section_lines = _clean_lines(section.bullets)
        if not section_heading or not section_lines:
            continue
        lines.extend(["", f"## {section_heading}"])
        lines.extend(f"- {line}" for line in section_lines)

    uncertain_lines = _clean_lines(dossier.uncertain_lines)
    if uncertain_lines:
        lines.extend(["", "## Uncertain or Illegible Text"])
        lines.extend(f"- {line}" for line in uncertain_lines)

    if dossier.source_page_urls:
        lines.extend(["", "## Source Pages"])
        lines.extend(
            f"- page {page_number}: {url}"
            for page_number, url in enumerate(dossier.source_page_urls, start=1)
        )
    return "\n".join(lines).strip()


__all__ = ["render_professor_markdown"]
