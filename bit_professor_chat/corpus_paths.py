from __future__ import annotations

from pathlib import Path


DEFAULT_PROFESSOR_CORPUS_DIR = Path("lab_4_deep_agents") / "professors"
DEFAULT_PROFESSOR_SUMMARY_PATH = Path("lab_4_deep_agents") / "professors.md"


def _resolve_project_relative_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def resolve_professor_corpus_dir(
    project_root: Path,
    professor_dir: Path | None = None,
) -> Path:
    return _resolve_project_relative_path(
        project_root,
        professor_dir or DEFAULT_PROFESSOR_CORPUS_DIR,
    )


def resolve_professor_summary_path(
    project_root: Path,
    summary_path: Path | None = None,
) -> Path:
    return _resolve_project_relative_path(
        project_root,
        summary_path or DEFAULT_PROFESSOR_SUMMARY_PATH,
    )


__all__ = [
    "DEFAULT_PROFESSOR_CORPUS_DIR",
    "DEFAULT_PROFESSOR_SUMMARY_PATH",
    "resolve_professor_corpus_dir",
    "resolve_professor_summary_path",
]
