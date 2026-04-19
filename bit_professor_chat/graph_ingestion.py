from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional for markdown-only runs
    GraphDatabase = None

from .config import TutorSettings
from .ingestion_models import StructuredGraphInsertionResult
from .structured_profiles import StructuredProfessorProfileArtifact, normalize_lookup_text


LAB3_STRUCTURED_OUTPUT_DIR = Path("lab_3_langgraph_swarm") / "structured_output"
STRUCTURED_OUTPUT_FILE_SUFFIX = "-profile.json"
DEFAULT_TYPED_GRAPH_NAME = "BIT_CSAT_TYPED"


def _kg_gen_removed_error() -> RuntimeError:
    return RuntimeError(
        "The kg-gen pipeline has been removed. Use the OCR -> structured output "
        "flow instead: `uv run python lab_3_langgraph_swarm/build_structured_output.py` "
        "followed by `uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py`."
    )


def require_kg_gen() -> None:
    raise _kg_gen_removed_error()


def save_graph_json(graph: Any, output_path: Path) -> None:
    raise _kg_gen_removed_error()


def save_graph_html(graph: Any, output_path: Path) -> None:
    raise _kg_gen_removed_error()


def graph_to_payload(graph: Any) -> dict[str, Any]:
    raise _kg_gen_removed_error()


def graph_from_payload(payload: dict[str, Any]) -> Any:
    raise _kg_gen_removed_error()


def generate_professor_graph(markdown_text: str, settings: TutorSettings, *, context: str = "") -> Any:
    raise _kg_gen_removed_error()


def insert_professor_graph(
    *,
    graph: Any,
    settings: TutorSettings,
    graph_name: str,
    professor_name: str,
    detail_url: str,
) -> dict[str, int]:
    raise _kg_gen_removed_error()


def require_neo4j_driver() -> type[GraphDatabase]:
    if GraphDatabase is None:
        raise ImportError(
            "neo4j is required for graph-backed workflows. Install the Neo4j Python "
            "driver to run Lab 1 or Lab 3."
        )
    return GraphDatabase


def _open_driver(settings: TutorSettings) -> Any:
    settings.require_neo4j()
    neo4j_driver = require_neo4j_driver()
    return neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _drop_user_schema(session: Any) -> None:
    constraint_rows = session.run(
        "SHOW CONSTRAINTS YIELD name RETURN name ORDER BY name"
    ).data()
    for row in constraint_rows:
        session.run(f"DROP CONSTRAINT {_quote_identifier(row['name'])} IF EXISTS")

    index_rows = session.run(
        """
        SHOW INDEXES YIELD name, type, owningConstraint
        WHERE type <> 'LOOKUP' AND (owningConstraint IS NULL OR owningConstraint = '')
        RETURN name
        ORDER BY name
        """
    ).data()
    for row in index_rows:
        session.run(f"DROP INDEX {_quote_identifier(row['name'])} IF EXISTS")


def reset_neo4j_database(
    settings: TutorSettings,
    *,
    drop_schema: bool = True,
) -> None:
    driver = _open_driver(settings)
    try:
        with driver.session(database=settings.neo4j_database) as neo4j_session:
            neo4j_session.run("MATCH (n) DETACH DELETE n")
            if drop_schema:
                _drop_user_schema(neo4j_session)
    finally:
        driver.close()


def _ensure_constraints(session: Any) -> None:
    statements = (
        "CREATE CONSTRAINT professor_detail_url_unique IF NOT EXISTS FOR (p:Professor) REQUIRE p.detail_url IS UNIQUE",
        "CREATE CONSTRAINT organization_normalized_name_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.normalized_name IS UNIQUE",
        "CREATE CONSTRAINT research_topic_normalized_name_unique IF NOT EXISTS FOR (t:ResearchTopic) REQUIRE t.normalized_name IS UNIQUE",
        "CREATE CONSTRAINT experience_key_unique IF NOT EXISTS FOR (e:Experience) REQUIRE e.experience_key IS UNIQUE",
        "CREATE CONSTRAINT publication_key_unique IF NOT EXISTS FOR (p:Publication) REQUIRE p.publication_key IS UNIQUE",
        "CREATE CONSTRAINT award_key_unique IF NOT EXISTS FOR (a:Award) REQUIRE a.award_key IS UNIQUE",
        "CREATE INDEX professor_name_index IF NOT EXISTS FOR (p:Professor) ON (p.name)",
        "CREATE INDEX research_topic_name_index IF NOT EXISTS FOR (t:ResearchTopic) ON (t.name)",
        "CREATE INDEX organization_name_index IF NOT EXISTS FOR (o:Organization) ON (o.name)",
    )
    for statement in statements:
        session.run(statement)


def resolve_structured_output_dir(
    project_root: Path,
    structured_output_dir: Path | None = None,
) -> Path:
    if structured_output_dir is None:
        return project_root / LAB3_STRUCTURED_OUTPUT_DIR
    return (
        structured_output_dir
        if structured_output_dir.is_absolute()
        else project_root / structured_output_dir
    )


def _slug_from_structured_output_path(path: Path) -> str:
    return path.name.removesuffix(STRUCTURED_OUTPUT_FILE_SUFFIX)


def _select_structured_output_paths(
    *,
    base_dir: Path,
    only_slugs: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    selected_slugs = {slug.strip() for slug in (only_slugs or []) if slug.strip()}
    paths = sorted(base_dir.glob(f"*{STRUCTURED_OUTPUT_FILE_SUFFIX}"))
    if selected_slugs:
        paths = [
            path for path in paths if _slug_from_structured_output_path(path) in selected_slugs
        ]
    if limit is not None:
        paths = paths[:limit]
    return paths


def _load_structured_output_artifact(path: Path) -> StructuredProfessorProfileArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StructuredProfessorProfileArtifact.model_validate(payload)


def load_structured_output_records(
    *,
    project_root: Path,
    structured_output_dir: Path | None = None,
    only_slugs: list[str] | None = None,
    limit: int | None = None,
) -> list[tuple[Path, StructuredProfessorProfileArtifact]]:
    resolved_output_dir = resolve_structured_output_dir(project_root, structured_output_dir)
    if not resolved_output_dir.exists():
        raise FileNotFoundError(
            f"Structured output directory does not exist: {resolved_output_dir}"
        )

    output_paths = _select_structured_output_paths(
        base_dir=resolved_output_dir,
        only_slugs=only_slugs,
        limit=limit,
    )
    if not output_paths:
        raise FileNotFoundError(
            f"No structured output files were found under {resolved_output_dir}."
        )
    return [(path, _load_structured_output_artifact(path)) for path in output_paths]


def _merge_organization(
    session: Any,
    *,
    name: str,
    org_type: str,
    graph_name: str,
) -> str | None:
    normalized_name = normalize_lookup_text(name)
    if not normalized_name:
        return None
    session.run(
        """
        MERGE (o:Organization {normalized_name: $normalized_name})
        ON CREATE SET
          o.name = $name,
          o.org_type = $org_type,
          o.graph_name = $graph_name
        SET
          o.name = CASE WHEN $name <> '' THEN $name ELSE o.name END,
          o.org_type =
            CASE
              WHEN $org_type <> '' AND (o.org_type IS NULL OR o.org_type = 'unknown') THEN $org_type
              ELSE coalesce(o.org_type, 'unknown')
            END,
          o.graph_name = $graph_name
        """,
        normalized_name=normalized_name,
        name=name.strip(),
        org_type=org_type.strip() or "unknown",
        graph_name=graph_name,
    )
    return normalized_name


def _merge_research_topic(
    session: Any,
    *,
    name: str,
    normalized_name: str,
    graph_name: str,
) -> str | None:
    topic_key = normalize_lookup_text(normalized_name or name)
    if not topic_key:
        return None
    session.run(
        """
        MERGE (t:ResearchTopic {normalized_name: $normalized_name})
        ON CREATE SET
          t.name = $name,
          t.graph_name = $graph_name
        SET
          t.name = CASE WHEN $name <> '' THEN $name ELSE t.name END,
          t.graph_name = $graph_name
        """,
        normalized_name=topic_key,
        name=name.strip(),
        graph_name=graph_name,
    )
    return topic_key


def _clear_professor_subgraph(session: Any, *, detail_url: str) -> None:
    session.run(
        """
        MATCH (p:Professor {detail_url: $detail_url})-[r:AFFILIATED_WITH|HAS_RESEARCH_INTEREST]->()
        DELETE r
        """,
        detail_url=detail_url,
    )
    session.run(
        """
        MATCH (p:Professor {detail_url: $detail_url})-[:HAS_EXPERIENCE]->(e:Experience {professor_detail_url: $detail_url})
        DETACH DELETE e
        """,
        detail_url=detail_url,
    )
    session.run(
        """
        MATCH (p:Professor {detail_url: $detail_url})-[:AUTHORED]->(pub:Publication {professor_detail_url: $detail_url})
        DETACH DELETE pub
        """,
        detail_url=detail_url,
    )
    session.run(
        """
        MATCH (p:Professor {detail_url: $detail_url})-[:RECEIVED]->(award:Award {professor_detail_url: $detail_url})
        DETACH DELETE award
        """,
        detail_url=detail_url,
    )


def _cleanup_orphan_nodes(session: Any) -> None:
    session.run(
        """
        MATCH (n)
        WHERE (n:Organization OR n:ResearchTopic OR n:Experience OR n:Publication OR n:Award)
          AND NOT (n)--()
        DETACH DELETE n
        """
    )


def _insert_structured_artifact(
    *,
    session: Any,
    artifact: StructuredProfessorProfileArtifact,
    graph_name: str,
) -> None:
    metadata = artifact.metadata
    profile = artifact.profile
    professor = profile.professor
    detail_url = metadata.detail_url.strip()
    if not detail_url:
        raise ValueError("metadata.detail_url is required")

    _clear_professor_subgraph(session, detail_url=detail_url)

    session.run(
        """
        MERGE (p:Professor {detail_url: $detail_url})
        SET
          p.name = $name,
          p.aliases = $aliases,
          p.title = $title,
          p.school_name = $school_name,
          p.discipline = $discipline,
          p.biography_text = $biography_text,
          p.emails = $emails,
          p.phones = $phones,
          p.websites = $websites,
          p.locations = $locations,
          p.slug = $slug,
          p.source_ocr_markdown_path = $source_ocr_markdown_path,
          p.warnings = $warnings,
          p.graph_name = $graph_name
        """,
        detail_url=detail_url,
        name=professor.name.strip() or metadata.name.strip(),
        aliases=[value for value in professor.aliases if value.strip()],
        title=professor.title,
        school_name=professor.school_name,
        discipline=professor.discipline,
        biography_text=professor.biography_text,
        emails=professor.emails,
        phones=professor.phones,
        websites=professor.websites,
        locations=professor.locations,
        slug=metadata.slug,
        source_ocr_markdown_path=metadata.source_ocr_markdown_path,
        warnings=profile.warnings,
        graph_name=graph_name,
    )

    for organization in profile.organizations:
        _merge_organization(
            session,
            name=organization.name,
            org_type=organization.org_type,
            graph_name=graph_name,
        )

    if professor.school_name.strip():
        school_normalized_name = _merge_organization(
            session,
            name=professor.school_name,
            org_type="school",
            graph_name=graph_name,
        )
        if school_normalized_name:
            session.run(
                """
                MATCH (p:Professor {detail_url: $detail_url})
                MATCH (o:Organization {normalized_name: $normalized_name})
                MERGE (p)-[r:AFFILIATED_WITH]->(o)
                SET r.graph_name = $graph_name
                """,
                detail_url=detail_url,
                normalized_name=school_normalized_name,
                graph_name=graph_name,
            )

    for topic in profile.research_topics:
        topic_key = _merge_research_topic(
            session,
            name=topic.name,
            normalized_name=topic.normalized_name,
            graph_name=graph_name,
        )
        if topic_key is None:
            continue
        session.run(
            """
            MATCH (p:Professor {detail_url: $detail_url})
            MATCH (t:ResearchTopic {normalized_name: $normalized_name})
            MERGE (p)-[r:HAS_RESEARCH_INTEREST]->(t)
            SET r.graph_name = $graph_name
            """,
            detail_url=detail_url,
            normalized_name=topic_key,
            graph_name=graph_name,
        )

    for experience in profile.experiences:
        experience_key = f"{detail_url}|{experience.kind}|{experience.order}"
        session.run(
            """
            MERGE (e:Experience {experience_key: $experience_key})
            SET
              e.professor_detail_url = $detail_url,
              e.kind = $kind,
              e.title = $title,
              e.degree = $degree,
              e.field = $field,
              e.organization_name = $organization_name,
              e.location = $location,
              e.start_text = $start_text,
              e.end_text = $end_text,
              e.is_current = $is_current,
              e.order = $order,
              e.raw_text = $raw_text,
              e.graph_name = $graph_name
            """,
            experience_key=experience_key,
            detail_url=detail_url,
            kind=experience.kind,
            title=experience.title,
            degree=experience.degree,
            field=experience.field,
            organization_name=experience.organization_name,
            location=experience.location,
            start_text=experience.start_text,
            end_text=experience.end_text,
            is_current=experience.is_current,
            order=experience.order,
            raw_text=experience.raw_text,
            graph_name=graph_name,
        )
        session.run(
            """
            MATCH (p:Professor {detail_url: $detail_url})
            MATCH (e:Experience {experience_key: $experience_key})
            MERGE (p)-[r:HAS_EXPERIENCE]->(e)
            SET r.graph_name = $graph_name
            """,
            detail_url=detail_url,
            experience_key=experience_key,
            graph_name=graph_name,
        )
        if experience.organization_name.strip():
            organization_key = _merge_organization(
                session,
                name=experience.organization_name,
                org_type="unknown",
                graph_name=graph_name,
            )
            if organization_key:
                session.run(
                    """
                    MATCH (e:Experience {experience_key: $experience_key})
                    MATCH (o:Organization {normalized_name: $normalized_name})
                    MERGE (e)-[r:AT]->(o)
                    SET r.graph_name = $graph_name
                    """,
                    experience_key=experience_key,
                    normalized_name=organization_key,
                    graph_name=graph_name,
                )

    for publication in profile.publications:
        publication_key = f"{detail_url}|{publication.order}"
        session.run(
            """
            MERGE (pub:Publication {publication_key: $publication_key})
            SET
              pub.professor_detail_url = $detail_url,
              pub.order = $order,
              pub.title = $title,
              pub.authors_raw = $authors_raw,
              pub.year = $year,
              pub.venue = $venue,
              pub.publication_type = $publication_type,
              pub.doi_or_url = $doi_or_url,
              pub.raw_text = $raw_text,
              pub.graph_name = $graph_name
            """,
            publication_key=publication_key,
            detail_url=detail_url,
            order=publication.order,
            title=publication.title,
            authors_raw=publication.authors_raw,
            year=publication.year,
            venue=publication.venue,
            publication_type=publication.publication_type,
            doi_or_url=publication.doi_or_url,
            raw_text=publication.raw_text,
            graph_name=graph_name,
        )
        session.run(
            """
            MATCH (p:Professor {detail_url: $detail_url})
            MATCH (pub:Publication {publication_key: $publication_key})
            MERGE (p)-[r:AUTHORED]->(pub)
            SET r.graph_name = $graph_name
            """,
            detail_url=detail_url,
            publication_key=publication_key,
            graph_name=graph_name,
        )

    for award in profile.awards:
        award_key = f"{detail_url}|{award.order}"
        session.run(
            """
            MERGE (a:Award {award_key: $award_key})
            SET
              a.professor_detail_url = $detail_url,
              a.order = $order,
              a.name = $name,
              a.year = $year,
              a.granting_org_name = $granting_org_name,
              a.level = $level,
              a.raw_text = $raw_text,
              a.graph_name = $graph_name
            """,
            award_key=award_key,
            detail_url=detail_url,
            order=award.order,
            name=award.name,
            year=award.year,
            granting_org_name=award.granting_org_name,
            level=award.level,
            raw_text=award.raw_text,
            graph_name=graph_name,
        )
        session.run(
            """
            MATCH (p:Professor {detail_url: $detail_url})
            MATCH (a:Award {award_key: $award_key})
            MERGE (p)-[r:RECEIVED]->(a)
            SET r.graph_name = $graph_name
            """,
            detail_url=detail_url,
            award_key=award_key,
            graph_name=graph_name,
        )
        if award.granting_org_name.strip():
            organization_key = _merge_organization(
                session,
                name=award.granting_org_name,
                org_type="unknown",
                graph_name=graph_name,
            )
            if organization_key:
                session.run(
                    """
                    MATCH (a:Award {award_key: $award_key})
                    MATCH (o:Organization {normalized_name: $normalized_name})
                    MERGE (a)-[r:GRANTED_BY]->(o)
                    SET r.graph_name = $graph_name
                    """,
                    award_key=award_key,
                    normalized_name=organization_key,
                    graph_name=graph_name,
                )


def verify_neo4j_graph(settings: TutorSettings) -> dict[str, Any]:
    driver = _open_driver(settings)
    try:
        with driver.session(database=settings.neo4j_database) as neo4j_session:
            node_count = neo4j_session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            relationship_count = neo4j_session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]
            distinct_professors = [
                record["professor_name"]
                for record in neo4j_session.run(
                    """
                    MATCH (p:Professor)
                    RETURN p.name AS professor_name
                    ORDER BY professor_name
                    """
                )
            ]
            label_rows = neo4j_session.run(
                """
                CALL () {
                  MATCH (n:Professor) RETURN 'Professor' AS label, count(n) AS count
                  UNION ALL
                  MATCH (n:Organization) RETURN 'Organization' AS label, count(n) AS count
                  UNION ALL
                  MATCH (n:ResearchTopic) RETURN 'ResearchTopic' AS label, count(n) AS count
                  UNION ALL
                  MATCH (n:Experience) RETURN 'Experience' AS label, count(n) AS count
                  UNION ALL
                  MATCH (n:Publication) RETURN 'Publication' AS label, count(n) AS count
                  UNION ALL
                  MATCH (n:Award) RETURN 'Award' AS label, count(n) AS count
                }
                RETURN label, count
                """
            ).data()
    finally:
        driver.close()
    return {
        "node_count": node_count,
        "relationship_count": relationship_count,
        "distinct_professors": distinct_professors,
        "label_counts": {row["label"]: row["count"] for row in label_rows},
    }


def insert_structured_output_to_neo4j(
    *,
    settings: TutorSettings,
    project_root: Path | None = None,
    structured_output_dir: Path | None = None,
    only_slugs: list[str] | None = None,
    limit: int | None = None,
    reset_database: bool = False,
    graph_name: str = DEFAULT_TYPED_GRAPH_NAME,
) -> StructuredGraphInsertionResult:
    resolved_project_root = project_root or settings.project_root
    structured_output_records = load_structured_output_records(
        project_root=resolved_project_root,
        structured_output_dir=structured_output_dir,
        only_slugs=only_slugs,
        limit=limit,
    )
    resolved_output_dir = resolve_structured_output_dir(
        resolved_project_root,
        structured_output_dir,
    )

    if reset_database:
        reset_neo4j_database(settings)

    driver = _open_driver(settings)
    failures: list[dict[str, str]] = []
    inserted_professors: list[str] = []
    try:
        with driver.session(database=settings.neo4j_database) as neo4j_session:
            _ensure_constraints(neo4j_session)
            for path, artifact in structured_output_records:
                slug = artifact.metadata.slug or _slug_from_structured_output_path(path)
                try:
                    _insert_structured_artifact(
                        session=neo4j_session,
                        artifact=artifact,
                        graph_name=graph_name,
                    )
                    inserted_professors.append(slug)
                except Exception as exc:
                    failures.append({"slug": slug or path.name, "error": str(exc)})
            _cleanup_orphan_nodes(neo4j_session)
    finally:
        driver.close()

    verification = verify_neo4j_graph(settings)
    return StructuredGraphInsertionResult(
        output_dir=str(resolved_output_dir.relative_to(resolved_project_root)),
        professor_count=len(structured_output_records),
        success_count=len(inserted_professors),
        failure_count=len(failures),
        failures=failures,
        inserted_professors=sorted(inserted_professors),
        node_count=verification["node_count"],
        relationship_count=verification["relationship_count"],
        label_counts=verification["label_counts"],
    )


__all__ = [
    "DEFAULT_TYPED_GRAPH_NAME",
    "LAB3_STRUCTURED_OUTPUT_DIR",
    "STRUCTURED_OUTPUT_FILE_SUFFIX",
    "insert_structured_output_to_neo4j",
    "load_structured_output_records",
    "require_neo4j_driver",
    "reset_neo4j_database",
    "resolve_structured_output_dir",
    "verify_neo4j_graph",
]
