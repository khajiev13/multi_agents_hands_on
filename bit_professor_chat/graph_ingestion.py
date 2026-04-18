from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

try:
    from kg_gen import KGGen
    from kg_gen.steps._1_get_entities import get_entities
    from kg_gen.steps._2_get_relations import get_relations
    import dspy
except ImportError:  # pragma: no cover - optional for Lab 2 markdown-only runs
    KGGen = None
    get_entities = None
    get_relations = None
    dspy = None

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional for Lab 2 markdown-only runs
    GraphDatabase = None

from .config import TutorSettings
from .ingestion_models import StructuredProfessorReview, StructuredSeedInsertionResult
from .structured_seed import load_structured_seed_reviews, resolve_structured_seed_dir


def require_kg_gen() -> type[KGGen]:
    if KGGen is None or get_entities is None or get_relations is None or dspy is None:
        raise ImportError(
            "kg-gen is required for graph generation. Install the project dependencies "
            "used by Lab 1 or run Lab 2 with build_graph=False."
        )
    return KGGen


def require_neo4j_driver() -> type[GraphDatabase]:
    if GraphDatabase is None:
        raise ImportError(
            "neo4j is required for graph-backed workflows. Install the Neo4j Python "
            "driver to run Lab 1 or any build_graph=True flow."
        )
    return GraphDatabase


def save_graph_json(graph: Any, output_path: Path) -> None:
    payload = graph_to_payload(graph)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_graph_html(graph: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require_kg_gen().visualize(graph, str(output_path), open_in_browser=False)


def graph_to_payload(graph: Any) -> dict[str, Any]:
    return {
        "entities": sorted(str(entity) for entity in getattr(graph, "entities", [])),
        "edges": sorted(str(edge) for edge in getattr(graph, "edges", [])),
        "relations": sorted(
            [list(relation) for relation in getattr(graph, "relations", [])],
            key=lambda value: (value[0], value[1], value[2]),
        ),
        "entity_clusters": {
            key: sorted(str(value) for value in values)
            for key, values in sorted((getattr(graph, "entity_clusters", None) or {}).items())
        }
        or None,
        "edge_clusters": {
            key: sorted(str(value) for value in values)
            for key, values in sorted((getattr(graph, "edge_clusters", None) or {}).items())
        }
        or None,
    }


def graph_from_payload(payload: dict[str, Any]) -> Any:
    return require_kg_gen().from_dict(payload)


def build_kg_generator(settings: TutorSettings) -> KGGen:
    kg_class = require_kg_gen()
    config = settings.require_graph_generation()
    model_name = config["model"]
    if "/" not in model_name:
        model_name = f"openai/{model_name}"
    return kg_class(
        model=model_name,
        api_key=config["api_key"],
        api_base=config["base_url"],
        temperature=0.0,
    )


def generate_professor_graph_with_generator(
    *,
    markdown_text: str,
    kg: KGGen,
    context: str,
) -> Any:
    with dspy.context(lm=kg.lm):
        entities = get_entities(markdown_text, is_conversation=False)
        relations = get_relations(
            markdown_text,
            entities,
            is_conversation=False,
            context=context,
        )
    return kg.from_dict(
        {
            "entities": set(entities),
            "relations": set(tuple(relation) for relation in relations),
            "edges": {relation[1] for relation in relations},
        }
    )


def aggregate_graphs(*, kg: KGGen, graphs: list[Any]) -> Any:
    return kg.aggregate(graphs)


def cluster_graph(*, kg: KGGen, graph: Any, context: str) -> Any:
    return kg.cluster(graph, context=context)


def generate_professor_graph(
    markdown_text: str,
    settings: TutorSettings,
    *,
    context: str = "Beijing Institute of Technology professor profile",
) -> Any:
    kg = build_kg_generator(settings)
    return generate_professor_graph_with_generator(
        markdown_text=markdown_text,
        kg=kg,
        context=context,
    )


def reset_neo4j_database(settings: TutorSettings) -> None:
    settings.require_neo4j()
    neo4j_driver = require_neo4j_driver()
    driver = neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    with driver.session(database=settings.neo4j_database) as neo4j_session:
        neo4j_session.run("MATCH (n) DETACH DELETE n")
    driver.close()


def _normalize_rel_type(predicate: str) -> str:
    return re.sub(
        r"[^A-Z0-9_]",
        "_",
        predicate.upper().replace(" ", "_").replace("-", "_"),
    )


def insert_professor_graph(
    *,
    graph: Any,
    settings: TutorSettings,
    graph_name: str,
    professor_name: str,
    detail_url: str,
) -> dict[str, int]:
    settings.require_neo4j()
    neo4j_driver = require_neo4j_driver()
    driver = neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    with driver.session(database=settings.neo4j_database) as neo4j_session:
        neo4j_session.run(
            """
            MERGE (p:Entity {name: $professor_name})
            SET p.graph_name = $graph_name,
                p.professor_name = $professor_name,
                p.detail_url = $detail_url,
                p.source_professors =
                    CASE
                        WHEN p.source_professors IS NULL THEN [$professor_name]
                        WHEN NOT $professor_name IN p.source_professors THEN p.source_professors + $professor_name
                        ELSE p.source_professors
                    END
            """,
            graph_name=graph_name,
            professor_name=professor_name,
            detail_url=detail_url,
        )

        neo4j_session.run(
            """
            UNWIND $entities AS entity
            MERGE (n:Entity {name: entity})
            SET n.graph_name = $graph_name,
                n.source_professors =
                    CASE
                        WHEN n.source_professors IS NULL THEN [$professor_name]
                        WHEN NOT $professor_name IN n.source_professors THEN n.source_professors + $professor_name
                        ELSE n.source_professors
                    END
            FOREACH (_ IN CASE WHEN entity = $professor_name THEN [1] ELSE [] END |
                SET n.professor_name = $professor_name,
                    n.detail_url = $detail_url
            )
            """,
            entities=list(graph.entities),
            graph_name=graph_name,
            professor_name=professor_name,
            detail_url=detail_url,
        )

        for subject, predicate, obj in graph.relations:
            rel_type = _normalize_rel_type(predicate)
            neo4j_session.run(
                f"""
                MATCH (s:Entity {{name: $subject}})
                MATCH (o:Entity {{name: $object}})
                MERGE (s)-[r:{rel_type}]->(o)
                SET r.predicate = $predicate,
                    r.graph_name = $graph_name,
                    r.source_professors =
                        CASE
                            WHEN r.source_professors IS NULL THEN [$professor_name]
                            WHEN NOT $professor_name IN r.source_professors THEN r.source_professors + $professor_name
                            ELSE r.source_professors
                        END
                """,
                subject=subject,
                object=obj,
                predicate=predicate,
                graph_name=graph_name,
                professor_name=professor_name,
            )

        node_count = neo4j_session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        relationship_count = neo4j_session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
    driver.close()
    return {"node_count": node_count, "relationship_count": relationship_count}


def verify_neo4j_graph(settings: TutorSettings) -> dict[str, Any]:
    settings.require_neo4j()
    neo4j_driver = require_neo4j_driver()
    driver = neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    with driver.session(database=settings.neo4j_database) as neo4j_session:
        node_count = neo4j_session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        relationship_count = neo4j_session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
        distinct_professors = [
            record["professor_name"]
            for record in neo4j_session.run(
                """
                MATCH (n:Entity)
                WHERE n.professor_name IS NOT NULL
                RETURN DISTINCT n.professor_name AS professor_name
                ORDER BY professor_name
                """
            )
        ]
    driver.close()
    return {
        "node_count": node_count,
        "relationship_count": relationship_count,
        "distinct_professors": distinct_professors,
    }


STRUCTURED_SCHEMA_CONSTRAINT_QUERIES = (
    "CREATE CONSTRAINT professor_professor_id IF NOT EXISTS FOR (n:Professor) REQUIRE n.professor_id IS UNIQUE",
    "CREATE CONSTRAINT organization_org_id IF NOT EXISTS FOR (n:Organization) REQUIRE n.org_id IS UNIQUE",
    "CREATE CONSTRAINT research_topic_topic_id IF NOT EXISTS FOR (n:ResearchTopic) REQUIRE n.topic_id IS UNIQUE",
    "CREATE CONSTRAINT education_experience_id IF NOT EXISTS FOR (n:EducationExperience) REQUIRE n.experience_id IS UNIQUE",
    "CREATE CONSTRAINT employment_experience_id IF NOT EXISTS FOR (n:EmploymentExperience) REQUIRE n.experience_id IS UNIQUE",
    "CREATE CONSTRAINT academic_service_role_id IF NOT EXISTS FOR (n:AcademicServiceRole) REQUIRE n.service_id IS UNIQUE",
    "CREATE CONSTRAINT award_award_id IF NOT EXISTS FOR (n:Award) REQUIRE n.award_id IS UNIQUE",
    "CREATE CONSTRAINT publication_publication_id IF NOT EXISTS FOR (n:Publication) REQUIRE n.publication_id IS UNIQUE",
)

STRUCTURED_LABELS = (
    "Professor",
    "Organization",
    "ResearchTopic",
    "EducationExperience",
    "EmploymentExperience",
    "AcademicServiceRole",
    "Award",
    "Publication",
)


def _stable_lookup_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _normalize_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _normalize_string_list(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _resolve_source_path(project_root: Path, source_file: str) -> Path:
    source_path = Path(source_file)
    if source_path.is_absolute():
        return source_path
    return project_root / source_path


def _prepare_organization_payloads(
    review: StructuredProfessorReview,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    canonical_orgs: dict[str, dict[str, Any]] = {}
    alias_to_key: dict[str, str] = {}
    for organization in review.organizations:
        aliases = _normalize_string_list([organization.name, *organization.aliases])
        alias_keys = sorted(
            {
                alias_key
                for alias_key in (_normalize_lookup_key(alias) for alias in aliases)
                if alias_key
            }
        )
        canonical_key = alias_keys[0] if alias_keys else _normalize_lookup_key(organization.name)
        if not canonical_key:
            continue
        current = canonical_orgs.get(canonical_key)
        if current is None:
            current = {
                "org_id": organization.org_id or _stable_lookup_id("org", canonical_key),
                "name": organization.name,
                "aliases": aliases,
                "org_type": organization.org_type,
                "normalized_name": canonical_key,
            }
        else:
            current["aliases"] = _normalize_string_list([*current["aliases"], *aliases])
            if current["org_type"] == "unknown" and organization.org_type != "unknown":
                current["org_type"] = organization.org_type
        canonical_orgs[canonical_key] = current
        for alias_key in alias_keys:
            alias_to_key[alias_key] = canonical_key
    return canonical_orgs, alias_to_key


def _resolve_organization_payload(
    raw_name: str,
    *,
    canonical_orgs: dict[str, dict[str, Any]],
    alias_to_key: dict[str, str],
) -> dict[str, Any] | None:
    cleaned_name = re.sub(r"\s+", " ", raw_name).strip()
    if not cleaned_name:
        return None
    lookup_key = _normalize_lookup_key(cleaned_name)
    if not lookup_key:
        return None
    canonical_key = alias_to_key.get(lookup_key)
    if canonical_key is None:
        for known_key in canonical_orgs:
            if lookup_key == known_key or lookup_key in known_key or known_key in lookup_key:
                canonical_key = known_key
                break
    if canonical_key is None:
        canonical_key = lookup_key
        canonical_orgs[canonical_key] = {
            "org_id": _stable_lookup_id("org", canonical_key),
            "name": cleaned_name,
            "aliases": [cleaned_name],
            "org_type": "unknown",
            "normalized_name": canonical_key,
        }
        alias_to_key[lookup_key] = canonical_key
    return canonical_orgs[canonical_key]


def _compile_structured_review_payload(
    *,
    review: StructuredProfessorReview,
    project_root: Path,
) -> dict[str, Any]:
    canonical_orgs, alias_to_key = _prepare_organization_payloads(review)
    organization_payloads: dict[str, dict[str, Any]] = {}

    def resolve_org_id(raw_name: str) -> str | None:
        payload = _resolve_organization_payload(
            raw_name,
            canonical_orgs=canonical_orgs,
            alias_to_key=alias_to_key,
        )
        if payload is None:
            return None
        organization_payloads[payload["org_id"]] = payload
        return payload["org_id"]

    affiliation_org_ids = [
        org_id
        for org_id in (
            resolve_org_id(name) for name in review.affiliation_organization_names
        )
        if org_id is not None
    ]

    education_experiences = []
    for record in review.education_experiences:
        education_experiences.append(
            {
                "experience_id": record.experience_id,
                "degree": record.degree,
                "field": record.field,
                "organization_name_raw": record.organization_name_raw,
                "start_text": record.start_text,
                "end_text": record.end_text,
                "is_current": record.is_current,
                "order": record.order,
                "raw_text": record.raw_text,
                "organization_org_id": resolve_org_id(record.organization_name_raw),
            }
        )

    employment_experiences = []
    for record in review.employment_experiences:
        employment_experiences.append(
            {
                "experience_id": record.experience_id,
                "role_title": record.role_title,
                "organization_name_raw": record.organization_name_raw,
                "location": record.location,
                "start_text": record.start_text,
                "end_text": record.end_text,
                "is_current": record.is_current,
                "order": record.order,
                "raw_text": record.raw_text,
                "organization_org_id": resolve_org_id(record.organization_name_raw),
            }
        )

    academic_service_roles = []
    for record in review.academic_service_roles:
        academic_service_roles.append(
            {
                "service_id": record.service_id,
                "role_title": record.role_title,
                "organization_name_raw": record.organization_name_raw,
                "service_type": record.service_type,
                "raw_text": record.raw_text,
                "organization_org_id": resolve_org_id(record.organization_name_raw),
            }
        )

    awards = [
        {
            "award_id": record.award_id,
            "name": record.name,
            "category": record.category,
            "year": record.year,
            "granting_org_name_raw": record.granting_org_name_raw,
            "level": record.level,
            "raw_text": record.raw_text,
        }
        for record in review.awards
    ]

    publications = [
        {
            "publication_id": record.publication_id,
            "title": record.title,
            "authors_raw": record.authors_raw,
            "year": record.year,
            "venue": record.venue,
            "publication_type": record.publication_type,
            "doi_or_isbn": record.doi_or_isbn,
            "raw_text": record.raw_text,
        }
        for record in review.publications
    ]

    topic_payloads: dict[str, dict[str, Any]] = {}
    for topic in review.research_topics:
        topic_payloads[topic.topic_id] = {
            "topic_id": topic.topic_id,
            "name": topic.name,
            "normalized_name": topic.normalized_name,
            "language": topic.language,
        }

    raw_ocr_markdown = _resolve_source_path(project_root, review.professor.source_file).read_text(
        encoding="utf-8"
    )

    return {
        "professor": {
            "professor_id": review.professor.professor_id,
            "name_local": review.professor.name_local,
            "name_english": review.professor.name_english,
            "aliases": list(review.professor.aliases),
            "title": review.professor.title,
            "emails": list(review.professor.emails),
            "phones": list(review.professor.phones),
            "biography_text": review.professor.biography_text,
            "source_file": review.professor.source_file,
            "detail_url": review.metadata.detail_url,
            "slug": review.metadata.slug,
            "raw_ocr_markdown": raw_ocr_markdown,
        },
        "organizations": sorted(
            organization_payloads.values(),
            key=lambda item: item["org_id"],
        ),
        "affiliation_org_ids": _normalize_string_list(affiliation_org_ids),
        "research_topics": sorted(topic_payloads.values(), key=lambda item: item["topic_id"]),
        "education_experiences": education_experiences,
        "employment_experiences": employment_experiences,
        "academic_service_roles": academic_service_roles,
        "awards": awards,
        "publications": publications,
    }


def _ensure_structured_constraints(neo4j_session: Any) -> None:
    for query in STRUCTURED_SCHEMA_CONSTRAINT_QUERIES:
        neo4j_session.run(query)


def _upsert_structured_professor(tx: Any, payload: dict[str, Any]) -> None:
    professor = payload["professor"]
    tx.run(
        """
        MERGE (p:Professor {professor_id: $professor.professor_id})
        SET p.name_local = $professor.name_local,
            p.name_english = $professor.name_english,
            p.aliases = $professor.aliases,
            p.title = $professor.title,
            p.emails = $professor.emails,
            p.phones = $professor.phones,
            p.biography_text = $professor.biography_text,
            p.source_file = $professor.source_file,
            p.detail_url = $professor.detail_url,
            p.slug = $professor.slug,
            p.raw_ocr_markdown = $professor.raw_ocr_markdown
        """,
        professor=professor,
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})-[:HAS_EDUCATION|HELD_ROLE|HAS_SERVICE_ROLE|RECEIVED|AUTHORED]->(owned)
        DETACH DELETE owned
        """,
        professor_id=professor["professor_id"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})-[r:AFFILIATED_WITH|HAS_RESEARCH_INTEREST]->()
        DELETE r
        """,
        professor_id=professor["professor_id"],
    )
    tx.run(
        """
        UNWIND $organizations AS org
        MERGE (o:Organization {org_id: org.org_id})
        SET o.aliases = reduce(
                acc = coalesce(o.aliases, []),
                alias IN org.aliases |
                CASE WHEN alias IN acc THEN acc ELSE acc + alias END
            ),
            o.normalized_name = org.normalized_name,
            o.org_type =
                CASE
                    WHEN o.org_type IS NULL OR o.org_type = 'unknown' THEN org.org_type
                    ELSE o.org_type
                END,
            o.name =
                CASE
                    WHEN o.name IS NULL OR size(o.name) = 0 THEN org.name
                    ELSE o.name
                END
        """,
        organizations=payload["organizations"],
    )
    tx.run(
        """
        UNWIND $research_topics AS topic
        MERGE (t:ResearchTopic {topic_id: topic.topic_id})
        SET t.name = topic.name,
            t.normalized_name = topic.normalized_name,
            t.language = topic.language
        """,
        research_topics=payload["research_topics"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $affiliation_org_ids AS org_id
        MATCH (o:Organization {org_id: org_id})
        MERGE (p)-[:AFFILIATED_WITH]->(o)
        """,
        professor_id=professor["professor_id"],
        affiliation_org_ids=payload["affiliation_org_ids"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $research_topics AS topic
        MATCH (t:ResearchTopic {topic_id: topic.topic_id})
        MERGE (p)-[:HAS_RESEARCH_INTEREST]->(t)
        """,
        professor_id=professor["professor_id"],
        research_topics=payload["research_topics"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $education_experiences AS item
        MERGE (e:EducationExperience {experience_id: item.experience_id})
        SET e.degree = item.degree,
            e.field = item.field,
            e.organization_name_raw = item.organization_name_raw,
            e.start_text = item.start_text,
            e.end_text = item.end_text,
            e.is_current = item.is_current,
            e.order = item.order,
            e.raw_text = item.raw_text
        MERGE (p)-[:HAS_EDUCATION]->(e)
        FOREACH (_ IN CASE WHEN item.organization_org_id IS NULL THEN [] ELSE [1] END |
            MERGE (o:Organization {org_id: item.organization_org_id})
            MERGE (e)-[:AT]->(o)
        )
        """,
        professor_id=professor["professor_id"],
        education_experiences=payload["education_experiences"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $employment_experiences AS item
        MERGE (e:EmploymentExperience {experience_id: item.experience_id})
        SET e.role_title = item.role_title,
            e.organization_name_raw = item.organization_name_raw,
            e.location = item.location,
            e.start_text = item.start_text,
            e.end_text = item.end_text,
            e.is_current = item.is_current,
            e.order = item.order,
            e.raw_text = item.raw_text
        MERGE (p)-[:HELD_ROLE]->(e)
        FOREACH (_ IN CASE WHEN item.organization_org_id IS NULL THEN [] ELSE [1] END |
            MERGE (o:Organization {org_id: item.organization_org_id})
            MERGE (e)-[:AT]->(o)
        )
        """,
        professor_id=professor["professor_id"],
        employment_experiences=payload["employment_experiences"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $academic_service_roles AS item
        MERGE (s:AcademicServiceRole {service_id: item.service_id})
        SET s.role_title = item.role_title,
            s.organization_name_raw = item.organization_name_raw,
            s.service_type = item.service_type,
            s.raw_text = item.raw_text
        MERGE (p)-[:HAS_SERVICE_ROLE]->(s)
        FOREACH (_ IN CASE WHEN item.organization_org_id IS NULL THEN [] ELSE [1] END |
            MERGE (o:Organization {org_id: item.organization_org_id})
            MERGE (s)-[:AT]->(o)
        )
        """,
        professor_id=professor["professor_id"],
        academic_service_roles=payload["academic_service_roles"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $awards AS item
        MERGE (a:Award {award_id: item.award_id})
        SET a.name = item.name,
            a.category = item.category,
            a.year = item.year,
            a.granting_org_name_raw = item.granting_org_name_raw,
            a.level = item.level,
            a.raw_text = item.raw_text
        MERGE (p)-[:RECEIVED]->(a)
        """,
        professor_id=professor["professor_id"],
        awards=payload["awards"],
    )
    tx.run(
        """
        MATCH (p:Professor {professor_id: $professor_id})
        UNWIND $publications AS item
        MERGE (pub:Publication {publication_id: item.publication_id})
        SET pub.title = item.title,
            pub.authors_raw = item.authors_raw,
            pub.year = item.year,
            pub.venue = item.venue,
            pub.publication_type = item.publication_type,
            pub.doi_or_isbn = item.doi_or_isbn,
            pub.raw_text = item.raw_text
        MERGE (p)-[:AUTHORED]->(pub)
        """,
        professor_id=professor["professor_id"],
        publications=payload["publications"],
    )


def _cleanup_structured_orphans(neo4j_session: Any) -> None:
    neo4j_session.run("MATCH (n:Organization) WHERE NOT (n)--() DELETE n")
    neo4j_session.run("MATCH (n:ResearchTopic) WHERE NOT (n)--() DELETE n")


def verify_structured_neo4j_graph(settings: TutorSettings) -> dict[str, Any]:
    settings.require_neo4j()
    neo4j_driver = require_neo4j_driver()
    driver = neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    with driver.session(database=settings.neo4j_database) as neo4j_session:
        node_count = neo4j_session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        relationship_count = neo4j_session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
        label_counts = {
            label: neo4j_session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for label in STRUCTURED_LABELS
        }
    driver.close()
    return {
        "node_count": node_count,
        "relationship_count": relationship_count,
        "label_counts": label_counts,
    }


def insert_structured_seed_to_neo4j(
    *,
    settings: TutorSettings,
    project_root: Path | None = None,
    seed_dir: Path | None = None,
    only_slugs: list[str] | None = None,
    limit: int | None = None,
    reset_database: bool = False,
) -> StructuredSeedInsertionResult:
    resolved_project_root = project_root or settings.project_root
    settings.require_neo4j()
    structured_reviews = load_structured_seed_reviews(
        project_root=resolved_project_root,
        seed_dir=seed_dir,
        only_slugs=only_slugs,
        limit=limit,
    )
    resolved_seed_dir = resolve_structured_seed_dir(resolved_project_root, seed_dir)

    neo4j_driver = require_neo4j_driver()
    driver = neo4j_driver.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    if reset_database:
        reset_neo4j_database(settings)

    inserted_professors: list[str] = []
    failures: list[dict[str, str]] = []
    with driver.session(database=settings.neo4j_database) as neo4j_session:
        _ensure_structured_constraints(neo4j_session)
        for _, review in structured_reviews:
            try:
                payload = _compile_structured_review_payload(
                    review=review,
                    project_root=resolved_project_root,
                )
                with neo4j_session.begin_transaction() as tx:
                    _upsert_structured_professor(tx, payload)
                    tx.commit()
                inserted_professors.append(review.metadata.slug)
            except Exception as exc:
                failures.append({"slug": review.metadata.slug, "error": str(exc)})
        _cleanup_structured_orphans(neo4j_session)
    driver.close()

    verification = verify_structured_neo4j_graph(settings)
    return StructuredSeedInsertionResult(
        seed_dir=str(resolved_seed_dir.relative_to(resolved_project_root)),
        professor_count=len(structured_reviews),
        success_count=len(inserted_professors),
        failure_count=len(failures),
        failures=failures,
        inserted_professors=sorted(inserted_professors),
        node_count=verification["node_count"],
        relationship_count=verification["relationship_count"],
        label_counts=verification["label_counts"],
    )
