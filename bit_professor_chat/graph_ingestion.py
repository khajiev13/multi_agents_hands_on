from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from kg_gen import KGGen
except ImportError:  # pragma: no cover - optional for Lab 2 markdown-only runs
    KGGen = None

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional for Lab 2 markdown-only runs
    GraphDatabase = None

from .config import TutorSettings


def require_kg_gen() -> type[KGGen]:
    if KGGen is None:
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
    if hasattr(graph, "model_dump_json"):
        output_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
        return

    payload = {
        "entities": list(getattr(graph, "entities", [])),
        "relations": list(getattr(graph, "relations", [])),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_professor_graph(markdown_text: str, settings: TutorSettings) -> Any:
    kg_class = require_kg_gen()
    kg = kg_class(
        model=f"openai/{settings.lab_tutor_llm_model}",
        api_key=settings.lab_tutor_llm_api_key,
        api_base=settings.lab_tutor_llm_base_url,
        temperature=0.0,
    )
    return kg.generate(
        input_data=markdown_text,
        context="Beijing Institute of Technology professor profile",
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
