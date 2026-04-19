from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from neo4j import GraphDatabase

from .config import TutorSettings


def _normalize_keywords(keywords: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for keyword in keywords:
        value = keyword.strip().lower()
        if not value or value in cleaned:
            continue
        cleaned.append(value)
    return cleaned


@dataclass(frozen=True)
class QueryTrace:
    name: str
    cypher: str
    params: dict[str, Any]
    row_count: int
    preview: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfessorMatch:
    professor_name: str
    matched_name: str
    detail_url: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfessorFact:
    source_name: str
    target_name: str
    relationship_type: str
    predicate: str
    source_professors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopicMatch:
    professor_name: str
    detail_url: str | None
    matched_nodes: list[str]
    matched_predicates: list[str]
    match_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Neo4jQueryService:
    """Deterministic query helpers for the typed Lab 3 professor graph."""

    def __init__(self, settings: TutorSettings) -> None:
        self.settings = settings

    def _run(
        self,
        name: str,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], QueryTrace]:
        params = params or {}
        driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password),
        )
        try:
            with driver.session(database=self.settings.neo4j_database) as session:
                rows = session.run(cypher, params).data()
        finally:
            driver.close()

        trace = QueryTrace(
            name=name,
            cypher=cypher.strip(),
            params=params,
            row_count=len(rows),
            preview=rows[:5],
        )
        return rows, trace

    def resolve_professor(self, name_hint: str, *, limit: int = 5) -> tuple[list[ProfessorMatch], QueryTrace]:
        normalized_hint = name_hint.strip().lower()
        rows, trace = self._run(
            "resolve_professor",
            """
            MATCH (p:Professor)
            WITH p,
                 [value IN ([coalesce(p.name, '')] + coalesce(p.aliases, [])) WHERE trim(value) <> ''] AS candidate_names
            WITH p, candidate_names,
                 [value IN candidate_names WHERE
                    toLower(value) = $needle
                    OR toLower(value) CONTAINS $needle
                 ] AS matched_names
            WHERE size(matched_names) > 0
            RETURN
              p.name AS professor_name,
              coalesce(head(matched_names), p.name) AS matched_name,
              p.detail_url AS detail_url,
              CASE
                WHEN any(value IN candidate_names WHERE toLower(value) = $needle) THEN 0
                WHEN any(value IN candidate_names WHERE toLower(value) STARTS WITH $needle) THEN 1
                ELSE 2
              END AS score
            ORDER BY score, professor_name
            LIMIT $limit
            """,
            {"needle": normalized_hint, "limit": limit},
        )
        matches = [
            ProfessorMatch(
                professor_name=row["professor_name"],
                matched_name=row["matched_name"],
                detail_url=row["detail_url"],
                score=row["score"],
            )
            for row in rows
        ]
        return matches, trace

    def get_professor_facts(
        self,
        professor_name: str,
        *,
        keywords: Sequence[str] | None = None,
        limit: int = 18,
    ) -> tuple[list[ProfessorFact], QueryTrace]:
        normalized_keywords = _normalize_keywords(keywords or [])
        rows, trace = self._run(
            "get_professor_facts",
            """
            MATCH (p:Professor)
            WHERE p.name = $professor_name OR $professor_name IN coalesce(p.aliases, [])
            WITH p
            CALL (p) {
              MATCH (p)-[:AFFILIATED_WITH]->(o:Organization)
              RETURN p.name AS professor_name, p.name AS source_name, o.name AS target_name, 'AFFILIATED_WITH' AS relationship_type
              UNION ALL
              MATCH (p)-[:HAS_RESEARCH_INTEREST]->(t:ResearchTopic)
              RETURN p.name AS professor_name, p.name AS source_name, t.name AS target_name, 'HAS_RESEARCH_INTEREST' AS relationship_type
              UNION ALL
              MATCH (p)-[:HAS_EXPERIENCE]->(e:Experience)
              RETURN
                p.name AS professor_name,
                p.name AS source_name,
                CASE
                  WHEN coalesce(e.raw_text, '') <> '' THEN e.raw_text
                  WHEN coalesce(e.title, '') <> '' THEN e.title
                  WHEN coalesce(e.degree, '') <> '' THEN e.degree
                  WHEN coalesce(e.organization_name, '') <> '' THEN e.organization_name
                  ELSE coalesce(e.field, '')
                END AS target_name,
                'HAS_EXPERIENCE' AS relationship_type
              UNION ALL
              MATCH (p)-[:AUTHORED]->(pub:Publication)
              RETURN
                p.name AS professor_name,
                p.name AS source_name,
                CASE
                  WHEN coalesce(pub.title, '') <> '' THEN pub.title
                  ELSE coalesce(pub.raw_text, '')
                END AS target_name,
                'AUTHORED' AS relationship_type
              UNION ALL
              MATCH (p)-[:RECEIVED]->(a:Award)
              RETURN
                p.name AS professor_name,
                p.name AS source_name,
                CASE
                  WHEN coalesce(a.name, '') <> '' THEN a.name
                  ELSE coalesce(a.raw_text, '')
                END AS target_name,
                'RECEIVED' AS relationship_type
              UNION ALL
              UNWIND coalesce(p.emails, []) AS value
              RETURN p.name AS professor_name, p.name AS source_name, value AS target_name, 'HAS_EMAIL' AS relationship_type
              UNION ALL
              UNWIND coalesce(p.phones, []) AS value
              RETURN p.name AS professor_name, p.name AS source_name, value AS target_name, 'HAS_PHONE' AS relationship_type
              UNION ALL
              UNWIND coalesce(p.websites, []) AS value
              RETURN p.name AS professor_name, p.name AS source_name, value AS target_name, 'HAS_WEBSITE' AS relationship_type
              UNION ALL
              UNWIND coalesce(p.locations, []) AS value
              RETURN p.name AS professor_name, p.name AS source_name, value AS target_name, 'LOCATED_IN' AS relationship_type
              UNION ALL
              WITH p WHERE coalesce(p.title, '') <> ''
              RETURN p.name AS professor_name, p.name AS source_name, p.title AS target_name, 'HAS_TITLE' AS relationship_type
              UNION ALL
              WITH p WHERE coalesce(p.discipline, '') <> ''
              RETURN p.name AS professor_name, p.name AS source_name, p.discipline AS target_name, 'IN_DISCIPLINE' AS relationship_type
            }
            WITH professor_name, source_name, target_name, relationship_type
            WHERE target_name IS NOT NULL
              AND trim(target_name) <> ''
              AND (
                size($keywords) = 0
                OR any(keyword IN $keywords WHERE
                    toLower(source_name) CONTAINS keyword
                    OR toLower(target_name) CONTAINS keyword
                    OR toLower(relationship_type) CONTAINS keyword
                )
              )
            RETURN
              source_name,
              target_name,
              relationship_type,
              relationship_type AS predicate,
              [professor_name] AS source_professors
            ORDER BY predicate, target_name
            LIMIT $limit
            """,
            {
                "professor_name": professor_name,
                "keywords": normalized_keywords,
                "limit": limit,
            },
        )
        facts = [
            ProfessorFact(
                source_name=row["source_name"],
                target_name=row["target_name"],
                relationship_type=row["relationship_type"],
                predicate=row["predicate"],
                source_professors=list(row["source_professors"] or []),
            )
            for row in rows
        ]
        return facts, trace

    def find_professors_by_topics(
        self,
        keywords: Sequence[str],
        *,
        limit: int = 5,
    ) -> tuple[list[TopicMatch], QueryTrace]:
        normalized_keywords = _normalize_keywords(keywords)
        rows, trace = self._run(
            "find_professors_by_topics",
            """
            MATCH (p:Professor)-[:HAS_RESEARCH_INTEREST]->(t:ResearchTopic)
            WITH p, t,
                 [keyword IN $keywords WHERE
                    toLower(t.name) CONTAINS keyword
                    OR toLower(coalesce(t.normalized_name, '')) CONTAINS keyword
                 ] AS hits
            WHERE size(hits) > 0
            RETURN
              p.name AS professor_name,
              p.detail_url AS detail_url,
              collect(DISTINCT t.name)[0..5] AS matched_nodes,
              ['HAS_RESEARCH_INTEREST'] AS matched_predicates,
              count(DISTINCT t) AS match_count
            ORDER BY match_count DESC, professor_name
            LIMIT $limit
            """,
            {"keywords": normalized_keywords, "limit": limit},
        )
        matches = [
            TopicMatch(
                professor_name=row["professor_name"],
                detail_url=row["detail_url"],
                matched_nodes=list(row["matched_nodes"] or []),
                matched_predicates=list(row["matched_predicates"] or []),
                match_count=row["match_count"],
            )
            for row in rows
        ]
        return matches, trace


__all__ = [
    "Neo4jQueryService",
    "ProfessorFact",
    "ProfessorMatch",
    "QueryTrace",
    "TopicMatch",
]
