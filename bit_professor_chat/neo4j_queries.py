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


@dataclass(frozen=True)
class GraphOverview:
    professor_count: int
    relationship_count: int
    sample_professors: list[str]
    common_predicates: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Neo4jQueryService:
    """Deterministic query helpers for Lab 2's supervisor workflow."""

    def __init__(self, settings: TutorSettings) -> None:
        self.settings = settings

    def _run(self, name: str, cypher: str, params: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], QueryTrace]:
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

    def get_graph_overview(self, *, limit: int = 8) -> tuple[GraphOverview, list[QueryTrace]]:
        professor_count_rows, professor_trace = self._run(
            "graph_overview_professor_count",
            """
            MATCH (p:Entity)
            WHERE p.professor_name IS NOT NULL
            RETURN count(DISTINCT p.professor_name) AS professor_count
            """,
        )
        relationship_rows, relationship_trace = self._run(
            "graph_overview_relationship_count",
            """
            MATCH ()-[r]->()
            RETURN count(r) AS relationship_count
            """,
        )
        professor_rows, professors_trace = self._run(
            "graph_overview_sample_professors",
            """
            MATCH (p:Entity)
            WHERE p.professor_name IS NOT NULL
            RETURN DISTINCT p.professor_name AS professor_name
            ORDER BY professor_name
            LIMIT $limit
            """,
            {"limit": limit},
        )
        predicate_rows, predicates_trace = self._run(
            "graph_overview_common_predicates",
            """
            MATCH ()-[r]->()
            RETURN coalesce(r.predicate, type(r)) AS predicate, count(*) AS relationship_count
            ORDER BY relationship_count DESC, predicate
            LIMIT $limit
            """,
            {"limit": limit},
        )

        overview = GraphOverview(
            professor_count=professor_count_rows[0]["professor_count"] if professor_count_rows else 0,
            relationship_count=relationship_rows[0]["relationship_count"] if relationship_rows else 0,
            sample_professors=[row["professor_name"] for row in professor_rows],
            common_predicates=[row["predicate"] for row in predicate_rows],
        )
        return overview, [professor_trace, relationship_trace, professors_trace, predicates_trace]

    def resolve_professor(self, name_hint: str, *, limit: int = 5) -> tuple[list[ProfessorMatch], QueryTrace]:
        normalized_hint = name_hint.strip().lower()
        rows, trace = self._run(
            "resolve_professor",
            """
            MATCH (p:Entity)
            WHERE p.professor_name IS NOT NULL
              AND (
                toLower(p.professor_name) = $needle
                OR toLower(p.name) = $needle
                OR toLower(p.professor_name) CONTAINS $needle
                OR toLower(p.name) CONTAINS $needle
              )
            RETURN DISTINCT
              p.professor_name AS professor_name,
              p.name AS matched_name,
              p.detail_url AS detail_url,
              CASE
                WHEN toLower(p.professor_name) = $needle OR toLower(p.name) = $needle THEN 0
                WHEN toLower(p.professor_name) STARTS WITH $needle OR toLower(p.name) STARTS WITH $needle THEN 1
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
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE $professor_name IN coalesce(r.source_professors, [])
              AND (
                size($keywords) = 0
                OR any(keyword IN $keywords WHERE
                    toLower(source.name) CONTAINS keyword
                    OR toLower(target.name) CONTAINS keyword
                    OR toLower(coalesce(r.predicate, type(r))) CONTAINS keyword
                )
              )
            RETURN
              source.name AS source_name,
              target.name AS target_name,
              type(r) AS relationship_type,
              coalesce(r.predicate, type(r)) AS predicate,
              coalesce(r.source_professors, []) AS source_professors
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
            MATCH (source:Entity)-[r]->(target:Entity)
            UNWIND coalesce(r.source_professors, []) AS professor_name
            WITH professor_name, source, target, r,
                 [keyword IN $keywords WHERE
                    toLower(source.name) CONTAINS keyword
                    OR toLower(target.name) CONTAINS keyword
                    OR toLower(coalesce(r.predicate, type(r))) CONTAINS keyword
                 ] AS hits
            WHERE size(hits) > 0
            OPTIONAL MATCH (p:Entity)
            WHERE p.professor_name = professor_name
            RETURN
              professor_name,
              head(collect(DISTINCT p.detail_url)) AS detail_url,
              collect(DISTINCT target.name)[0..5] AS matched_nodes,
              collect(DISTINCT coalesce(r.predicate, type(r)))[0..5] AS matched_predicates,
              count(*) AS match_count
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
