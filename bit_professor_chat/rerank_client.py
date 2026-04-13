from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import requests


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float
    document: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "relevance_score": self.relevance_score,
            "document": self.document,
        }


class JinaStyleReranker:
    """Small HTTP adapter for OpenAI/Jina-style rerank APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        requested_top_n = max(1, min(top_n or len(documents), len(documents)))
        response = self._session.post(
            self.base_url,
            json={
                "model": self.model,
                "query": query,
                "documents": list(documents),
                "top_n": requested_top_n,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results") or payload.get("data") or []

        ranked: list[RerankResult] = []
        for row in rows:
            index = row.get("index")
            if index is None:
                index = row.get("document_index")
            if index is None:
                continue

            try:
                resolved_index = int(index)
            except (TypeError, ValueError):
                continue
            if resolved_index < 0 or resolved_index >= len(documents):
                continue

            raw_score = row.get("relevance_score")
            if raw_score is None:
                raw_score = row.get("score")
            if raw_score is None:
                raw_score = row.get("rerank_score")
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0

            document_value = documents[resolved_index]
            row_document = row.get("document")
            if isinstance(row_document, dict):
                document_value = str(
                    row_document.get("text")
                    or row_document.get("content")
                    or document_value
                )
            elif isinstance(row_document, str):
                document_value = row_document

            ranked.append(
                RerankResult(
                    index=resolved_index,
                    relevance_score=score,
                    document=document_value,
                )
            )

        ranked.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranked[:requested_top_n]


__all__ = ["JinaStyleReranker", "RerankResult"]
