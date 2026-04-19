from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def discover_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upward until a repo marker is found."""

    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path(__file__).resolve().parents[1])
    candidates.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        for current in (candidate, *candidate.parents):
            if current in seen:
                continue
            seen.add(current)
            if (current / "pyproject.toml").exists() or (current / ".env").exists():
                return current
    return Path.cwd().resolve()


def _resolve_mcp_command(project_root: Path) -> str:
    """Prefer the project's uv-managed binary, then fall back to PATH."""

    quiet_wrapper = project_root / "bit_professor_chat" / "mcp_neo4j_quiet.sh"
    if quiet_wrapper.exists():
        return str(quiet_wrapper.resolve())

    local_binary = project_root / ".venv" / "bin" / "mcp-neo4j-cypher"
    if local_binary.exists():
        return str(local_binary.resolve())

    system_binary = shutil.which("mcp-neo4j-cypher")
    if system_binary:
        return system_binary

    return "mcp-neo4j-cypher"


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _resolve_api_base(base_url: str | None, endpoint: str) -> str:
    if not base_url:
        return endpoint

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}{endpoint[3:]}"
    return f"{normalized}{endpoint}"


@dataclass(frozen=True)
class TutorSettings:
    project_root: Path
    dotenv_path: Path
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    ocr_api_key: str | None = None
    ocr_base_url: str | None = None
    ocr_model: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_dims: int | None = None
    embedding_batch_size: int = 10
    rerank_api_key: str | None = None
    rerank_base_url: str | None = None
    rerank_model: str | None = None
    vector_dir: str | None = None
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = "neo4j"
    neo4j_mcp_command: str | None = "mcp-neo4j-cypher"
    schema_sample_size: int = 200

    @classmethod
    def from_env(
        cls,
        dotenv_path: Path | None = None,
        *,
        require_llm: bool = True,
    ) -> "TutorSettings":
        project_root = discover_project_root(dotenv_path.parent if dotenv_path else None)
        resolved_dotenv = dotenv_path.resolve() if dotenv_path else project_root / ".env"
        load_dotenv(resolved_dotenv, override=False)

        llm_api_key = _first_env("BIT_PROF_LLM_API_KEY", "LAB_TUTOR_LLM_API_KEY")
        llm_base_url = _first_env("BIT_PROF_LLM_BASE_URL", "LAB_TUTOR_LLM_BASE_URL")
        llm_model = _first_env("BIT_PROF_LLM_MODEL", "LAB_TUTOR_LLM_MODEL")
        missing = [
            name
            for name, value in {
                "BIT_PROF_LLM_API_KEY": llm_api_key,
                "BIT_PROF_LLM_BASE_URL": llm_base_url,
                "BIT_PROF_LLM_MODEL": llm_model,
            }.items()
            if not value
        ]
        if missing and require_llm:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required environment variables: {missing_list}")

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if any([neo4j_uri, neo4j_username, neo4j_password]) and not all(
            [neo4j_uri, neo4j_username, neo4j_password]
        ):
            raise ValueError(
                "Neo4j configuration is partial. Set NEO4J_URI, NEO4J_USERNAME, and "
                "NEO4J_PASSWORD together or omit all three."
            )

        embedding_dims = _first_env("BIT_PROF_EMBEDDING_DIMS", "LAB_TUTOR_EMBEDDING_DIMS")
        embedding_batch_size = _first_env(
            "BIT_PROF_EMBEDDING_BATCH_SIZE", "LAB_TUTOR_EMBEDDING_BATCH_SIZE"
        )

        return cls(
            project_root=project_root,
            dotenv_path=resolved_dotenv,
            llm_api_key=llm_api_key or "",
            llm_base_url=llm_base_url or "",
            llm_model=llm_model or "",
            ocr_api_key=(
                _first_env("BIT_PROF_OCR_API_KEY", "LAB_TUTOR_LLM_API_KEY") or llm_api_key
            ),
            ocr_base_url=(
                _first_env("BIT_PROF_OCR_BASE_URL", "LAB_TUTOR_LLM_BASE_URL") or llm_base_url
            ),
            ocr_model=(
                _first_env("BIT_PROF_OCR_MODEL", "LAB_TUTOR_OCR_MODEL")
                or "qwen-vl-ocr-latest"
            ),
            embedding_api_key=(
                _first_env("BIT_PROF_EMBEDDING_API_KEY", "LAB_TUTOR_EMBEDDING_API_KEY")
                or llm_api_key
            ),
            embedding_base_url=(
                _first_env("BIT_PROF_EMBEDDING_BASE_URL", "LAB_TUTOR_EMBEDDING_BASE_URL")
                or llm_base_url
            ),
            embedding_model=(
                _first_env("BIT_PROF_EMBEDDING_MODEL", "LAB_TUTOR_EMBEDDING_MODEL")
                or "text-embedding-v4"
            ),
            embedding_dims=int(embedding_dims) if embedding_dims else 2048,
            embedding_batch_size=int(embedding_batch_size or "10"),
            rerank_api_key=(
                _first_env("BIT_PROF_RERANK_API_KEY", "LAB_TUTOR_RERANK_API_KEY")
                or llm_api_key
            ),
            rerank_base_url=(
                _first_env("BIT_PROF_RERANK_BASE_URL", "LAB_TUTOR_RERANK_BASE_URL")
                or llm_base_url
            ),
            rerank_model=(
                _first_env("BIT_PROF_RERANK_MODEL", "LAB_TUTOR_RERANK_MODEL")
                or "bge-reranker-v2-m3"
            ),
            vector_dir=_first_env("BIT_PROF_VECTOR_DIR", "LAB_TUTOR_VECTOR_DIR"),
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j") if neo4j_uri else None,
            neo4j_mcp_command=_resolve_mcp_command(project_root) if neo4j_uri else None,
            schema_sample_size=int(os.getenv("NEO4J_MCP_SCHEMA_SAMPLE_SIZE", "200")),
        )

    @property
    def lab_tutor_llm_api_key(self) -> str:
        return self.llm_api_key

    @property
    def lab_tutor_llm_base_url(self) -> str:
        return self.llm_base_url

    @property
    def lab_tutor_llm_model(self) -> str:
        return self.llm_model

    @property
    def chroma_dir(self) -> Path:
        configured = self.vector_dir or str(self.project_root / "artifacts" / "lab2" / "chroma")
        path = Path(configured)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def corpus_artifact_dir(self) -> Path:
        return self.project_root / "artifacts" / "lab2"

    @property
    def structured_output_dir(self) -> Path:
        return self.project_root / "lab_3_langgraph_swarm" / "structured_output"

    @property
    def corpus_index_path(self) -> Path:
        return self.corpus_artifact_dir / "professor_index.json"

    @property
    def retrieval_manifest_path(self) -> Path:
        return self.corpus_artifact_dir / "retrieval_manifest.json"

    def require_neo4j(self) -> None:
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise ValueError(
                "Neo4j settings are not configured. Lab 1 requires NEO4J_URI, "
                "NEO4J_USERNAME, and NEO4J_PASSWORD."
            )

    def require_embeddings(self) -> dict[str, Any]:
        missing = [
            name
            for name, value in {
                "BIT_PROF_EMBEDDING_API_KEY": self.embedding_api_key,
                "BIT_PROF_EMBEDDING_BASE_URL": self.embedding_base_url,
                "BIT_PROF_EMBEDDING_MODEL": self.embedding_model,
            }.items()
            if not value
        ]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                "Embedding configuration is required for Lab 2 retrieval. "
                f"Missing: {missing_list}"
            )
        return {
            "api_key": self.embedding_api_key,
            "base_url": self.embedding_base_url,
            "model": self.embedding_model,
            "dimensions": self.embedding_dims,
            "batch_size": self.embedding_batch_size,
        }

    def require_ocr(self) -> dict[str, Any]:
        missing = [
            name
            for name, value in {
                "BIT_PROF_OCR_API_KEY": self.ocr_api_key,
                "BIT_PROF_OCR_BASE_URL": self.ocr_base_url,
                "BIT_PROF_OCR_MODEL": self.ocr_model,
            }.items()
            if not value
        ]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                "OCR configuration is required for instructor corpus prep. "
                f"Missing: {missing_list}"
            )
        return {
            "api_key": self.ocr_api_key,
            "base_url": self.ocr_base_url,
            "model": self.ocr_model,
        }

    def require_reranker(self) -> dict[str, Any]:
        missing = [
            name
            for name, value in {
                "BIT_PROF_RERANK_API_KEY": self.rerank_api_key,
                "BIT_PROF_RERANK_BASE_URL": self.rerank_base_url,
                "BIT_PROF_RERANK_MODEL": self.rerank_model,
            }.items()
            if not value
        ]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                "Reranker configuration is required for Lab 2 topic matching. "
                f"Missing: {missing_list}"
            )
        return {
            "api_key": self.rerank_api_key,
            "base_url": self.rerank_base_url,
            "model": self.rerank_model,
        }

    def rerank_endpoint(self) -> str:
        return _resolve_api_base(self.rerank_base_url, "/v1/rerank")

    def neo4j_mcp_connection(self) -> dict[str, object]:
        """Return the stdio MCP connection config LangChain expects."""

        self.require_neo4j()
        return {
            "transport": "stdio",
            "command": self.neo4j_mcp_command,
            "env": {
                "FASTMCP_SHOW_CLI_BANNER": "false",
                "FASTMCP_CHECK_FOR_UPDATES": "off",
            },
            "args": [
                "--db-url",
                self.neo4j_uri,
                "--username",
                self.neo4j_username,
                "--password",
                self.neo4j_password,
                "--database",
                self.neo4j_database,
                "--transport",
                "stdio",
                "--read-only",
                "--schema-sample-size",
                str(self.schema_sample_size),
            ],
        }
