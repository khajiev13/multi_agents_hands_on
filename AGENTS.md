# Repository Guidelines

## Project Structure & Module Organization
`bit_professor_chat/` contains the reusable Python package: config loading, ingestion, OCR/model helpers, Neo4j queries, and the MCP chat agent. `lab_1_langchain_pipeline/` and `lab_2_langgraph_workflow/` are notebook-first teaching labs with supporting render/prep scripts. `professors/` stores source markdown dossiers, while `artifacts/` stores generated outputs such as graphs, Chroma data, logs, and OCR notes. Root files worth knowing: `pyproject.toml` for dependencies, `.env.example` for local config, and `docker-compose.yml` for Neo4j.

## Build, Test, and Development Commands
Use `uv` for local setup and execution.

- `uv sync` installs the Python 3.11 environment from `pyproject.toml` and `uv.lock`.
- `docker compose up -d neo4j` starts the local Neo4j service used by Lab 1 and the MCP agent.
- `uv run jupyter lab` opens the notebook environment for both labs.
- `uv run python -m bit_professor_chat.mcp_agent "Who works on NLP?" --show-trace` runs the CLI agent against Neo4j.
- `uv run python lab_2_langgraph_workflow/prepare_lab2_corpus.py --limit 5` rebuilds a small Lab 2 corpus sample.
- `uv run python lab_1_langchain_pipeline/workflow_render.py` regenerates Lab 1 diagram assets; use the matching Lab 2 renderer when editing that workflow.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and explicit type hints on public helpers. Keep imports clean, prefer small pure functions, and reuse dataclasses for structured results. Mirror current file naming such as `graph_ingestion.py`, `ocr_transcript.py`, and `prepare_lab2_corpus.py`. No formatter or linter config is checked in, so match the surrounding style closely and keep lines readable.

## Testing Guidelines
There is no dedicated `tests/` suite in this checkout. Validate changes with targeted smoke checks: run the affected notebook cells, exercise the CLI agent, and rerun the relevant render/prep script when artifacts change. For data or graph changes, verify the generated files under `artifacts/` and confirm Neo4j connectivity before claiming completion.

## Commit & Pull Request Guidelines
This snapshot does not include `.git`, so local history is unavailable. Use short imperative commit subjects, for example `Add Lab 2 corpus refresh limit`. In pull requests, describe the affected lab or package area, list required `.env` or Neo4j changes, and call out regenerated artifacts. Include screenshots only when notebook UI or diagram output changes.

## Security & Configuration Tips
Start from `.env.example`, keep secrets in `.env`, and never commit API keys. Treat `artifacts/` as generated output unless a change explicitly updates reference data for the labs.
