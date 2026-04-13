# multi_agents_hands_on

Hands-on teaching materials for building multi-agent systems with LangChain and LangGraph.

## What is included

- `lab_1_langchain_pipeline/`: notebook-first Lab 1 materials and workflow assets
- `lab_2_langgraph_workflow/`: notebook-first Lab 2 materials and workflow assets
- `lab_3_langgraph_swarm/`: swarm-oriented LangGraph lab notebook and diagrams
- `lab_4_deep_agents/`: deep-agents lab notebook, workflow assets, and supporting skills
- `bit_professor_chat/`: reusable Python package for ingestion, OCR/model helpers, Neo4j queries, and the MCP chat agent
- `professors/`: source professor dossiers used by the labs
- `docker-compose.yml`: local Neo4j service definition

## Quick Start

1. Install the Python environment:

```bash
uv sync
```

2. Copy the example environment file and fill in secrets locally:

```bash
cp .env.example .env
```

3. Start Neo4j:

```bash
docker compose up -d neo4j
```

4. Launch Jupyter Lab:

```bash
uv run jupyter lab
```

## Useful Commands

```bash
uv run python -m bit_professor_chat.mcp_agent "Who works on NLP?" --show-trace
uv run python lab_2_langgraph_workflow/prepare_lab2_corpus.py --limit 5
uv run python lab_1_langchain_pipeline/workflow_render.py
```

## Notes

- `.env` is intentionally not committed.
- `artifacts/` contains generated outputs and is intentionally ignored.
