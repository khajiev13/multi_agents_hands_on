# multi_agents_hands_on

Hands-on teaching materials for building multi-agent systems with LangChain and LangGraph.

![Course data flow](docs/course_data_flow.svg)

This README is written for students working through the labs. The professor dataset you need for Labs 2, 3, and 4 is already committed in the repo, so your main job is to set up the environment and load the right local data for each lab.

## Start Here

1. Install the project environment:

```bash
uv sync
```

2. Copy the example environment file:

```bash
cp .env.example .env
```

3. Launch Jupyter when you are ready to work through the notebooks:

```bash
uv run jupyter lab
```

4. Start Neo4j if you are doing Lab 3:

```bash
docker compose up -d neo4j
```

## How Each Lab Uses Data

### Lab 2

Lab 2 works from the committed professor markdown corpus:

- `professors/`
- `professors.md`

### Lab 3

Lab 3 uses a local Neo4j database. Before opening the notebook, load Neo4j from the committed structured seed files:

```bash
docker compose up -d neo4j
uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py
```

That command reads:

- `lab_3_langgraph_swarm/structured_seed/*-structured.json`
- `lab_3_langgraph_swarm/structured_seed/*-ocr.md`

Then the notebook in `lab_3_langgraph_swarm/` queries the local graph.

To confirm Neo4j is working before opening the notebook, open [http://localhost:7474](http://localhost:7474) in your browser and log in with the Neo4j credentials from `.env`. The Bolt URL should be `bolt://localhost:7687`.

Run these quick checks in the Neo4j Browser query box:

```cypher
MATCH (p:Professor) RETURN count(p)
```

```cypher
MATCH (n) RETURN count(n)
```

```cypher
MATCH ()-[r]->() RETURN count(r)
```

After loading the full Lab 3 seed set, you should see `44` professors.

### Lab 4

Lab 4 works from prepared local files in the repo:

- `professors/`
- `professors.md`
- `lab_4_deep_agents/incoming_artifacts/`
- `lab_4_deep_agents/skills/add-professor-from-incoming/`

The student Lab 4 notebook is built around that prepared local workspace.

## Repository Layout

- `lab_1_langchain_pipeline/`: notebook-first Lab 1 materials and workflow assets
- `lab_2_langgraph_workflow/`: notebook-first Lab 2 materials and workflow assets
- `lab_3_langgraph_swarm/`: swarm-oriented LangGraph lab notebook and diagrams
- `lab_4_deep_agents/`: deep-agents lab notebook, workflow assets, and supporting skills
- `bit_professor_chat/`: reusable Python package for ingestion, OCR/model helpers, Neo4j queries, and the MCP chat agent
- `professors/`: source professor dossiers used by the labs
- `professors.md`: compact index built from the committed professor dossier set
- `lab_3_langgraph_swarm/structured_seed/`: committed structured Neo4j seed files for Lab 3
- `lab_4_deep_agents/incoming_artifacts/`: prepared file inputs for the Lab 4 add-professor workflow
- `docker-compose.yml`: local Neo4j service definition

## Commands You Will Probably Use

```bash
uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py
uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py --limit 1
uv run python -m bit_professor_chat.mcp_agent "Who works on NLP?" --show-trace
uv run python lab_3_langgraph_swarm/notebook_factory.py
uv run python lab_4_deep_agents/notebook_factory.py
```

## Model Settings

For the interactive agent turns in Labs 3 and 4, fill in the chat model settings from `.env.example`.

The offline Lab 3 seed loader itself only needs Neo4j settings. It does not need LLM or OCR credentials.

## Instructor Prep

If you are preparing the course materials rather than following the labs as a student, use the live refresh flow when you want to crawl the BIT site, OCR the professor posters, build `kg-gen` review artifacts, and save schema-guided structured JSON drafts before any Neo4j insertion work:

```bash
uv run python lab_1_langchain_pipeline/prepare_lab1_graph.py --max-concurrency 8
```

A full successful instructor refresh updates:

- `artifacts/lab1-pre-insertion-review/*`

This pre-insertion phase stops after saving OCR markdown, per-page graphs, per-professor aggregates, clustered graph artifacts, structured JSON drafts, and corpus-wide aggregate/clustered review artifacts.

Once those review artifacts are approved, promote them into the committed Lab 3 student seed directory with:

```bash
uv run python lab_1_langchain_pipeline/promote_lab1_structured_seed.py --artifact-namespace lab1-pre-insertion-review
```

Targeted smoke runs can use `--only-slugs` or `--limit`, for example:

```bash
uv run python lab_1_langchain_pipeline/prepare_lab1_graph.py --only-slugs filippo-fabrocini,gao-guangyu
```

Smoke runs write to the selected artifact namespace and do not rebuild `professors.md` or overwrite the committed Lab 3 seed set.

The live URL + OCR extension for Lab 4 is documented separately in [lab_4_deep_agents/instructor_live_url_ocr_variant.md](lab_4_deep_agents/instructor_live_url_ocr_variant.md).

## Notes

- `.env` is intentionally not committed.
- `artifacts/` contains generated outputs and is intentionally ignored.
- Student-ready Lab 3 seed data lives under `lab_3_langgraph_swarm/structured_seed/`, not under `artifacts/`.
- `docs/course_data_flow.excalidraw` is the editable source for the README architecture diagram.
- `docs/course_data_flow.svg` is the rendered README diagram asset.
