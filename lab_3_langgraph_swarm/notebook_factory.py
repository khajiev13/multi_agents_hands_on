from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).parent
NOTEBOOK_PATH = ROOT / "lab_3_langgraph_swarm.ipynb"


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        md(
            """
            # Lab 3: Minimal Neo4j Swarm Handoffs

            **Editable diagram:** [lab_3_swarm.excalidraw](lab_3_swarm.excalidraw)  
            **Workflow preview:** [lab_3_swarm.svg](lab_3_swarm.svg)

            This lab keeps the Lab 2 question domain, but removes the supervisor. The only new idea is the swarm handoff pattern:

            - the current agent talks to the user directly,
            - handoff tools transfer control to another specialist,
            - `active_agent` persists across turns for the same `thread_id`.

            This notebook assumes Neo4j has already been seeded from the committed local dataset with
            `uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py`. We use three agents:

            - `FrontDeskAgent` has no Neo4j tools and only decides who should own the conversation.
            - `ProfessorLookupAgent` handles named-professor questions.
            - `ResearchMatchAgent` handles topic-to-professor questions.

            Both specialists use narrow deterministic Neo4j tools built on `Neo4jQueryService`, not the full MCP surface.

            **Learn more:** [LangChain handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs), [LangGraph swarm reference](https://reference.langchain.com/python/langgraph-swarm/swarm/create_swarm), [LangGraph multi-agent concepts](https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/multi_agent.md)
            """
        ),
        md(
            """
            ## 1. Environment Check

            Learning goal: verify that this notebook is running inside the project environment and that the swarm package is available.

            Theory: Lab 3 only works if three layers are in place before we talk about handoffs: a model, Neo4j access, and the swarm orchestration package. We keep the setup cell short so the rest of the notebook can stay focused on agent ownership and handoffs.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import json
            import sys
            import warnings
            from importlib.metadata import version
            from pathlib import Path
            from typing import Any

            PROJECT_ROOT = Path.cwd().resolve()
            if not (PROJECT_ROOT / "pyproject.toml").exists():
                for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
                    if (candidate / "pyproject.toml").exists():
                        PROJECT_ROOT = candidate
                        break

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            warnings.filterwarnings("ignore", message="IProgress not found.*")

            from langchain.agents import create_agent
            from langchain.messages import AIMessage, ToolMessage
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph_swarm import SwarmState, create_handoff_tool, create_swarm

            from bit_professor_chat.config import TutorSettings
            from bit_professor_chat.model_factory import build_model
            from bit_professor_chat.neo4j_queries import Neo4jQueryService

            def pretty(data: Any) -> None:
                print(json.dumps(data, ensure_ascii=False, indent=2))

            settings = TutorSettings.from_env(PROJECT_ROOT / ".env")
            model = build_model(settings)
            query_service = Neo4jQueryService(settings)

            pretty(
                {
                    "project_root": str(PROJECT_ROOT),
                    "model": settings.lab_tutor_llm_model,
                    "neo4j_uri": settings.neo4j_uri,
                    "langgraph_swarm_version": version("langgraph-swarm"),
                }
            )
            '''
        ),
        md(
            """
            ## 2. Neo4j Preflight

            Learning goal: confirm that Neo4j is ready before we start the swarm work.

            Theory: Lab 3 is student-ready only when the graph is prepared ahead of time. If the local database is empty, stop here, seed it from the committed graph JSON files, and then come back to the notebook.
            """
        ),
        code(
            '''
            try:
                preflight_overview, preflight_traces = query_service.get_graph_overview(limit=8)
            except Exception as exc:
                raise RuntimeError(
                    "Neo4j is not ready. Start it with `docker compose up -d neo4j`, then "
                    "seed the database with `uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py`."
                ) from exc

            if preflight_overview.professor_count == 0:
                raise RuntimeError(
                    "Neo4j is empty. Run `uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py` "
                    "before continuing with Lab 3."
                )

            pretty(
                {
                    "status": "ready",
                    "professor_count": preflight_overview.professor_count,
                    "relationship_count": preflight_overview.relationship_count,
                    "seed_command": "uv run python lab_3_langgraph_swarm/prepare_lab3_graph.py",
                }
            )
            '''
        ),
        md(
            """
            ## 3. Swarm Mental Model

            Learning goal: make the control-flow difference between Lab 2 and Lab 3 explicit before we define any code.

            Theory: Lab 2 used a supervisor. `StudentAgent` always stayed in charge, workers ran one task, and control returned to the supervisor. Lab 3 is different. The active agent owns the conversation directly. A handoff tool updates `active_agent`, and the next turn resumes from that remembered owner.
            """
        ),
        code(
            '''
            pretty(
                {
                    "lab_2_supervisor": {
                        "control": "StudentAgent stays in charge",
                        "worker_return_path": "workers always return to the supervisor",
                        "persistent_field": "route / worker_result / final_answer style query state",
                    },
                    "lab_3_swarm": {
                        "control": "the current specialist owns the conversation",
                        "worker_return_path": "no supervisor hop is required after a handoff",
                        "persistent_field": "messages + active_agent only",
                    },
                }
            )
            '''
        ),
        md(
            """
            ## 4. Verify the Neo4j Surface

            Learning goal: confirm the prepared local graph is reachable and see the small deterministic query layer that the swarm will sit on top of.

            Theory: this lab is not about Cypher generation. The Neo4j surface is intentionally narrow, because the teaching target is handoffs, not tool discovery over a large API.
            """
        ),
        code(
            '''
            overview, traces = preflight_overview, preflight_traces

            pretty(
                {
                    "professor_count": overview.professor_count,
                    "relationship_count": overview.relationship_count,
                    "sample_professors": overview.sample_professors,
                    "common_predicates": overview.common_predicates[:5],
                }
            )
            '''
        ),
        md(
            """
            ## 5. Define the Deterministic Neo4j Tools

            Learning goal: expose just enough tool surface for the two specialists.

            Theory: both specialists talk to the same database, but they do not get the same tool set. This is the simplest useful example of tool partitioning inside a swarm.
            """
        ),
        code(
            '''
            @tool
            def resolve_professor_tool(name_hint: str) -> dict[str, Any]:
                """Resolve a professor name hint into ranked Neo4j matches."""
                matches, trace = query_service.resolve_professor(name_hint=name_hint, limit=5)
                return {
                    "matches": [match.to_dict() for match in matches],
                    "trace": trace.to_dict(),
                }


            @tool
            def get_professor_facts_tool(professor_name: str) -> dict[str, Any]:
                """Fetch a compact set of professor-grounded facts from Neo4j."""
                facts, trace = query_service.get_professor_facts(
                    professor_name=professor_name,
                    keywords=[],
                    limit=10,
                )
                return {
                    "facts": [fact.to_dict() for fact in facts],
                    "trace": trace.to_dict(),
                }


            @tool
            def find_professors_by_topics_tool(keywords: list[str]) -> dict[str, Any]:
                """Recommend professors from a small list of topic keywords."""
                matches, trace = query_service.find_professors_by_topics(
                    keywords=keywords,
                    limit=5,
                )
                return {
                    "matches": [match.to_dict() for match in matches],
                    "trace": trace.to_dict(),
                }


            pretty(
                {
                    "front_desk_tools": [],
                    "professor_lookup_tools": [
                        resolve_professor_tool.name,
                        get_professor_facts_tool.name,
                    ],
                    "research_match_tools": [find_professors_by_topics_tool.name],
                }
            )
            '''
        ),
        md(
            """
            ## 6. Add Handoff Tools and Specialist Agents

            Learning goal: build three agents with different prompts and different tool boundaries.

            Theory: `FrontDeskAgent` owns the first turn only. After that, specialists can keep control or hand off to each other. The handoff tool is the swarm's control-flow primitive: it updates `active_agent` and routes execution to another agent node.
            """
        ),
        code(
            '''
            transfer_to_professor_lookup = create_handoff_tool(
                agent_name="ProfessorLookupAgent",
                name="transfer_to_professor_lookup",
                description="Hand off to ProfessorLookupAgent for named professor questions.",
            )

            transfer_to_research_match = create_handoff_tool(
                agent_name="ResearchMatchAgent",
                name="transfer_to_research_match",
                description="Hand off to ResearchMatchAgent for topic-to-professor matching.",
            )


            FRONT_DESK_PROMPT = """
            You are FrontDeskAgent for the BIT professor graph.

            You have no Neo4j tools. Your only job is to decide which specialist should own the conversation.

            Routing rules:
            - If the user names a specific professor, hand off to ProfessorLookupAgent.
            - If the user asks which professors match a topic, research direction, or interest area, hand off to ResearchMatchAgent.
            - Do not answer substantive database questions yourself.
            - After you decide, call exactly one handoff tool.
            """.strip()


            PROFESSOR_LOOKUP_PROMPT = """
            You are ProfessorLookupAgent for the BIT professor graph.

            Scope:
            - Answer only named-professor questions.
            - Use Neo4j evidence returned by your tools.
            - Start with resolve_professor_tool before using get_professor_facts_tool.

            Handoff rule:
            - If the user is asking for topic-based recommendations or "which professors should I look at", hand off to ResearchMatchAgent.

            Answer style:
            - Ground every answer in tool output.
            - If the graph does not contain the answer, say so clearly.
            - If name resolution is ambiguous, explain that ambiguity instead of guessing.
            """.strip()


            RESEARCH_MATCH_PROMPT = """
            You are ResearchMatchAgent for the BIT professor graph.

            Scope:
            - Recommend professors for a research direction or topic.
            - Use find_professors_by_topics_tool with a short list of concrete keywords.

            Handoff rule:
            - If the user is asking about one named professor, hand off to ProfessorLookupAgent.

            Answer style:
            - Ground every answer in tool output.
            - Prefer concise recommendations with matched nodes or predicates when helpful.
            - If the graph has no match, say that clearly instead of guessing.
            """.strip()


            front_desk_agent = create_agent(
                model,
                tools=[transfer_to_professor_lookup, transfer_to_research_match],
                system_prompt=FRONT_DESK_PROMPT,
                name="FrontDeskAgent",
            )

            professor_lookup_agent = create_agent(
                model,
                tools=[
                    resolve_professor_tool,
                    get_professor_facts_tool,
                    transfer_to_research_match,
                ],
                system_prompt=PROFESSOR_LOOKUP_PROMPT,
                name="ProfessorLookupAgent",
            )

            research_match_agent = create_agent(
                model,
                tools=[
                    find_professors_by_topics_tool,
                    transfer_to_professor_lookup,
                ],
                system_prompt=RESEARCH_MATCH_PROMPT,
                name="ResearchMatchAgent",
            )

            pretty(
                {
                    "agent_names": [
                        front_desk_agent.name,
                        professor_lookup_agent.name,
                        research_match_agent.name,
                    ]
                }
            )
            '''
        ),
        md(
            """
            ## 7. Compile the Swarm with Memory

            Learning goal: compile the swarm with `InMemorySaver` and keep the state as small as possible.

            Theory: the whole Lab 3 state is just `messages` plus `active_agent`. We do not reintroduce Lab 2 supervisor fields, because that would blur the main design lesson.
            """
        ),
        code(
            '''
            class Lab3State(SwarmState):
                """Lab 3 keeps only messages plus active_agent."""


            swarm_workflow = create_swarm(
                [
                    front_desk_agent,
                    professor_lookup_agent,
                    research_match_agent,
                ],
                default_active_agent="FrontDeskAgent",
                state_schema=Lab3State,
            )
            swarm_app = swarm_workflow.compile(checkpointer=InMemorySaver())


            def stringify_content(content: Any) -> str:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text") or item.get("content")
                            parts.append(text if text else json.dumps(item, ensure_ascii=False))
                        else:
                            parts.append(str(item))
                    return "\\n".join(part for part in parts if part)
                if isinstance(content, dict):
                    return json.dumps(content, ensure_ascii=False, indent=2)
                return str(content)


            def extract_last_ai_message(messages: list[Any]) -> str:
                for message in reversed(messages):
                    if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                        return stringify_content(message.content)
                return ""


            def summarize_tool_message(message: ToolMessage) -> dict[str, Any]:
                content = stringify_content(message.content)
                if content.startswith("Successfully transferred to "):
                    return {
                        "event": "handoff",
                        "target": content.removeprefix("Successfully transferred to "),
                    }

                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    return {
                        "event": "tool",
                        "tool": getattr(message, "name", None) or "tool",
                        "preview": content[:160],
                    }

                trace = payload.get("trace", {}) if isinstance(payload, dict) else {}
                summary = {
                    "event": "tool",
                    "tool": trace.get("name") or getattr(message, "name", None) or "tool",
                }

                if isinstance(payload, dict) and "matches" in payload:
                    matches = payload.get("matches", [])
                    summary["result_count"] = len(matches)
                    summary["sample_professors"] = [
                        match.get("professor_name")
                        for match in matches[:3]
                        if isinstance(match, dict) and match.get("professor_name")
                    ]
                    return summary

                if isinstance(payload, dict) and "facts" in payload:
                    facts = payload.get("facts", [])
                    predicates: list[str] = []
                    for fact in facts:
                        if not isinstance(fact, dict):
                            continue
                        predicate = fact.get("predicate")
                        if predicate and predicate not in predicates:
                            predicates.append(predicate)
                    summary["result_count"] = len(facts)
                    summary["sample_predicates"] = predicates[:3]
                    return summary

                return summary


            def summarize_tool_steps(messages: list[Any]) -> list[dict[str, Any]]:
                return [
                    summarize_tool_message(message)
                    for message in messages
                    if isinstance(message, ToolMessage)
                ][-4:]


            def checkpoint_active_agent(thread_id: str) -> str | None:
                snapshot = swarm_app.get_state({"configurable": {"thread_id": thread_id}})
                return snapshot.values.get("active_agent")


            def run_turn(question: str, *, thread_id: str) -> dict[str, Any]:
                config = {"configurable": {"thread_id": thread_id}}
                prior_snapshot = swarm_app.get_state(config)
                prior_values = getattr(prior_snapshot, "values", {}) or {}
                prior_message_count = len(prior_values.get("messages", []))
                started_from_active_agent = (
                    prior_values.get("active_agent") or "FrontDeskAgent"
                )
                state = swarm_app.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config,
                )
                current_turn_messages = state.get("messages", [])[prior_message_count:]
                return {
                    "question": question,
                    "thread_id": thread_id,
                    "started_from_active_agent": started_from_active_agent,
                    "active_agent": state.get("active_agent"),
                    "tool_steps": summarize_tool_steps(current_turn_messages),
                    "assistant_reply": extract_last_ai_message(state.get("messages", [])),
                }


            pretty(
                {
                    "swarm_nodes": list(swarm_app.get_graph().nodes.keys()),
                    "default_active_agent": "FrontDeskAgent",
                    "state_fields": list(Lab3State.__annotations__.keys()),
                }
            )
            '''
        ),
        md(
            """
            ## 8. Scenario: Named Professor

            Learning goal: watch the front desk hand off to the professor specialist for one direct professor question.
            """
        ),
        code(
            '''
            professor_turn = run_turn(
                "What are CHENG Cheng's research interests?",
                thread_id="lab3-professor",
            )
            pretty(professor_turn)
            '''
        ),
        md(
            """
            ## 9. Scenario: Topic Match

            Learning goal: route a broad research-interest question directly to the topic-matching specialist.
            """
        ),
        code(
            '''
            topic_turn = run_turn(
                "I am interested in computer vision and multimedia content analysis. Which professors should I look at?",
                thread_id="lab3-topic",
            )
            pretty(topic_turn)
            '''
        ),
        md(
            """
            ## 10. Scenario: Same-Thread Memory

            Learning goal: show that the swarm remembers who was active after the first turn, and that a later follow-up can hand off from one specialist to another.

            Theory: after Turn 1, the checkpoint stores `active_agent`. When Turn 2 reuses the same `thread_id`, the swarm resumes from that remembered owner instead of resetting to `FrontDeskAgent`.
            """
        ),
        code(
            '''
            memory_thread = "lab3-memory"

            turn_1 = run_turn(
                "I am interested in computer vision and multimedia content analysis. Which professors should I look at?",
                thread_id=memory_thread,
            )
            remembered_agent = checkpoint_active_agent(memory_thread)
            turn_2 = run_turn(
                "What about CHENG Cheng specifically?",
                thread_id=memory_thread,
            )

            pretty(
                {
                    "turn_1_final_active_agent": turn_1["active_agent"],
                    "checkpoint_active_agent_before_turn_2": remembered_agent,
                    "turn_2_started_from_active_agent": turn_2["started_from_active_agent"],
                    "turn_2_final_active_agent": turn_2["active_agent"],
                    "turn_2_tool_steps": turn_2["tool_steps"],
                    "turn_2_reply": turn_2["assistant_reply"],
                }
            )
            '''
        ),
        md(
            """
            ## 11. Wrap-Up

            Lab 3 teaches only three new ideas beyond Lab 2:

            - `active_agent` is now the routing state.
            - handoff tools move control between specialists.
            - the same `thread_id` resumes from the last active owner.

            The query surface stays intentionally small so the swarm mechanics stay visible.
            """
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
    )


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(nbformat.writes(notebook), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
