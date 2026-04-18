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
            # Lab 3: Manual Tool-Calling Swarm

            **Editable diagram:** [lab_3_swarm.excalidraw](lab_3_swarm.excalidraw)  
            **Workflow preview:** [lab_3_swarm.svg](lab_3_swarm.svg)

            This lab keeps the Lab 2 question domain, but removes the supervisor and the prebuilt agent wrapper. The core swarm idea is still the same:

            - the current specialist talks to the user directly,
            - handoff tools transfer control to another specialist,
            - `active_agent` persists across turns for the same `thread_id`.

            This notebook starts Neo4j locally and seeds it from the committed typed structured profile dataset in Section 2, so the graph-backed swarm flow is fully reproducible from inside the notebook. We use two specialists:

            - `ProfessorLookupAgent` handles named-professor questions.
            - `ResearchMatchAgent` handles topic-to-professor questions.

            Both specialists are explicit LangGraph subgraphs with a tiny `llm -> ToolNode -> llm` loop, so the tool-calling and handoff mechanics stay visible.

            **Learn more:** [LangChain handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs), [LangGraph swarm reference](https://reference.langchain.com/python/langgraph-swarm/swarm/create_swarm), [LangGraph multi-agent concepts](https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/multi_agent.md)
            """
        ),
        md(
            """
            ## 1. Environment Check

            Learning goal: verify that this notebook is running inside the project environment and that the swarm package is available.

            Theory: Lab 3 only works if three layers are in place before we talk about handoffs: a model, Neo4j access, and the swarm orchestration package. We keep the setup cell short so the rest of the notebook can stay focused on explicit swarm ownership and tool execution.
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
            from typing import Any, Annotated

            PROJECT_ROOT = Path.cwd().resolve()
            if not (PROJECT_ROOT / "pyproject.toml").exists():
                for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
                    if (candidate / "pyproject.toml").exists():
                        PROJECT_ROOT = candidate
                        break

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            warnings.filterwarnings("ignore", message="IProgress not found.*")

            from langchain.messages import AIMessage, SystemMessage, ToolMessage
            from langchain_core.runnables import RunnableConfig
            from langchain.tools import InjectedToolCallId, tool
            from langgraph.config import get_stream_writer
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, StateGraph
            from langgraph.prebuilt import InjectedState, ToolNode
            from langgraph.types import Command
            from langgraph_swarm import SwarmState, create_swarm
            from langgraph_swarm.handoff import METADATA_KEY_HANDOFF_DESTINATION

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
            ## 2. Start Neo4j and Seed the Graph

            Learning goal: start the local Neo4j service and load the committed Lab 3 professor graph directly from the notebook.

            Theory: Lab 3 is built around a prepared typed professor graph. Instead of asking students to leave the notebook and seed Neo4j manually, we can do the setup here: bring up Docker for Neo4j, load the committed structured profile JSON files, and then verify the resulting counts before moving on to swarm handoffs.
            """
        ),
        code(
            '''
            import subprocess

            from bit_professor_chat.graph_ingestion import insert_structured_output_to_neo4j

            docker_result = subprocess.run(
                ["docker", "compose", "up", "-d", "neo4j"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if docker_result.returncode != 0:
                raise RuntimeError(
                    "Could not start Neo4j with `docker compose up -d neo4j`.\\n\\n"
                    f"stdout:\\n{docker_result.stdout}\\n\\nstderr:\\n{docker_result.stderr}"
                )

            seed_summary = insert_structured_output_to_neo4j(
                settings=settings,
                project_root=PROJECT_ROOT,
                reset_database=True,
            )

            pretty(
                {
                    "status": "seeded",
                    "docker_command": "docker compose up -d neo4j",
                    "seed_call": "insert_structured_output_to_neo4j(..., reset_database=True)",
                    "professor_count": seed_summary.professor_count,
                    "success_count": seed_summary.success_count,
                    "node_count": seed_summary.node_count,
                    "relationship_count": seed_summary.relationship_count,
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
            ## 4. Define the Deterministic Neo4j Tools

            Learning goal: expose just enough tool surface for the two specialists.

            Theory: both specialists talk to the same database, but they do not get the same tool set. This is the simplest useful example of tool partitioning inside a swarm.
            """
        ),
        code(
            '''
            def emit_teaching_event(event: dict[str, Any]) -> None:
                writer = get_stream_writer()
                writer(event)


            @tool
            def resolve_professor_tool(name_hint: str) -> dict[str, Any]:
                """Resolve a professor name hint into ranked Neo4j matches."""
                emit_teaching_event(
                    {
                        "kind": "tool_call",
                        "agent": "ProfessorLookupAgent",
                        "tool": "resolve_professor_tool",
                        "input": {"name_hint": name_hint},
                    }
                )
                matches, _ = query_service.resolve_professor(name_hint=name_hint, limit=5)
                result = {
                    "matches": [match.to_dict() for match in matches],
                }
                emit_teaching_event(
                    {
                        "kind": "tool_result",
                        "agent": "ProfessorLookupAgent",
                        "tool": "resolve_professor_tool",
                        "match_count": len(result["matches"]),
                    }
                )
                return result


            @tool
            def get_professor_facts_tool(professor_name: str) -> dict[str, Any]:
                """Fetch a compact set of professor-grounded facts from Neo4j."""
                emit_teaching_event(
                    {
                        "kind": "tool_call",
                        "agent": "ProfessorLookupAgent",
                        "tool": "get_professor_facts_tool",
                        "input": {"professor_name": professor_name},
                    }
                )
                facts, _ = query_service.get_professor_facts(
                    professor_name=professor_name,
                    keywords=[],
                    limit=10,
                )
                result = {
                    "facts": [fact.to_dict() for fact in facts],
                }
                emit_teaching_event(
                    {
                        "kind": "tool_result",
                        "agent": "ProfessorLookupAgent",
                        "tool": "get_professor_facts_tool",
                        "fact_count": len(result["facts"]),
                    }
                )
                return result


            @tool
            def find_professors_by_topics_tool(keywords: list[str]) -> dict[str, Any]:
                """Recommend professors from a small list of topic keywords."""
                emit_teaching_event(
                    {
                        "kind": "tool_call",
                        "agent": "ResearchMatchAgent",
                        "tool": "find_professors_by_topics_tool",
                        "input": {"keywords": keywords},
                    }
                )
                matches, _ = query_service.find_professors_by_topics(
                    keywords=keywords,
                    limit=5,
                )
                result = {
                    "matches": [match.to_dict() for match in matches],
                }
                emit_teaching_event(
                    {
                        "kind": "tool_result",
                        "agent": "ResearchMatchAgent",
                        "tool": "find_professors_by_topics_tool",
                        "match_count": len(result["matches"]),
                    }
                )
                return result


            pretty(
                {
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
            ## 5. Build Custom Handoff Tools and Specialist Graphs

            Learning goal: build two explicit specialist subgraphs with visible tool-calling and handoff logic.

            Theory: each specialist is just a tiny LangGraph loop: an `llm` node decides what to do next, `ToolNode` executes Neo4j tools or a handoff tool, and the loop continues until the model stops requesting tools. The custom handoff tool is what updates `active_agent` in the parent swarm.
            """
        ),
        code(
            '''
            class Lab3State(SwarmState):
                """Lab 3 keeps only messages plus active_agent."""


            def create_lab3_handoff_tool(
                *,
                agent_name: str,
                name: str,
                description: str,
            ):
                @tool(name, description=description)
                def handoff_to_agent(
                    state: Annotated[Any, InjectedState],
                    tool_call_id: Annotated[str, InjectedToolCallId],
                ) -> Command:
                    emit_teaching_event(
                        {
                            "kind": "handoff",
                            "from_agent": state.get("active_agent"),
                            "to_agent": agent_name,
                            "tool": name,
                        }
                    )
                    tool_message = ToolMessage(
                        content=f"Successfully transferred to {agent_name}",
                        name=name,
                        tool_call_id=tool_call_id,
                    )
                    return Command(
                        goto=agent_name,
                        graph=Command.PARENT,
                        update={
                            "messages": [*state["messages"], tool_message],
                            "active_agent": agent_name,
                        },
                    )

                handoff_to_agent.metadata = {
                    METADATA_KEY_HANDOFF_DESTINATION: agent_name
                }
                return handoff_to_agent


            transfer_to_professor_lookup = create_lab3_handoff_tool(
                agent_name="ProfessorLookupAgent",
                name="transfer_to_professor_lookup",
                description="Hand off to ProfessorLookupAgent for named professor questions.",
            )

            transfer_to_research_match = create_lab3_handoff_tool(
                agent_name="ResearchMatchAgent",
                name="transfer_to_research_match",
                description="Hand off to ResearchMatchAgent for topic-to-professor matching.",
            )


            PROFESSOR_LOOKUP_PROMPT = """
            You are ProfessorLookupAgent for the BIT professor graph.

            Scope:
            - Handle named-professor questions.
            - Use Neo4j evidence returned by your tools.
            - For named-professor questions, call resolve_professor_tool first.
            - If you get one clear match, call get_professor_facts_tool before answering.

            Handoff rule:
            - If the user is asking for topic-based recommendations or "which professors should I look at", call transfer_to_research_match.

            Answer style:
            - Ground every answer in tool output.
            - If the graph does not contain the answer, say so clearly.
            - If name resolution is ambiguous, explain that ambiguity instead of guessing.
            - Do not answer from prior knowledge.
            """.strip()


            RESEARCH_MATCH_PROMPT = """
            You are ResearchMatchAgent for the BIT professor graph.

            Scope:
            - Recommend professors for a research direction or topic.
            - Use find_professors_by_topics_tool with a short list of concrete keywords before answering.

            Handoff rule:
            - If the user is asking about one named professor, call transfer_to_professor_lookup.

            Answer style:
            - Ground every answer in tool output.
            - Prefer concise recommendations with matched nodes when helpful.
            - If the graph has no match, say that clearly instead of guessing.
            - Do not answer from prior knowledge.
            """.strip()


            def build_specialist_agent(
                *,
                agent_name: str,
                system_prompt: str,
                tools: list[Any],
            ):
                bound_model = model.bind_tools(tools)
                tool_node = ToolNode(tools, name="tools")

                def call_model(state: Lab3State, config: RunnableConfig) -> dict[str, Any]:
                    emit_teaching_event(
                        {
                            "kind": "agent_turn",
                            "agent": agent_name,
                            "message_count": len(state["messages"]),
                            "thread_id": config["configurable"].get("thread_id"),
                        }
                    )
                    response = bound_model.invoke(
                        [SystemMessage(content=system_prompt), *state["messages"]]
                    )
                    return {
                        "messages": [response],
                        "active_agent": agent_name,
                    }

                def route_after_model(state: Lab3State):
                    last_message = state["messages"][-1]
                    if isinstance(last_message, AIMessage) and last_message.tool_calls:
                        return "tools"
                    return END

                workflow = StateGraph(Lab3State)
                workflow.add_node("llm", call_model)
                workflow.add_node("tools", tool_node)
                workflow.add_edge(START, "llm")
                workflow.add_conditional_edges("llm", route_after_model)
                workflow.add_edge("tools", "llm")
                return workflow.compile(name=agent_name)


            professor_lookup_agent = build_specialist_agent(
                agent_name="ProfessorLookupAgent",
                system_prompt=PROFESSOR_LOOKUP_PROMPT,
                tools=[
                    resolve_professor_tool,
                    get_professor_facts_tool,
                    transfer_to_research_match,
                ],
            )

            research_match_agent = build_specialist_agent(
                agent_name="ResearchMatchAgent",
                system_prompt=RESEARCH_MATCH_PROMPT,
                tools=[
                    find_professors_by_topics_tool,
                    transfer_to_professor_lookup,
                ],
            )
            from IPython.display import HTML, Image, display
            from langchain_core.runnables.graph import MermaidDrawMethod

            display(HTML("<strong>ProfessorLookupAgent graph</strong>"))
            display(
                Image(
                    professor_lookup_agent.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )
                )
            )

            display(HTML("<strong>ResearchMatchAgent graph</strong>"))
            display(
                Image(
                    research_match_agent.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )
                )
            )
            '''
        ),
        md(
            """
            ## 6. Compile the Swarm with Memory

            Learning goal: compile the swarm with `InMemorySaver` and keep the state as small as possible.

            Theory: `create_swarm(...)` is still the parent orchestration layer. The only persistent Lab 3 state is `messages` plus `active_agent`, and the specialists themselves are now explicit compiled subgraphs rather than prebuilt agents.
            """
        ),
        code(
            '''
            swarm_workflow = create_swarm(
                [
                    professor_lookup_agent,
                    research_match_agent,
                ],
                default_active_agent="ProfessorLookupAgent",
                state_schema=Lab3State,
            )
            swarm_app = swarm_workflow.compile(checkpointer=InMemorySaver())

            display(HTML("<strong>Whole swarm graph</strong>"))
            display(
                Image(
                    swarm_app.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )
                )
            )
            '''
        ),
        md(
            """
            ## 7. Gradio App

            Learning goal: interact with the swarm through a small UI and inspect handoffs, tool usage, `active_agent`, and same-thread memory without exposing chain-of-thought.

            Theory: the chat stays student-friendly on the left, while the right panel surfaces only the swarm mechanics we care about: which specialist acted, which deterministic Neo4j tool ran, when a handoff happened, and which `thread_id` is still carrying memory. Resetting the app creates a fresh `thread_id`, so students can see the same memory rule from Step 9 in a more interactive form first.
            """
        ),
        code(
            '''
            import uuid

            import gradio as gr

            GRADIO_EXAMPLES = [
                "What are CHENG Cheng's research interests?",
                "Show me a few facts about CHE Haiying.",
                "I am interested in computer vision and multimedia content analysis. Which professors should I look at?",
            ]


            def message_to_text(message: Any) -> str:
                content = getattr(message, "content", message)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                parts.append(str(item["text"]))
                            else:
                                parts.append(json.dumps(item, ensure_ascii=False))
                        else:
                            parts.append(str(item))
                    return "\\n".join(parts)
                return str(content)


            def build_thread_id() -> str:
                return f"lab3-gradio-{uuid.uuid4().hex[:8]}"


            def placeholder_trace(thread_id: str) -> dict[str, Any]:
                return {
                    "thread_id": thread_id,
                    "events": [],
                    "message": "Send a question to inspect the latest swarm mechanics.",
                }


            def placeholder_snapshot(thread_id: str) -> dict[str, Any]:
                return {
                    "thread_id": thread_id,
                    "active_agent": None,
                    "stored_message_count": 0,
                    "turns_in_session": 0,
                }


            def new_gradio_session() -> dict[str, Any]:
                thread_id = build_thread_id()
                return {
                    "chat_history": [],
                    "thread_id": thread_id,
                    "latest_trace_payload": placeholder_trace(thread_id),
                    "latest_snapshot": placeholder_snapshot(thread_id),
                }


            def normalize_teaching_event(raw_event: dict[str, Any], *, thread_id: str) -> dict[str, Any]:
                event = dict(raw_event)
                event["thread_id"] = thread_id
                if "active_agent" not in event:
                    event["active_agent"] = (
                        event.get("agent")
                        or event.get("to_agent")
                        or event.get("from_agent")
                    )
                return event


            def build_trace_payload(thread_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
                if not events:
                    return {
                        "thread_id": thread_id,
                        "events": [],
                        "message": "The swarm answered without emitting tool or handoff events for this turn.",
                    }
                return {
                    "thread_id": thread_id,
                    "events": events,
                }


            def build_swarm_snapshot(
                *,
                thread_id: str,
                chat_history: list[dict[str, str]],
                state_values: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                stored_messages = list((state_values or {}).get("messages", []))
                return {
                    "thread_id": thread_id,
                    "active_agent": (state_values or {}).get("active_agent"),
                    "stored_message_count": len(stored_messages),
                    "turns_in_session": sum(1 for item in chat_history if item["role"] == "user"),
                }


            def ask_swarm(prompt: str, session_state: dict[str, Any] | None):
                session = dict(session_state or new_gradio_session())
                prompt = prompt.strip()
                if not prompt:
                    return (
                        "",
                        session["chat_history"],
                        session["latest_trace_payload"],
                        session["latest_snapshot"],
                        session,
                    )

                thread_id = session["thread_id"]
                config = {"configurable": {"thread_id": thread_id}}
                teaching_events: list[dict[str, Any]] = []

                try:
                    for part in swarm_app.stream(
                        {"messages": [{"role": "user", "content": prompt}]},
                        config=config,
                        stream_mode="custom",
                        subgraphs=True,
                        version="v2",
                    ):
                        if part["type"] != "custom":
                            continue
                        teaching_events.append(
                            normalize_teaching_event(part["data"], thread_id=thread_id)
                        )

                    state_values = swarm_app.get_state(config).values
                    assistant_reply = message_to_text(state_values["messages"][-1])
                    chat_history = [
                        *session["chat_history"],
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_reply},
                    ]
                    trace_payload = build_trace_payload(thread_id, teaching_events)
                    snapshot = build_swarm_snapshot(
                        thread_id=thread_id,
                        chat_history=chat_history,
                        state_values=state_values,
                    )
                except Exception as exc:
                    error_message = f"Swarm query failed: {exc}"
                    chat_history = [
                        *session["chat_history"],
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": error_message},
                    ]
                    trace_payload = {
                        "thread_id": thread_id,
                        "events": [
                            {
                                "kind": "error",
                                "thread_id": thread_id,
                                "message": error_message,
                            }
                        ],
                    }
                    snapshot = build_swarm_snapshot(
                        thread_id=thread_id,
                        chat_history=chat_history,
                    )

                session["chat_history"] = chat_history
                session["latest_trace_payload"] = trace_payload
                session["latest_snapshot"] = snapshot
                return "", chat_history, trace_payload, snapshot, session


            def reset_swarm_chat():
                session = new_gradio_session()
                return (
                    "",
                    session["chat_history"],
                    session["latest_trace_payload"],
                    session["latest_snapshot"],
                    session,
                )


            initial_session = new_gradio_session()

            with gr.Blocks(fill_width=True) as demo:
                gr.Markdown(
                    """### Lab 3 Gradio App
            Ask the notebook-local swarm a question, then inspect the teaching trace and remembered swarm state for the current `thread_id`."""
                )

                session_state = gr.State(initial_session)

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Swarm conversation",
                            height=420,
                            value=[],
                        )
                        prompt = gr.Textbox(
                            label="Ask the Lab 3 swarm",
                            placeholder="Try one of the example prompts below.",
                        )
                        with gr.Row():
                            ask_button = gr.Button("Ask", variant="primary")
                            reset_button = gr.Button("Start a new chat")
                        gr.Examples(examples=GRADIO_EXAMPLES, inputs=prompt)
                    with gr.Column(scale=2):
                        trace_view = gr.JSON(
                            label="Latest teaching trace",
                            value=initial_session["latest_trace_payload"],
                        )
                        snapshot_view = gr.JSON(
                            label="Current swarm snapshot",
                            value=initial_session["latest_snapshot"],
                        )
                        gr.Markdown(
                            "The trace shows only swarm mechanics: active specialist, deterministic Neo4j tools, handoffs, and the current `thread_id`."
                        )

                ask_button.click(
                    ask_swarm,
                    inputs=[prompt, session_state],
                    outputs=[prompt, chatbot, trace_view, snapshot_view, session_state],
                )
                prompt.submit(
                    ask_swarm,
                    inputs=[prompt, session_state],
                    outputs=[prompt, chatbot, trace_view, snapshot_view, session_state],
                )
                reset_button.click(
                    reset_swarm_chat,
                    outputs=[prompt, chatbot, trace_view, snapshot_view, session_state],
                )

            demo.launch(inline=True, height=920, width="100%")
            '''
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
