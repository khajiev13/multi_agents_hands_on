from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).parent
NOTEBOOK_PATH = ROOT / "lab_4_deep_agents.ipynb"


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        md(
            """
            # Lab 4: Pure Deep Agent Professor Workspace

            **Editable diagram:** [lab_4_workflow.excalidraw](lab_4_workflow.excalidraw)  
            **Workflow preview:** [lab_4_workflow.svg](lab_4_workflow.svg)

            This lab teaches **Deep Agents** as the next abstraction after Lab 3:

            - Lab 3 used a **swarm** to hand control between specialists.
            - Lab 4 uses **one deep agent** to manage an open-ended, file-based task.
            - The deep agent answers broad questions from a compact index, opens full dossiers only when needed, and delegates one add workflow to a subagent.

            We keep the core lesson pure:

            - no Neo4j
            - no custom `StateGraph`
            - no swarm router
            - no live URL discovery or OCR in the main path

            The knowledge base lives entirely in files:

            - `/professors.md` is the compact index
            - `/professors/<slug>.md` stores the full dossier
            - `/incoming/<slug>.md` stores one prepared incoming artifact
            - `/skills/add-professor-from-incoming/SKILL.md` stores the reusable core workflow

            The runtime workspace starts from a **small curated starter set** plus one prepared incoming artifact. That keeps the lesson deterministic. The live URL + OCR variant now lives in the instructor-only note [instructor_live_url_ocr_variant.md](instructor_live_url_ocr_variant.md).

            ![Workflow preview](lab_4_workflow.svg)

            **Learn more:** [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview), [customization](https://docs.langchain.com/oss/python/deepagents/customization), [skills](https://docs.langchain.com/oss/python/deepagents/skills), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents)
            """
        ),
        md(
            """
            ## 1. Environment Check

            Learning goal: verify that this notebook is running inside the project environment and that `deepagents` is installed.

            This setup cell stays intentionally small. The core lesson only needs the Deep Agents runtime, one model, and a few local markdown helpers.

            ![Architecture step 01](workflow_steps/step_01.svg)
            """
        ),
        code(
            '''
            from __future__ import annotations

            import json
            import shutil
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

            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend
            from langchain.messages import AIMessage
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver

            from bit_professor_chat.config import TutorSettings
            from bit_professor_chat.legacy_cache import (
                build_cached_summary_line,
                read_professor_markdown_metadata,
            )
            from bit_professor_chat.model_factory import build_model

            def pretty(data: Any) -> None:
                print(json.dumps(data, ensure_ascii=False, indent=2))

            settings = TutorSettings.from_env(PROJECT_ROOT / ".env")
            model = build_model(settings)

            LAB_ROOT = PROJECT_ROOT / "lab_4_deep_agents"
            SOURCE_DOSSIER_DIR = PROJECT_ROOT / "professors"
            SOURCE_CORE_SKILL_DIR = LAB_ROOT / "skills" / "add-professor-from-incoming"
            SOURCE_INCOMING_DIR = LAB_ROOT / "incoming_artifacts"
            RUNTIME_ROOT = PROJECT_ROOT / "artifacts" / "lab4" / "runtime"
            STARTER_DOSSIERS = [
                "che-haiying.md",
                "cheng-cheng.md",
                "gao-guangyu.md",
                "gao-yang.md",
                "huang-yonggang.md",
                "yan-bo.md",
                "zhao-fengnian.md",
                "zheng-hong.md",
            ]
            INCOMING_ARTIFACT_NAME = "zhang-qingjun.md"

            pretty(
                {
                    "project_root": str(PROJECT_ROOT),
                    "model": settings.lab_tutor_llm_model,
                    "deepagents_version": version("deepagents"),
                    "runtime_root": str(RUNTIME_ROOT),
                    "starter_dossiers": STARTER_DOSSIERS,
                    "incoming_artifact": INCOMING_ARTIFACT_NAME,
                }
            )
            '''
        ),
        md(
            """
            ## 2. Why Deep Agents?

            Learning goal: make the abstraction jump explicit before we write any code.

            Theory: a basic agent is good for a short tool loop. A custom LangGraph is good when you already know the control flow. A swarm is good when conversational ownership should move between specialists. A deep agent is better when the task is open-ended, file-centric, and easier to express as policy plus tools instead of a hand-built graph.

            ![Architecture step 02](workflow_steps/step_02.svg)
            """
        ),
        code(
            '''
            pretty(
                {
                    "basic_create_agent": "short bounded tool loop",
                    "custom_langgraph": "explicit graph when you know the workflow shape",
                    "swarm": "specialists hand off conversational ownership",
                    "deep_agent": "one harness with files, skills, and delegated subagents",
                    "lab_4_focus": "prompt rules + file tools + one skill-driven subagent",
                }
            )
            '''
        ),
        md(
            """
            ## 3. Prepare the Runtime Workspace

            Learning goal: create a safe file sandbox for the lab and seed it with a small deterministic dataset.

            Theory: the runtime root is the deep agent's world. We seed it from a curated starter subset of the repo's professor markdown files, add one prepared incoming artifact under `/incoming/`, and rebuild a fresh compact index. That keeps the main lesson deterministic and easy to reason about.

            ![Architecture step 03](workflow_steps/step_03.svg)
            """
        ),
        code(
            '''
            def rebuild_runtime_index(runtime_root: Path) -> list[str]:
                dossier_dir = runtime_root / "professors"
                summary_lines: list[str] = []
                for markdown_path in sorted(dossier_dir.glob("*.md")):
                    markdown_text = markdown_path.read_text(encoding="utf-8")
                    metadata = read_professor_markdown_metadata(markdown_path)
                    display_name = str(metadata.get("markdown_title") or markdown_path.stem)
                    summary_lines.append(build_cached_summary_line(markdown_text, display_name))

                summary_lines = sorted(dict.fromkeys(summary_lines), key=str.casefold)
                lines = ["# BIT CSAT Professors", ""]
                lines.extend(f"- {line}" for line in summary_lines)
                lines.append("")

                index_path = runtime_root / "professors.md"
                index_path.write_text("\\n".join(lines), encoding="utf-8")
                return summary_lines


            def seed_runtime_workspace(*, reset: bool = True) -> dict[str, Any]:
                if reset and RUNTIME_ROOT.exists():
                    shutil.rmtree(RUNTIME_ROOT)

                (RUNTIME_ROOT / "professors").mkdir(parents=True, exist_ok=True)
                (RUNTIME_ROOT / "incoming").mkdir(parents=True, exist_ok=True)
                (RUNTIME_ROOT / "skills").mkdir(parents=True, exist_ok=True)

                for dossier_name in STARTER_DOSSIERS:
                    shutil.copy2(
                        SOURCE_DOSSIER_DIR / dossier_name,
                        RUNTIME_ROOT / "professors" / dossier_name,
                    )

                shutil.copy2(
                    SOURCE_INCOMING_DIR / INCOMING_ARTIFACT_NAME,
                    RUNTIME_ROOT / "incoming" / INCOMING_ARTIFACT_NAME,
                )
                shutil.copytree(
                    SOURCE_CORE_SKILL_DIR,
                    RUNTIME_ROOT / "skills" / "add-professor-from-incoming",
                    dirs_exist_ok=True,
                )

                summary_lines = rebuild_runtime_index(RUNTIME_ROOT)
                index_preview = (RUNTIME_ROOT / "professors.md").read_text(encoding="utf-8").splitlines()[:10]
                incoming_preview = (
                    RUNTIME_ROOT / "incoming" / INCOMING_ARTIFACT_NAME
                ).read_text(encoding="utf-8").splitlines()[:8]

                return {
                    "runtime_root": str(RUNTIME_ROOT),
                    "dossier_count": len(list((RUNTIME_ROOT / "professors").glob("*.md"))),
                    "index_line_count": len(summary_lines),
                    "starter_dossiers": STARTER_DOSSIERS,
                    "incoming_artifact_path": f"/incoming/{INCOMING_ARTIFACT_NAME}",
                    "core_skill_path": "/skills/add-professor-from-incoming/SKILL.md",
                    "index_preview": index_preview,
                    "incoming_preview": incoming_preview,
                }


            runtime_snapshot = seed_runtime_workspace(reset=True)
            pretty(runtime_snapshot)
            '''
        ),
        code(
            '''
            core_skill_preview = (
                RUNTIME_ROOT / "skills" / "add-professor-from-incoming" / "SKILL.md"
            ).read_text(encoding="utf-8")

            print("\\n".join(core_skill_preview.splitlines()[:32]))
            '''
        ),
        md(
            """
            ## 4. Define the Deterministic Helper

            Learning goal: keep the custom tool layer as small as possible.

            Theory: the deep agent already knows how to read, write, list, and search files. In the core lesson we add exactly one deterministic helper: rebuild the compact index after a successful dossier add.

            ![Architecture step 04](workflow_steps/step_04.svg)
            """
        ),
        code(
            '''
            @tool
            def rebuild_professors_index_tool() -> dict[str, Any]:
                """Rebuild /professors.md from the current dossier folder."""
                summary_lines = rebuild_runtime_index(RUNTIME_ROOT)
                index_path = RUNTIME_ROOT / "professors.md"
                return {
                    "index_path": "/professors.md",
                    "professor_count": len(summary_lines),
                    "preview": index_path.read_text(encoding="utf-8").splitlines()[:12],
                }


            pretty({"main_agent_custom_tools": [rebuild_professors_index_tool.name]})
            '''
        ),
        md(
            """
            ## 5. Build the Subagent and the Main Deep Agent

            Learning goal: keep the orchestration pure Deep Agents.

            Theory: the main agent uses built-in file tools plus one deterministic helper. The add workflow lives in one subagent with one skill. That is the whole design: no custom graph, no router nodes, and no separate supervisor.

            ![Architecture step 05](workflow_steps/step_05.svg)
            """
        ),
        code(
            '''
            BACKEND = FilesystemBackend(root_dir=str(RUNTIME_ROOT), virtual_mode=False)
            CHECKPOINTER = InMemorySaver()

            MAIN_SYSTEM_PROMPT = """
            You are ProfessorWorkspaceAgent for a local professor knowledge base.

            Rules:
            - Work only inside the runtime workspace.
            - Keep answers grounded in the local files.
            - For broad questions, start with /professors.md.
            - For named-professor questions, start with /professors.md and open one dossier only if you need more detail.
            - If the user asks to add the prepared incoming professor artifact, delegate to AddProfessorSubagent.
            - If AddProfessorSubagent returns status=added, call rebuild_professors_index_tool before replying.
            - After a successful add and index rebuild, reply with a short confirmation that names the professor and the new dossier path.
            """.strip()

            ADD_SUBAGENT_PROMPT = """
            You are AddProfessorSubagent.

            Use the add-professor-from-incoming skill for the exact workflow.

            Rules:
            - Inspect /incoming/ and /professors/.
            - Avoid duplicates by slug or by matching detail_url in the markdown header.
            - If the professor is new, write exactly one dossier file at /professors/<slug>.md.
            - Do not rebuild /professors.md.
            - Return exactly one compact final line with status, professor_name, slug, and markdown_path.
            """.strip()

            add_professor_subagent = {
                "name": "AddProfessorSubagent",
                "description": "Use this agent when the user wants to add the prepared incoming professor artifact into the local workspace.",
                "system_prompt": ADD_SUBAGENT_PROMPT,
                "model": model,
                "tools": [],
                "skills": ["/skills/"],
            }

            professor_workspace_agent = create_deep_agent(
                model=model,
                tools=[rebuild_professors_index_tool],
                system_prompt=MAIN_SYSTEM_PROMPT,
                backend=BACKEND,
                subagents=[add_professor_subagent],
                checkpointer=CHECKPOINTER,
                name="ProfessorWorkspaceAgent",
            )


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
                return str(content)


            def read_thread_messages(agent: Any, thread_id: str) -> list[Any]:
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    snapshot = agent.get_state(config)
                except Exception:
                    return []
                values = dict(getattr(snapshot, "values", {}) or {})
                return list(values.get("messages", []))


            def extract_tool_steps(messages: list[Any]) -> list[str]:
                steps: list[str] = []
                for message in messages:
                    if not isinstance(message, AIMessage):
                        continue
                    for tool_call in message.tool_calls or []:
                        steps.append(tool_call["name"])
                return steps[-8:]


            def final_reply(messages: list[Any]) -> str:
                for message in reversed(messages):
                    if isinstance(message, AIMessage) and not (message.tool_calls or []):
                        return stringify_content(message.content).strip()
                return ""


            def run_turn(user_text: str, *, thread_id: str) -> dict[str, Any]:
                config = {"configurable": {"thread_id": thread_id}}
                before_messages = read_thread_messages(professor_workspace_agent, thread_id)
                result = professor_workspace_agent.invoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config=config,
                )
                new_messages = result["messages"][len(before_messages):]
                return {
                    "thread_id": thread_id,
                    "assistant_reply": final_reply(new_messages) or final_reply(result["messages"]),
                    "tool_steps": extract_tool_steps(new_messages),
                }


            pretty(
                {
                    "runtime_root": str(RUNTIME_ROOT),
                    "incoming_artifact_path": f"/incoming/{INCOMING_ARTIFACT_NAME}",
                    "core_skill_path": "/skills/add-professor-from-incoming/SKILL.md",
                    "main_agent_name": "ProfessorWorkspaceAgent",
                    "subagent_name": add_professor_subagent["name"],
                }
            )
            '''
        ),
        md(
            """
            ## 6. Scenario A: Broad Question From the Compact Index

            Learning goal: show the cheapest useful path. The main agent should answer from `/professors.md` without subagent delegation.
            """
        ),
        code(
            '''
            broad_turn = run_turn(
                "Which professors should I look at for human-computer interaction and virtual environments?",
                thread_id="lab4-broad",
            )
            pretty(broad_turn)
            '''
        ),
        md(
            """
            ## 7. Scenario B: Named Professor Detail

            Learning goal: show the file-first read path. The main agent starts from the index and opens one dossier when the question needs more detail.
            """
        ),
        code(
            '''
            detail_turn = run_turn(
                "Tell me more about Cheng Cheng.",
                thread_id="lab4-detail",
            )
            pretty(detail_turn)
            '''
        ),
        md(
            """
            ## 8. Scenario C: Add the Prepared Incoming Professor

            Learning goal: show the deterministic maintenance flow:

            - the main agent delegates to `AddProfessorSubagent`,
            - the subagent follows one skill and uses built-in file tools,
            - one new dossier file is written from `/incoming/`,
            - the main agent rebuilds `/professors.md`.

            ![Architecture step 06](workflow_steps/step_06.svg)
            """
        ),
        code(
            '''
            add_turn = None
            added_name = None
            added_slug = None
            added_files: list[str] = []

            before_files = {path.name for path in (RUNTIME_ROOT / "professors").glob("*.md")}
            add_turn = run_turn(
                "Add the prepared incoming professor from /incoming/ to the workspace.",
                thread_id="lab4-add",
            )

            after_files = {path.name for path in (RUNTIME_ROOT / "professors").glob("*.md")}
            added_files = sorted(after_files - before_files)

            if added_files:
                added_path = RUNTIME_ROOT / "professors" / added_files[0]
                added_slug = added_path.stem
                added_metadata = read_professor_markdown_metadata(added_path)
                added_name = str(added_metadata.get("markdown_title") or added_slug)

            index_lines = (RUNTIME_ROOT / "professors.md").read_text(encoding="utf-8").splitlines()
            pretty(
                {
                    "turn": add_turn,
                    "incoming_artifact_path": f"/incoming/{INCOMING_ARTIFACT_NAME}",
                    "added_files": added_files,
                    "index_lines_for_added_professor": [
                        line for line in index_lines if added_name and added_name.lower() in line.lower()
                    ],
                }
            )
            '''
        ),
        md(
            """
            ## 9. Scenario D: Post-add Validation

            Learning goal: confirm that the updated local workspace is immediately usable after the add flow.
            """
        ),
        code(
            '''
            if not added_name:
                pretty(
                    {
                        "status": "skipped",
                        "reason": "Scenario C did not add a new professor, so there is nothing new to validate.",
                    }
                )
            else:
                post_add_turn = run_turn(
                    f"What are {added_name}'s main research interests?",
                    thread_id="lab4-add",
                )
                pretty(
                    {
                        "added_name": added_name,
                        "added_slug": added_slug,
                        "post_add_turn": post_add_turn,
                    }
                )
            '''
        ),
        md(
            """
            ## 10. Wrap-up

            Lab 4 teaches four Deep Agent ideas:

            - use **one deep agent** for an open-ended task instead of a hand-built graph,
            - use the built-in **file tools** as the main working surface,
            - keep the longer add workflow in **one skill**,
            - delegate one isolated workflow to **one subagent**.

            The core lesson stays deterministic on purpose. Students can understand the Deep Agents design before they look at the instructor-only live ingestion variant.
            """
        ),
    ]

    notebook = new_notebook(cells=cells, metadata={"language_info": {"name": "python", "pygments_lexer": "ipython3"}})
    return notebook


def main() -> None:
    NOTEBOOK_PATH.write_text(nbformat.writes(build_notebook()), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
