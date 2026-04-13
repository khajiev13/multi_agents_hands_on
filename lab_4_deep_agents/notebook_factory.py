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
            - The deep agent answers broad questions from a compact index, opens full dossiers only when needed, and delegates the add-by-URL workflow to one subagent.

            We keep the lesson pure:

            - no Neo4j
            - no custom `StateGraph`
            - no swarm router
            - no `create_agent`

            The knowledge base lives entirely in files:

            - `/professors.md` is the compact index
            - `/professors/<slug>.md` stores the full dossier
            - `/skills/add-professor-from-url/SKILL.md` stores the reusable add workflow

            The runtime workspace starts from a **small curated starter set**, not the full site. That keeps the lab compact and leaves room for the add-by-URL scenario to demonstrate a real file update.

            ![Workflow preview](lab_4_workflow.svg)

            **Learn more:** [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview), [customization](https://docs.langchain.com/oss/python/deepagents/customization), [skills](https://docs.langchain.com/oss/python/deepagents/skills), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents)
            """
        ),
        md(
            """
            ## 1. Environment Check

            Learning goal: verify that this notebook is running inside the project environment and that `deepagents` is installed.

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
            from langchain.tools import tool
            from langchain_core.messages import AIMessage, BaseMessage
            from langgraph.checkpoint.memory import InMemorySaver

            from bit_professor_chat.config import TutorSettings
            from bit_professor_chat.ingestion_models import ProfessorListing
            from bit_professor_chat.legacy_cache import (
                build_cached_summary_line,
                read_professor_markdown_metadata,
            )
            from bit_professor_chat.markdown_corpus import slugify_name, validate_professor_markdown
            from bit_professor_chat.model_factory import build_model, build_ocr_model
            from bit_professor_chat.ocr_transcript import (
                build_professor_markdown_from_page_notes,
                cleanup_markdown_artifact,
                extract_header_identity_lines,
                extract_identity_candidates,
                extract_identity_candidates_from_lines,
                extract_professor_poster_notes,
                needs_top_block_fallback,
                parse_ocr_page_notes,
            )
            from bit_professor_chat.source_discovery import (
                LISTING_URL,
                build_requests_session,
                collect_professor_links,
                discover_listing_pages,
                extract_image_urls,
            )

            def pretty(data: Any) -> None:
                print(json.dumps(data, ensure_ascii=False, indent=2))

            settings = TutorSettings.from_env(PROJECT_ROOT / ".env")
            model = build_model(settings)
            ocr_model = build_ocr_model(settings)

            LAB_ROOT = PROJECT_ROOT / "lab_4_deep_agents"
            SOURCE_DOSSIER_DIR = PROJECT_ROOT / "professors"
            SOURCE_SKILLS_DIR = LAB_ROOT / "skills"
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

            pretty(
                {
                    "project_root": str(PROJECT_ROOT),
                    "model": settings.lab_tutor_llm_model,
                    "ocr_model": settings.ocr_model,
                    "deepagents_version": version("deepagents"),
                    "runtime_root": str(RUNTIME_ROOT),
                    "starter_dossiers": STARTER_DOSSIERS,
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
                    "deep_agent": "one harness with todos, files, and delegated subagents",
                    "lab_4_focus": "prompt engineering + files + one skill-driven subagent",
                }
            )
            '''
        ),
        md(
            """
            ## 3. Prepare the Runtime Workspace

            Learning goal: create a safe file sandbox for the lab and rebuild a fresh compact index from the existing dossier folder.

            Theory: the runtime root is the deep agent's world. We seed it from a curated starter subset of the repo's professor markdown files, but keep it separate under `artifacts/lab4/runtime/` so the lab can write files safely. The compact index is rebuilt from the starter dossiers at startup so the broad-question path starts from a consistent summary file and the add-by-URL path still has something new to add.

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
                (RUNTIME_ROOT / "skills").mkdir(parents=True, exist_ok=True)

                for dossier_name in STARTER_DOSSIERS:
                    shutil.copy2(
                        SOURCE_DOSSIER_DIR / dossier_name,
                        RUNTIME_ROOT / "professors" / dossier_name,
                    )
                shutil.copytree(SOURCE_SKILLS_DIR, RUNTIME_ROOT / "skills", dirs_exist_ok=True)

                summary_lines = rebuild_runtime_index(RUNTIME_ROOT)
                skill_path = RUNTIME_ROOT / "skills" / "add-professor-from-url" / "SKILL.md"
                index_preview = (RUNTIME_ROOT / "professors.md").read_text(encoding="utf-8").splitlines()[:10]

                return {
                    "runtime_root": str(RUNTIME_ROOT),
                    "dossier_count": len(list((RUNTIME_ROOT / "professors").glob("*.md"))),
                    "index_line_count": len(summary_lines),
                    "starter_dossiers": STARTER_DOSSIERS,
                    "skill_path": str(skill_path),
                    "index_preview": index_preview,
                }


            runtime_snapshot = seed_runtime_workspace(reset=True)
            pretty(runtime_snapshot)
            '''
        ),
        code(
            '''
            skill_preview = (
                RUNTIME_ROOT / "skills" / "add-professor-from-url" / "SKILL.md"
            ).read_text(encoding="utf-8")

            print("\\n".join(skill_preview.splitlines()[:40]))
            '''
        ),
        md(
            """
            ## 4. Define the Low-level OCR Tools

            Learning goal: keep the custom tool layer narrow and deterministic.

            Theory: the deep agent already knows how to read, write, list, and search files. We add only a tiny deterministic helper layer:

            - fetch the BIT page images,
            - turn those images into canonical dossier markdown,
            - rebuild the compact index after a successful add.

            ![Architecture step 04](workflow_steps/step_04.svg)
            """
        ),
        code(
            '''
            REQUESTS_SESSION = build_requests_session()


            def _append_unique(values: list[str], candidate: str) -> None:
                candidate = candidate.strip()
                if candidate and candidate not in values:
                    values.append(candidate)


            @tool
            def extract_image_urls_tool(detail_url: str) -> dict[str, Any]:
                """Fetch one BIT professor detail page and return the poster image URLs."""
                image_urls = extract_image_urls(detail_url, REQUESTS_SESSION)
                return {
                    "detail_url": detail_url,
                    "image_urls": image_urls,
                    "page_count": len(image_urls),
                }


            @tool
            def ocr_images_to_professor_markdown_tool(
                detail_url: str,
                image_urls: list[str],
                professor_name_hint: str = "",
            ) -> dict[str, Any]:
                """OCR poster images into canonical professor dossier markdown."""
                if not image_urls:
                    raise ValueError("No image URLs were provided for OCR.")

                page_notes_markdown = extract_professor_poster_notes(
                    model=ocr_model,
                    image_urls=image_urls,
                    session=REQUESTS_SESSION,
                )
                pages = parse_ocr_page_notes(page_notes_markdown)
                detected_names = extract_identity_candidates(pages)
                expected_name = professor_name_hint.strip() or (detected_names[0] if detected_names else "")

                supplemental_header_lines: list[str] = []
                if needs_top_block_fallback(
                    pages=pages,
                    expected_name=expected_name or "unknown",
                ):
                    supplemental_header_lines = extract_header_identity_lines(
                        model=ocr_model,
                        image_url=image_urls[0],
                        session=REQUESTS_SESSION,
                    )

                for candidate in extract_identity_candidates_from_lines(supplemental_header_lines):
                    _append_unique(detected_names, candidate)

                if not expected_name:
                    if not detected_names:
                        raise ValueError("OCR did not yield a reliable professor name.")
                    expected_name = detected_names[0]

                listing = ProfessorListing(name=expected_name, detail_url=detail_url)
                canonical_markdown = cleanup_markdown_artifact(
                    build_professor_markdown_from_page_notes(
                        listing=listing,
                        image_urls=image_urls,
                        page_notes_markdown=page_notes_markdown,
                        supplemental_header_lines=supplemental_header_lines,
                    )
                )
                validation = validate_professor_markdown(
                    markdown_text=canonical_markdown,
                    expected_name=expected_name,
                    expected_detail_url=detail_url,
                )
                if validation.status != "valid":
                    raise ValueError("; ".join(validation.notes) or "OCR validation failed")

                title_line = canonical_markdown.splitlines()[0].removeprefix("# ").strip()
                professor_name = title_line or expected_name
                slug = slugify_name(professor_name)

                return {
                    "professor_name": professor_name,
                    "slug": slug,
                    "detail_url": detail_url,
                    "page_count": len(image_urls),
                    "canonical_markdown": canonical_markdown,
                    "summary_line": build_cached_summary_line(canonical_markdown, professor_name),
                    "detected_names": detected_names,
                }


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


            pretty(
                {
                    "main_agent_custom_tools": [rebuild_professors_index_tool.name],
                    "subagent_custom_tools": [
                        extract_image_urls_tool.name,
                        ocr_images_to_professor_markdown_tool.name,
                    ],
                }
            )
            '''
        ),
        md(
            """
            ## 5. Build the Subagent and the Main Deep Agent

            Learning goal: keep the orchestration pure Deep Agents.

            Theory: the main agent uses built-in file tools plus the built-in `task` tool. The add workflow lives in one subagent with one skill. That is the whole design: no custom graph, no router nodes, and no separate supervisor.

            ![Architecture step 05](workflow_steps/step_05.svg)
            """
        ),
        code(
            '''
            BACKEND = FilesystemBackend(root_dir=str(RUNTIME_ROOT), virtual_mode=False)
            CHECKPOINTER = InMemorySaver()

            MAIN_SYSTEM_PROMPT = """
            You are ProfessorWorkspaceAgent for a local professor knowledge base.

            Core rules:
            - Work only with the files in the runtime workspace.
            - Use file tools and the built-in task tool. Do not use execute in this lab.
            - Start substantial tasks with write_todos.
            - For broad questions, read /professors.md first.
            - For named-professor questions, start from /professors.md and then open the matching /professors/<slug>.md only if you need more detail.
            - If the user gives a BIT professor detail URL or asks to add a professor from a URL, always delegate that work to AddProfessorSubagent.
            - If AddProfessorSubagent returns status=added, you MUST call rebuild_professors_index_tool before replying.
            - Rebuilding /professors.md after a successful add is mandatory, not optional.
            - The compact index format must stay:
              # BIT CSAT Professors
              - <Name>: <keyword1>, <keyword2>, <keyword3>[, <keyword4>]
            - When rebuilding the compact index, sort alphabetically by professor name.
            - Do not try to hand-edit /professors.md line by line when rebuild_professors_index_tool can do it deterministically.
            - After a successful add and index rebuild, reply with a short confirmation that names the professor and the new dossier path.
            - Keep answers grounded in the local files.
            """.strip()

            ADD_SUBAGENT_PROMPT = """
            You are AddProfessorSubagent.

            Goal:
            - Add exactly one BIT professor dossier from one official detail URL.

            Rules:
            - Start with write_todos.
            - Use the add-professor-from-url skill for the exact workflow.
            - Use file tools to inspect existing dossiers and to write the new dossier file.
            - Use extract_image_urls_tool before OCR.
            - Use ocr_images_to_professor_markdown_tool to obtain canonical dossier markdown.
            - Write exactly one dossier file at /professors/<slug>.md when the professor is new.
            - If a matching detail_url or slug already exists, do not create a duplicate file.
            - After the dossier write succeeds, stop immediately and return to the parent agent.
            - Do not rebuild /professors.md. The parent agent handles that.
            - Do not inspect unrelated files after the dossier write is complete.
            - Do not use execute.
            - Return exactly one compact final line with status, professor_name, slug, markdown_path, and page_count.
            """.strip()

            add_professor_subagent = {
                "name": "AddProfessorSubagent",
                "description": "Use this agent when a user provides a BIT detail URL and wants that professor added to the local workspace.",
                "system_prompt": ADD_SUBAGENT_PROMPT,
                "model": model,
                "tools": [
                    extract_image_urls_tool,
                    ocr_images_to_professor_markdown_tool,
                ],
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


            def safe_state_values(thread_id: str) -> dict[str, Any]:
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    snapshot = professor_workspace_agent.get_state(config)
                except Exception:
                    return {}
                return dict(getattr(snapshot, "values", {}) or {})


            def extract_tool_steps(messages: list[BaseMessage]) -> list[str]:
                steps: list[str] = []
                for message in messages:
                    if not isinstance(message, AIMessage):
                        continue
                    for tool_call in message.tool_calls or []:
                        args = tool_call.get("args") or {}
                        summary_parts: list[str] = []
                        if isinstance(args, dict):
                            for key in ("subagent_type", "detail_url", "file_path", "path", "pattern"):
                                value = args.get(key)
                                if value:
                                    summary_parts.append(f"{key}={value}")
                        suffix = f" [{', '.join(summary_parts[:2])}]" if summary_parts else ""
                        steps.append(f"{tool_call['name']}{suffix}")
                return steps


            def final_reply(messages: list[BaseMessage]) -> str:
                for message in reversed(messages):
                    if isinstance(message, AIMessage) and not (message.tool_calls or []):
                        return stringify_content(message.content).strip()
                return ""


            def normalize_todos(todos: Any) -> list[str]:
                if not todos:
                    return []
                normalized: list[str] = []
                for item in todos:
                    if isinstance(item, dict):
                        title = item.get("title") or item.get("content") or item.get("todo") or "todo"
                        status = item.get("status") or "unknown"
                        normalized.append(f"{status}: {title}")
                    else:
                        normalized.append(str(item))
                return normalized


            def run_turn(user_text: str, *, thread_id: str) -> dict[str, Any]:
                config = {"configurable": {"thread_id": thread_id}}
                before_values = safe_state_values(thread_id)
                before_messages = list(before_values.get("messages", []))
                result = professor_workspace_agent.invoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config=config,
                )
                new_messages = result["messages"][len(before_messages):]
                return {
                    "thread_id": thread_id,
                    "assistant_reply": final_reply(new_messages) or final_reply(result["messages"]),
                    "tool_steps": extract_tool_steps(new_messages),
                    "todos": normalize_todos(result.get("todos", [])),
                    "message_delta": len(new_messages),
                }


            pretty(
                {
                    "runtime_root": str(RUNTIME_ROOT),
                    "subagent_skill_path": "/skills/add-professor-from-url/SKILL.md",
                    "main_agent_name": "ProfessorWorkspaceAgent",
                    "subagent_name": add_professor_subagent["name"],
                }
            )
            '''
        ),
        md(
            """
            ## 6. Scenario A: Broad Question From the Compact Index

            Learning goal: show the cheapest useful path. The main agent should answer from `/professors.md` without OCR and without subagent delegation.
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
            ## 8. Scenario C: Add a Professor From a Detail URL

            Learning goal: show the deep-agent-only maintenance flow:

            - the main agent writes todos,
            - delegates to `AddProfessorSubagent`,
            - the subagent uses the skill plus the two low-level OCR tools,
            - a new dossier file is written,
            - the main agent rebuilds `/professors.md`.

            The helper below tries to find one official BIT professor detail URL that is not already present in the local runtime workspace. If the network is unavailable, the cell will skip cleanly instead of crashing the notebook.

            ![Architecture step 06](workflow_steps/step_06.svg)
            """
        ),
        code(
            '''
            def runtime_detail_url_index() -> set[str]:
                detail_urls: set[str] = set()
                for markdown_path in (RUNTIME_ROOT / "professors").glob("*.md"):
                    metadata = read_professor_markdown_metadata(markdown_path)
                    detail_url = metadata.get("detail_url")
                    if detail_url:
                        detail_urls.add(str(detail_url))
                return detail_urls


            def find_missing_listings(limit: int = 3) -> list[ProfessorListing]:
                listing_pages = discover_listing_pages(LISTING_URL, REQUESTS_SESSION)
                listings = collect_professor_links(listing_pages, REQUESTS_SESSION)
                existing_urls = runtime_detail_url_index()
                return [listing for listing in listings if listing.detail_url not in existing_urls][:limit]


            add_turn = None
            added_name = None
            added_slug = None
            added_files: list[str] = []
            candidate_payload: dict[str, Any] | None = None

            before_files = {path.name for path in (RUNTIME_ROOT / "professors").glob("*.md")}
            try:
                missing_candidates = find_missing_listings(limit=2)
            except Exception as exc:
                missing_candidates = []
                discovery_error = str(exc)
            else:
                discovery_error = None

            if not missing_candidates:
                pretty(
                    {
                        "status": "skipped",
                        "reason": discovery_error or "No missing professor URL was discovered from the BIT listing pages.",
                    }
                )
            else:
                candidate = missing_candidates[0]
                candidate_payload = candidate.to_dict()
                add_turn = run_turn(
                    f"Add this BIT professor to the workspace from the detail URL: {candidate.detail_url}",
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
                        "candidate": candidate_payload,
                        "turn": add_turn,
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
            - keep the long workflow in a **skill** instead of overloading the system prompt,
            - use the built-in **file tools** as the main working surface,
            - delegate one isolated workflow to a **subagent** with the built-in `task` tool.

            We intentionally do **not** use `execute` in the core lesson. If you later swap the backend to a real sandbox backend, the same deep-agent design can grow into command execution too. That is a backend change, not a control-flow redesign.
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
