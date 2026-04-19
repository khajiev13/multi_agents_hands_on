from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware.tool_retry import ToolRetryMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from .config import TutorSettings

SYSTEM_PROMPT = """
You are the BIT Professor Graph Assistant for Beijing Institute of Technology.

You answer student questions by querying Neo4j through the MCP tools provided to you.
Use only database evidence returned by the tools. If the graph does not contain an answer,
say that clearly instead of guessing.

Current graph conventions:
- Professors use the label `Professor` with fields like `name`, `aliases`, `detail_url`,
  `title`, `school_name`, `discipline`, `emails`, and `phones`.
- Related typed labels include `Organization`, `ResearchTopic`, `Experience`,
  `Publication`, and `Award`.
- Common relationship types include `AFFILIATED_WITH`, `HAS_RESEARCH_INTEREST`,
  `HAS_EXPERIENCE`, `AUTHORED`, `RECEIVED`, `AT`, and `GRANTED_BY`.
- This is a typed professor graph, not a generic `Entity` graph.

Working style:
1. Inspect the schema first when the shape of the graph is uncertain.
2. Resolve the professor or topic with simple Cypher first, then drill down.
3. When multiple independent read-only graph checks would help, prefer parallel tool calls in one turn instead of serializing them.
4. Prefer simple `MATCH ... RETURN ... LIMIT ...` queries over clever Cypher.
5. If a Cypher query fails, read the error, simplify the query, and retry.
6. Never use `RETURN` inside a list comprehension; Cypher list comprehensions use `|`.
7. Prefer concise answers with concrete names and relationships.
8. Mention when an answer is partial because the graph is incomplete.
""".strip()


@dataclass(frozen=True)
class ToolTrace:
    """One tool execution paired with the emitted arguments and a short result preview."""

    name: str
    args: dict[str, Any]
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentReply:
    """Structured result for one agent turn, including loop-inspection metadata."""

    answer: str
    tool_traces: list[ToolTrace]
    elapsed_seconds: float
    available_tools: list[str]
    llm_turn_count: int
    tool_call_count: int
    tool_calls_per_turn: list[int]


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def extract_final_answer(messages: Sequence[Any]) -> str:
    """Return the last AI message that is not asking for more tool calls."""

    for message in reversed(messages):
        if message.__class__.__name__ != "AIMessage":
            continue
        if getattr(message, "tool_calls", None):
            continue
        return stringify_content(message.content)
    return stringify_content(messages[-1].content) if messages else ""


def extract_tool_traces(messages: Sequence[Any]) -> list[ToolTrace]:
    """Rebuild a readable tool trace by matching ToolMessages back to tool call ids."""

    tool_args_by_call_id: dict[str, dict[str, Any]] = {}
    traces: list[ToolTrace] = []

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tool_args_by_call_id[tool_call["id"]] = {
                "name": tool_call["name"],
                "args": tool_call.get("args", {}),
            }

        if message.__class__.__name__ != "ToolMessage":
            continue

        tool_call_id = getattr(message, "tool_call_id", "")
        tool_metadata = tool_args_by_call_id.get(tool_call_id, {})
        traces.append(
            ToolTrace(
                name=getattr(message, "name", None)
                or tool_metadata.get("name", "tool"),
                args=tool_metadata.get("args", {}),
                content_preview=truncate_text(
                    stringify_content(message.content), 1800
                ),
            )
        )

    return traces


def extract_llm_turn_count(messages: Sequence[Any]) -> int:
    return sum(1 for message in messages if message.__class__.__name__ == "AIMessage")


def extract_tool_calls_per_turn(messages: Sequence[Any]) -> list[int]:
    return [
        len(message.tool_calls)
        for message in messages
        if message.__class__.__name__ == "AIMessage" and getattr(message, "tool_calls", None)
    ]


class ProfessorGraphChatService:
    """Thin service wrapper around a LangChain agent that uses Neo4j MCP tools."""

    def __init__(
        self,
        *,
        settings: TutorSettings,
        client: MultiServerMCPClient,
        agent: Any,
        tool_names: list[str],
    ) -> None:
        self.settings = settings
        self._client = client
        self._agent = agent
        self.tool_names = tool_names

    @classmethod
    async def create(
        cls, settings: TutorSettings | None = None
    ) -> "ProfessorGraphChatService":
        """Create the MCP client, fetch tools, and build the LangChain agent."""

        settings = settings or TutorSettings.from_env()
        model = ChatOpenAI(
            model=settings.lab_tutor_llm_model,
            api_key=settings.lab_tutor_llm_api_key,
            base_url=settings.lab_tutor_llm_base_url,
            temperature=0,
        )
        client = MultiServerMCPClient(
            {"neo4j": settings.neo4j_mcp_connection()},
            tool_name_prefix=True,
        )
        tools = await client.get_tools()
        agent = create_agent(
            model,
            tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[ToolRetryMiddleware(max_retries=1, on_failure="continue")],
            name="bit_professor_mcp_agent",
        )
        return cls(
            settings=settings,
            client=client,
            agent=agent,
            tool_names=[tool.name for tool in tools],
        )

    async def ask(
        self,
        question: str,
        conversation: Sequence[dict[str, str]] | None = None,
    ) -> AgentReply:
        """Run one question through the agent and expose answer, timing, and tool stats."""

        messages = list(conversation or [])
        messages.append({"role": "user", "content": question})
        start_time = time.perf_counter()
        result = await self._agent.ainvoke({"messages": messages})
        elapsed_seconds = time.perf_counter() - start_time
        result_messages = result["messages"]
        final_answer = extract_final_answer(result_messages)
        traces = extract_tool_traces(result_messages)
        tool_calls_per_turn = extract_tool_calls_per_turn(result_messages)
        return AgentReply(
            answer=final_answer,
            tool_traces=traces,
            elapsed_seconds=elapsed_seconds,
            available_tools=list(self.tool_names),
            llm_turn_count=extract_llm_turn_count(result_messages),
            tool_call_count=sum(tool_calls_per_turn),
            tool_calls_per_turn=tool_calls_per_turn,
        )


async def _run_cli(question: str, show_trace: bool) -> None:
    service = await ProfessorGraphChatService.create()
    reply = await service.ask(question)
    print(reply.answer)
    if show_trace:
        print("\nAgent timing:")
        print(json.dumps({"elapsed_seconds": round(reply.elapsed_seconds, 3)}, ensure_ascii=False))
        print("\nAgent loop stats:")
        print(
            json.dumps(
                {
                    "llm_turn_count": reply.llm_turn_count,
                    "tool_call_count": reply.tool_call_count,
                    "tool_calls_per_turn": reply.tool_calls_per_turn,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("\nAvailable MCP tools:")
        print(json.dumps(reply.available_tools, ensure_ascii=False, indent=2))
        if not reply.tool_traces:
            print("\nCypher trace:\n")
            print("The agent answered without calling MCP tools.")
            return

        print("\nCypher trace:")
        for index, trace in enumerate(reply.tool_traces, start=1):
            print(f"\n{index}. {trace.name}")
            print(json.dumps(trace.args, ensure_ascii=False, indent=2))
            print(trace.content_preview)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the BIT professor Neo4j graph through MCP."
    )
    parser.add_argument("question", help="Natural-language question for the graph")
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print MCP tool arguments and truncated results",
    )
    args = parser.parse_args()
    asyncio.run(_run_cli(args.question, args.show_trace))


if __name__ == "__main__":
    main()
