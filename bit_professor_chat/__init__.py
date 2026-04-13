"""Helpers for querying the BIT professor graph through Neo4j MCP."""

from typing import TYPE_CHECKING

from .config import TutorSettings

if TYPE_CHECKING:
    from .mcp_agent import (
        AgentReply,
        ProfessorGraphChatService,
        ToolTrace,
        extract_final_answer,
        extract_tool_traces,
    )

__all__ = [
    "AgentReply",
    "ProfessorGraphChatService",
    "ToolTrace",
    "TutorSettings",
    "extract_final_answer",
    "extract_tool_traces",
]


def __getattr__(name: str):
    if name in {
        "AgentReply",
        "ProfessorGraphChatService",
        "ToolTrace",
        "extract_final_answer",
        "extract_tool_traces",
    }:
        from .mcp_agent import (
            AgentReply,
            ProfessorGraphChatService,
            ToolTrace,
            extract_final_answer,
            extract_tool_traces,
        )

        exports = {
            "AgentReply": AgentReply,
            "ProfessorGraphChatService": ProfessorGraphChatService,
            "ToolTrace": ToolTrace,
            "extract_final_answer": extract_final_answer,
            "extract_tool_traces": extract_tool_traces,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
