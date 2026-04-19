"""
workflow_render.py - Build and render the Lab 3 workflow diagrams.

This script maintains one diagram model and writes:
  - lab_3_swarm.svg
  - workflow_steps/step_01.svg ... workflow_steps/step_08.svg

The master diagram is meant to teach the full Lab 3 swarm story in one view:
environment setup, graph seeding, minimal swarm state, explicit tool binding,
specialist subgraphs, handoff commands, and same-thread memory.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent
MASTER_SVG = ROOT / "lab_3_swarm.svg"
STEPS_DIR = ROOT / "workflow_steps"

CANVAS_W = 1960
CANVAS_H = 1260
FONT_FAMILY = "Inter, Arial, sans-serif"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"
DIM_OPACITY = 0.18


@dataclass(frozen=True)
class Box:
    element_id: str
    label: str | None
    x: int
    y: int
    width: int
    height: int
    fill: str
    stroke: str
    shape: str = "rectangle"
    dashed: bool = False
    font_size: int = 18
    stroke_width: float = 2.6
    text_color: str = TEXT_DARK

    @property
    def text_id(self) -> str:
        return f"{self.element_id}-text"

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def left_center(self) -> tuple[float, float]:
        return (self.x, self.y + self.height / 2)

    @property
    def right_center(self) -> tuple[float, float]:
        return (self.x + self.width, self.y + self.height / 2)

    @property
    def top_center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height)


@dataclass(frozen=True)
class Arrow:
    element_id: str
    points: list[tuple[float, float]]
    stroke: str = "#64748b"
    dashed: bool = False


@dataclass(frozen=True)
class StepSpec:
    number: int
    title: str
    subtitle: str
    highlight_ids: set[str]


TITLE = "Lab 3: Manual Tool-Calling Swarm"
SUBTITLE = (
    "A teaching view of the whole build and runtime path: explicit tools, "
    "bind_tools(...), ToolNode loops, handoffs, and thread_id memory."
)


BOXES = [
    Box("build-pill", "Build Sequence", 70, 160, 220, 42, fill="#fff7ed", stroke="#ea580c", font_size=20),
    Box("student-pill", "Student Turns", 70, 350, 200, 42, fill="#eef2ff", stroke="#4f46e5", font_size=20),
    Box("runtime-pill", "Runtime Walkthrough", 340, 350, 250, 42, fill="#ecfeff", stroke="#0891b2", font_size=20),
    Box("time-label", "Time", 44, 570, 72, 34, fill="#ffffff", stroke="#94a3b8", font_size=16),
    Box(
        "step-01",
        "Step 1\nEnv + imports\nTutorSettings,\nbuild_model",
        70,
        214,
        205,
        100,
        fill="#eff6ff",
        stroke="#2563eb",
        font_size=15,
    ),
    Box(
        "step-02",
        "Step 2\nNeo4j up + seed\ndocker compose +\ninsert_structured_output",
        295,
        214,
        205,
        100,
        fill="#ecfdf5",
        stroke="#16a34a",
        font_size=15,
    ),
    Box(
        "step-03",
        "Step 3\nMinimal state\nmessages +\nactive_agent",
        520,
        214,
        205,
        100,
        fill="#fffbeb",
        stroke="#d97706",
        font_size=15,
    ),
    Box(
        "step-04",
        "Step 4\nDeterministic tools\n@tool wrappers over\nNeo4jQueryService",
        745,
        214,
        205,
        100,
        fill="#ecfeff",
        stroke="#0891b2",
        font_size=15,
    ),
    Box(
        "step-05",
        "Step 5\nCustom handoffs\nhandoff tool returns\nCommand(...)",
        970,
        214,
        205,
        100,
        fill="#fff7ed",
        stroke="#ea580c",
        font_size=15,
    ),
    Box(
        "step-06",
        "Step 6\nBind tools\nmodel.bind_tools([...])\nper specialist",
        1195,
        214,
        205,
        100,
        fill="#eef2ff",
        stroke="#4f46e5",
        font_size=15,
    ),
    Box(
        "step-07",
        "Step 7\nSpecialist graph\nSTART -> llm -> ToolNode\n-> llm | END",
        1420,
        214,
        205,
        100,
        fill="#fdf2f8",
        stroke="#db2777",
        font_size=15,
    ),
    Box(
        "step-08",
        "Step 8\nCompile + run\ncreate_swarm + InMemorySaver\nstream/invoke with thread_id",
        1645,
        214,
        205,
        100,
        fill="#f5f3ff",
        stroke="#7c3aed",
        font_size=15,
    ),
    Box("student-column", None, 70, 400, 240, 740, fill="#f8fafc", stroke="#cbd5e1", dashed=True),
    Box("student-session", "Same Thread Demo", 94, 428, 192, 44, fill="#eef2ff", stroke="#4f46e5", font_size=18),
    Box("student", "Student", 112, 506, 152, 84, fill="#e7f5ff", stroke="#1971c2", shape="ellipse", font_size=22),
    Box(
        "turn-1",
        "Turn 1\nWhich professors match\ncomputer vision + multimedia?",
        88,
        640,
        204,
        104,
        fill="#ffffff",
        stroke="#334155",
        font_size=17,
    ),
    Box(
        "turn-2",
        "Turn 2\nWhat about CHENG Cheng\nspecifically?",
        88,
        846,
        204,
        104,
        fill="#ffffff",
        stroke="#334155",
        font_size=17,
    ),
    Box(
        "reuse-thread",
        "Reuse the same\nthread_id so the swarm\nremembers active_agent",
        88,
        1034,
        204,
        78,
        fill="#fffbeb",
        stroke="#d97706",
        font_size=16,
    ),
    Box("swarm-container", None, 340, 400, 1550, 740, fill="#ffffff", stroke="#111827", stroke_width=3.2),
    Box(
        "swarm-parent",
        "create_swarm([...],\ndefault_active_agent=\"ProfessorLookupAgent\")",
        390,
        442,
        610,
        92,
        fill="#fff7ed",
        stroke="#ea580c",
        font_size=19,
    ),
    Box(
        "handoff-command",
        "Handoff tool returns\nCommand(...)\nupdates parent messages +\nactive_agent",
        1030,
        442,
        330,
        92,
        fill="#fff7ed",
        stroke="#ea580c",
        dashed=True,
        font_size=15,
    ),
    Box(
        "checkpoint",
        "compile(checkpointer=InMemorySaver())\nthread_id -> messages + active_agent",
        1390,
        442,
        440,
        92,
        fill="#fffbeb",
        stroke="#d97706",
        font_size=18,
    ),
    Box("professor-lane", None, 390, 570, 460, 540, fill="#eff6ff", stroke="#60a5fa", dashed=True),
    Box("professor-header", "ProfessorLookupAgent", 418, 596, 404, 56, fill="#dbeafe", stroke="#2563eb", font_size=22),
    Box(
        "professor-prompt",
        "System prompt:\nHandle named professor questions",
        430,
        680,
        380,
        76,
        fill="#ffffff",
        stroke="#60a5fa",
        font_size=17,
    ),
    Box(
        "professor-bind",
        "bound_model = model.bind_tools([\nresolve_professor_tool,\nget_professor_facts_tool,\ntransfer_to_research_match,\n])",
        430,
        778,
        380,
        110,
        fill="#f8fafc",
        stroke="#64748b",
        dashed=True,
        font_size=15,
    ),
    Box(
        "professor-route",
        "route_after_model: tool_calls ? tools : END",
        430,
        905,
        380,
        40,
        fill="#f8fafc",
        stroke="#94a3b8",
        dashed=True,
        font_size=15,
    ),
    Box("professor-llm", "llm", 430, 970, 150, 70, fill="#eef2ff", stroke="#4f46e5", font_size=24),
    Box("professor-toolnode", "ToolNode", 660, 970, 150, 70, fill="#ede9fe", stroke="#7c3aed", font_size=22),
    Box(
        "professor-resolve",
        "resolve_\nprofessor_tool",
        430,
        1060,
        170,
        60,
        fill="#ecfeff",
        stroke="#0891b2",
        font_size=15,
    ),
    Box(
        "professor-facts",
        "get_professor_\nfacts_tool",
        430,
        1132,
        170,
        60,
        fill="#ecfeff",
        stroke="#0891b2",
        font_size=15,
    ),
    Box(
        "professor-handoff",
        "transfer_to_\nresearch_match",
        620,
        1088,
        190,
        80,
        fill="#fff7ed",
        stroke="#ea580c",
        font_size=16,
    ),
    Box("research-lane", None, 880, 570, 460, 540, fill="#ecfdf5", stroke="#4ade80", dashed=True),
    Box("research-header", "ResearchMatchAgent", 908, 596, 404, 56, fill="#dcfce7", stroke="#16a34a", font_size=22),
    Box(
        "research-prompt",
        "System prompt:\nRecommend professors for topics",
        920,
        680,
        380,
        76,
        fill="#ffffff",
        stroke="#4ade80",
        font_size=17,
    ),
    Box(
        "research-bind",
        "bound_model = model.bind_tools([\nfind_professors_by_topics_tool,\ntransfer_to_professor_lookup,\n])",
        920,
        778,
        380,
        96,
        fill="#f8fafc",
        stroke="#64748b",
        dashed=True,
        font_size=15,
    ),
    Box(
        "research-route",
        "route_after_model: tool_calls ? tools : END",
        920,
        905,
        380,
        40,
        fill="#f8fafc",
        stroke="#94a3b8",
        dashed=True,
        font_size=15,
    ),
    Box("research-llm", "llm", 920, 970, 150, 70, fill="#ecfdf5", stroke="#16a34a", font_size=24),
    Box("research-toolnode", "ToolNode", 1150, 970, 150, 70, fill="#ede9fe", stroke="#7c3aed", font_size=22),
    Box(
        "research-find",
        "find_professors_by_\ntopics_tool",
        920,
        1080,
        180,
        60,
        fill="#ecfeff",
        stroke="#0891b2",
        font_size=15,
    ),
    Box(
        "research-handoff",
        "transfer_to_\nprofessor_lookup",
        1110,
        1088,
        190,
        80,
        fill="#fff7ed",
        stroke="#ea580c",
        font_size=16,
    ),
    Box("services-lane", None, 1370, 570, 460, 540, fill="#f8fafc", stroke="#94a3b8", dashed=True),
    Box("services-header", "Shared Services + Trace", 1396, 596, 408, 56, fill="#ecfeff", stroke="#0891b2", font_size=21),
    Box(
        "neo4j-service",
        "Neo4jQueryService\nresolve_professor(...)\nget_professor_facts(...)\nfind_professors_by_topics(...)",
        1410,
        680,
        380,
        180,
        fill="#f0fdf4",
        stroke="#16a34a",
        font_size=18,
    ),
    Box(
        "trace-box",
        "emit_teaching_event(...)\nstream(..., stream_mode=\"custom\",\nsubgraphs=True)",
        1410,
        894,
        380,
        110,
        fill="#eff6ff",
        stroke="#2563eb",
        font_size=17,
    ),
    Box(
        "scenario-box",
        "Runtime story:\nTurn 1 may hand off to ResearchMatchAgent.\nTurn 2 on the same thread_id can hand back\nto ProfessorLookupAgent.",
        1410,
        1032,
        380,
        120,
        fill="#f5f3ff",
        stroke="#7c3aed",
        font_size=16,
    ),
]
BOX_BY_ID = {box.element_id: box for box in BOXES}


ARROWS = [
    Arrow("time-arrow", [(76, 610), (76, 1116)], stroke="#64748b"),
    Arrow("turn1-to-parent", [BOX_BY_ID["turn-1"].right_center, (330, 692), (330, 488), BOX_BY_ID["swarm-parent"].left_center], stroke="#334155"),
    Arrow("turn2-to-parent", [BOX_BY_ID["turn-2"].right_center, (330, 898), (330, 488), BOX_BY_ID["swarm-parent"].left_center], stroke="#334155", dashed=True),
    Arrow("checkpoint-to-parent", [BOX_BY_ID["checkpoint"].left_center, (1340, 488), (1340, 488), BOX_BY_ID["swarm-parent"].right_center], stroke="#d97706", dashed=True),
    Arrow("parent-to-professor", [(620, 534), (620, 596)], stroke="#2563eb"),
    Arrow("command-to-checkpoint", [BOX_BY_ID["handoff-command"].right_center, BOX_BY_ID["checkpoint"].left_center], stroke="#ea580c", dashed=True),
    Arrow("professor-prompt-to-bind", [BOX_BY_ID["professor-prompt"].bottom_center, BOX_BY_ID["professor-bind"].top_center], stroke="#2563eb"),
    Arrow("professor-bind-to-llm", [BOX_BY_ID["professor-bind"].bottom_center, (620, 930), (505, 930), BOX_BY_ID["professor-llm"].top_center], stroke="#4f46e5"),
    Arrow("professor-llm-to-toolnode", [BOX_BY_ID["professor-llm"].right_center, BOX_BY_ID["professor-toolnode"].left_center], stroke="#7c3aed"),
    Arrow("professor-toolnode-loop", [BOX_BY_ID["professor-toolnode"].top_center, (735, 944), (505, 944), BOX_BY_ID["professor-llm"].top_center], stroke="#7c3aed", dashed=True),
    Arrow("professor-toolnode-to-resolve", [BOX_BY_ID["professor-toolnode"].bottom_center, (735, 1068), (515, 1068), BOX_BY_ID["professor-resolve"].top_center], stroke="#0891b2"),
    Arrow("professor-toolnode-to-facts", [BOX_BY_ID["professor-toolnode"].bottom_center, (735, 1160), (515, 1160), BOX_BY_ID["professor-facts"].top_center], stroke="#0891b2"),
    Arrow("professor-toolnode-to-handoff", [BOX_BY_ID["professor-toolnode"].bottom_center, BOX_BY_ID["professor-handoff"].top_center], stroke="#ea580c"),
    Arrow("professor-resolve-to-neo4j", [BOX_BY_ID["professor-resolve"].right_center, (1340, 1090), (1340, 730), BOX_BY_ID["neo4j-service"].left_center], stroke="#16a34a", dashed=True),
    Arrow("professor-facts-to-neo4j", [BOX_BY_ID["professor-facts"].right_center, (1360, 1162), (1360, 770), (1410, 770)], stroke="#16a34a", dashed=True),
    Arrow("research-prompt-to-bind", [BOX_BY_ID["research-prompt"].bottom_center, BOX_BY_ID["research-bind"].top_center], stroke="#16a34a"),
    Arrow("research-bind-to-llm", [BOX_BY_ID["research-bind"].bottom_center, (1110, 930), (995, 930), BOX_BY_ID["research-llm"].top_center], stroke="#16a34a"),
    Arrow("research-llm-to-toolnode", [BOX_BY_ID["research-llm"].right_center, BOX_BY_ID["research-toolnode"].left_center], stroke="#7c3aed"),
    Arrow("research-toolnode-loop", [BOX_BY_ID["research-toolnode"].top_center, (1225, 944), (995, 944), BOX_BY_ID["research-llm"].top_center], stroke="#7c3aed", dashed=True),
    Arrow("research-toolnode-to-find", [BOX_BY_ID["research-toolnode"].bottom_center, (1225, 1104), (1010, 1104), BOX_BY_ID["research-find"].top_center], stroke="#0891b2"),
    Arrow("research-toolnode-to-handoff", [BOX_BY_ID["research-toolnode"].bottom_center, BOX_BY_ID["research-handoff"].top_center], stroke="#ea580c"),
    Arrow("research-find-to-neo4j", [BOX_BY_ID["research-find"].right_center, (1380, 1110), (1380, 806), (1410, 806)], stroke="#16a34a", dashed=True),
    Arrow("professor-handoff-to-command", [BOX_BY_ID["professor-handoff"].top_center, (715, 554), (1195, 554), BOX_BY_ID["handoff-command"].bottom_center], stroke="#ea580c", dashed=True),
    Arrow("research-handoff-to-command", [BOX_BY_ID["research-handoff"].top_center, (1205, 554), BOX_BY_ID["handoff-command"].bottom_center], stroke="#ea580c", dashed=True),
    Arrow("command-to-research", [BOX_BY_ID["handoff-command"].bottom_center, (1195, 570), (1110, 570), BOX_BY_ID["research-header"].top_center], stroke="#ea580c", dashed=True),
    Arrow("command-to-professor", [BOX_BY_ID["handoff-command"].bottom_center, (1195, 550), (620, 550), BOX_BY_ID["professor-header"].top_center], stroke="#ea580c", dashed=True),
    Arrow("parent-to-trace", [BOX_BY_ID["swarm-parent"].right_center, (1300, 488), (1300, 948), BOX_BY_ID["trace-box"].left_center], stroke="#2563eb", dashed=True),
    Arrow("trace-to-scenario", [BOX_BY_ID["trace-box"].bottom_center, BOX_BY_ID["scenario-box"].top_center], stroke="#7c3aed", dashed=True),
]


STEPS = [
    StepSpec(
        1,
        "Environment Check",
        "Load settings, build the model, and prepare the shared Neo4j service.",
        {"build-pill", "step-01", "neo4j-service", "services-lane", "services-header"},
    ),
    StepSpec(
        2,
        "Neo4j Seeded",
        "Start Docker Neo4j and load the committed structured professor graph.",
        {"build-pill", "step-02", "neo4j-service", "services-lane", "services-header"},
    ),
    StepSpec(
        3,
        "Minimal Swarm State",
        "Lab 3 keeps just messages and active_agent, then keys memory by thread_id.",
        {"build-pill", "step-03", "student-column", "student-session", "reuse-thread", "swarm-parent", "checkpoint", "swarm-container", "turn1-to-parent", "turn2-to-parent"},
    ),
    StepSpec(
        4,
        "Deterministic Tools",
        "Each specialist gets a small Neo4j tool surface instead of an open tool bag.",
        {
            "build-pill",
            "step-04",
            "professor-resolve",
            "professor-facts",
            "research-find",
            "neo4j-service",
            "services-lane",
            "services-header",
            "professor-toolnode-to-resolve",
            "professor-toolnode-to-facts",
            "research-toolnode-to-find",
            "professor-resolve-to-neo4j",
            "professor-facts-to-neo4j",
            "research-find-to-neo4j",
        },
    ),
    StepSpec(
        5,
        "Custom Handoffs",
        "A handoff tool returns Command(...) to update the parent swarm and switch owners.",
        {
            "build-pill",
            "step-05",
            "handoff-command",
            "checkpoint",
            "professor-handoff",
            "research-handoff",
            "professor-header",
            "research-header",
            "command-to-checkpoint",
            "professor-handoff-to-command",
            "research-handoff-to-command",
            "command-to-research",
            "command-to-professor",
        },
    ),
    StepSpec(
        6,
        "Tool Binding",
        "Each specialist binds a different tool list before the model starts calling tools.",
        {
            "build-pill",
            "step-06",
            "professor-lane",
            "research-lane",
            "professor-header",
            "research-header",
            "professor-bind",
            "research-bind",
            "professor-prompt",
            "research-prompt",
            "professor-prompt-to-bind",
            "research-prompt-to-bind",
        },
    ),
    StepSpec(
        7,
        "Specialist Loops",
        "Both specialists are explicit START -> llm -> ToolNode -> llm graphs with a visible tool-call route.",
        {
            "build-pill",
            "step-07",
            "professor-lane",
            "research-lane",
            "professor-header",
            "research-header",
            "professor-route",
            "research-route",
            "professor-llm",
            "professor-toolnode",
            "research-llm",
            "research-toolnode",
            "professor-llm-to-toolnode",
            "research-llm-to-toolnode",
            "professor-toolnode-loop",
            "research-toolnode-loop",
            "professor-toolnode-to-resolve",
            "professor-toolnode-to-facts",
            "research-toolnode-to-find",
            "professor-toolnode-to-handoff",
            "research-toolnode-to-handoff",
        },
    ),
    StepSpec(
        8,
        "Compile + Runtime Memory",
        "The compiled swarm streams teaching events, and the same thread_id resumes from the remembered active agent.",
        {
            "student-pill",
            "runtime-pill",
            "step-08",
            "student-column",
            "student-session",
            "student",
            "turn-1",
            "turn-2",
            "reuse-thread",
            "swarm-container",
            "swarm-parent",
            "checkpoint",
            "trace-box",
            "scenario-box",
            "time-label",
            "time-arrow",
            "turn1-to-parent",
            "turn2-to-parent",
            "checkpoint-to-parent",
            "parent-to-professor",
            "parent-to-trace",
            "trace-to-scenario",
        },
    ),
]


def text_dimensions(label: str, font_size: int) -> tuple[int, int]:
    lines = label.split("\n")
    line_gap = font_size + 6
    width = max(int(len(line) * font_size * 0.56) for line in lines) if lines else 0
    height = line_gap * max(len(lines), 1)
    return width, height


def excalidraw_shape(box: Box, timestamp: int) -> list[dict]:
    common = {
        "id": box.element_id,
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
        "angle": 0,
        "strokeColor": box.stroke,
        "backgroundColor": box.fill,
        "fillStyle": "solid",
        "strokeWidth": box.stroke_width,
        "strokeStyle": "dashed" if box.dashed else "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": abs(hash(box.element_id)) % 100000,
        "version": 1,
        "versionNonce": abs(hash(f"{box.element_id}-nonce")) % 100000,
        "isDeleted": False,
        "updated": timestamp,
        "link": None,
        "locked": False,
    }
    shape = {**common, "type": box.shape}
    if box.label is None:
        shape["boundElements"] = []
        return [shape]

    width, height = text_dimensions(box.label, box.font_size)
    text_x = box.center_x - width / 2
    text_y = box.center_y - height / 2
    shape["boundElements"] = [{"type": "text", "id": box.text_id}]
    text = {
        "id": box.text_id,
        "type": "text",
        "x": text_x,
        "y": text_y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": box.text_color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": abs(hash(box.text_id)) % 100000,
        "version": 1,
        "versionNonce": abs(hash(f"{box.text_id}-nonce")) % 100000,
        "isDeleted": False,
        "boundElements": [],
        "updated": timestamp,
        "link": None,
        "locked": False,
        "text": box.label,
        "fontSize": box.font_size,
        "fontFamily": 1,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": box.element_id,
        "originalText": box.label,
        "lineHeight": 1.25,
    }
    return [shape, text]


def excalidraw_arrow(arrow: Arrow, timestamp: int) -> dict:
    start_x, start_y = arrow.points[0]
    relative_points = [
        [round(point_x - start_x, 2), round(point_y - start_y, 2)]
        for point_x, point_y in arrow.points
    ]
    xs = [point[0] for point in relative_points]
    ys = [point[1] for point in relative_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return {
        "id": arrow.element_id,
        "type": "arrow",
        "x": start_x,
        "y": start_y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": arrow.stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2.2,
        "strokeStyle": "dashed" if arrow.dashed else "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": abs(hash(arrow.element_id)) % 100000,
        "version": 1,
        "versionNonce": abs(hash(f"{arrow.element_id}-nonce")) % 100000,
        "isDeleted": False,
        "boundElements": [],
        "updated": timestamp,
        "link": None,
        "locked": False,
        "points": relative_points,
        "lastCommittedPoint": relative_points[-1],
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": len(relative_points) > 2,
    }


def write_excalidraw(path: Path) -> None:
    timestamp = int(time.time() * 1000)
    elements: list[dict] = []
    for box in BOXES:
        elements.extend(excalidraw_shape(box, timestamp))
    for arrow in ARROWS:
        elements.append(excalidraw_arrow(arrow, timestamp))

    document = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#f8fafc",
            "gridSize": None,
        },
        "files": {},
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def svg_box(box: Box, dimmed: bool) -> str:
    opacity = DIM_OPACITY if dimmed else 1.0
    shape_tag = (
        f'<ellipse cx="{box.center_x}" cy="{box.center_y}" rx="{box.width / 2}" ry="{box.height / 2}" '
        if box.shape == "ellipse"
        else f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="24" '
    )
    dash = ' stroke-dasharray="8 8"' if box.dashed else ""
    lines = [
        shape_tag
        + f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="{box.stroke_width}"{dash} opacity="{opacity}"/>'
    ]
    if box.label is None:
        return "\n".join(lines)

    box_lines = box.label.split("\n")
    line_gap = box.font_size + 6
    for index, line in enumerate(box_lines):
        dy = (index - (len(box_lines) - 1) / 2) * line_gap
        lines.append(
            f'<text x="{box.center_x}" y="{box.center_y + dy + box.font_size * 0.35}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="{box.font_size}" fill="{box.text_color}" opacity="{opacity}">{escape(line)}</text>'
        )
    return "\n".join(lines)


def svg_arrow(arrow: Arrow, dimmed: bool) -> str:
    opacity = DIM_OPACITY if dimmed else 1.0
    path_parts = [f"M {arrow.points[0][0]} {arrow.points[0][1]}"]
    for point_x, point_y in arrow.points[1:]:
        path_parts.append(f"L {point_x} {point_y}")
    dash = ' stroke-dasharray="8 8"' if arrow.dashed else ""
    end_x, end_y = arrow.points[-1]
    prev_x, prev_y = arrow.points[-2]
    dx = end_x - prev_x
    dy = end_y - prev_y
    if abs(dx) > abs(dy):
        arrow_head = [
            (end_x, end_y),
            (end_x - 12 if dx > 0 else end_x + 12, end_y - 7),
            (end_x - 12 if dx > 0 else end_x + 12, end_y + 7),
        ]
    else:
        arrow_head = [
            (end_x, end_y),
            (end_x - 7, end_y - 12 if dy > 0 else end_y + 12),
            (end_x + 7, end_y - 12 if dy > 0 else end_y + 12),
        ]

    return "\n".join(
        [
            f'<path d="{" ".join(path_parts)}" fill="none" stroke="{arrow.stroke}" stroke-width="2.5"{dash} opacity="{opacity}"/>',
            f'<polygon points="{" ".join(f"{x},{y}" for x, y in arrow_head)}" fill="{arrow.stroke}" opacity="{opacity}"/>',
        ]
    )


def render_svg(*, highlight_ids: set[str] | None, title: str, subtitle: str, output_path: Path) -> None:
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        '<rect x="24" y="24" width="1912" height="1212" rx="32" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>',
        f'<text x="70" y="88" font-family="{FONT_FAMILY}" font-size="38" font-weight="700" fill="{TEXT_DARK}">{escape(title)}</text>',
        f'<text x="70" y="126" font-family="{FONT_FAMILY}" font-size="19" fill="{TEXT_MUTED}">{escape(subtitle)}</text>',
    ]

    for arrow in ARROWS:
        dimmed = highlight_ids is not None and arrow.element_id not in highlight_ids
        elements.append(svg_arrow(arrow, dimmed))
    for box in BOXES:
        dimmed = highlight_ids is not None and box.element_id not in highlight_ids
        elements.append(svg_box(box, dimmed))

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Lab 3 workflow diagrams.")
    parser.parse_args()

    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    render_svg(highlight_ids=None, title=TITLE, subtitle=SUBTITLE, output_path=MASTER_SVG)

    for step in STEPS:
        render_svg(
            highlight_ids=step.highlight_ids,
            title=f"Step {step.number:02d}: {step.title}",
            subtitle=step.subtitle,
            output_path=STEPS_DIR / f"step_{step.number:02d}.svg",
        )

    print(f"Wrote {MASTER_SVG}")
    print(f"Wrote {STEPS_DIR}/step_01.svg ... step_08.svg")


if __name__ == "__main__":
    main()
