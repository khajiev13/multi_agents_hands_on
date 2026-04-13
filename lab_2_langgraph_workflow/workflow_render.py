"""
workflow_render.py - Build and render the Lab 2 workflow diagrams.

This script maintains one diagram model and writes:
  - lab_2_workflow.excalidraw
  - lab_2_workflow.svg
  - workflow_steps/step_01.svg ... workflow_steps/step_14.svg

The master Excalidraw diagram is the editable artifact for the teaching flow.
The SVG files are notebook-friendly renderings derived from the same model.
Lab 2 builds a Chroma collection from local markdown, then queries it with a
supervisor over retrieval-first worker agents and one add-by-URL ingestion worker.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent
MASTER_EXCALIDRAW = ROOT / "lab_2_workflow.excalidraw"
MASTER_SVG = ROOT / "lab_2_workflow.svg"
STEPS_DIR = ROOT / "workflow_steps"

CANVAS_W = 1760
CANVAS_H = 980
HEADER_Y = 54
PHASE_Y = 170
PHASE_H = 730
STEP_H = 96
DIM_OPACITY = 0.22
FONT_FAMILY = "Inter, Arial, sans-serif"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"
TEXT_SOFT = "#64748b"
CARD_FILL = "#ffffff"
CARD_SHADOW = "#f8fafc"
CANVAS_FILL = "#f8fafc"
CANVAS_STROKE = "#e2e8f0"
CONNECTOR_STROKE = "#64748b"
ARTIFACT_FILL = "#f8fafc"
ARTIFACT_TEXT = "#334155"
CURRENT_STROKE = "#e8590c"
CURRENT_FILL = "#fff7ed"
CURRENT_SHADOW = "#fed7aa"
COMPACT_BADGE_STEPS = {10, 11, 12}


DOCS = {
    "uv": "https://docs.astral.sh/uv/getting-started/",
    "python_pathlib": "https://docs.python.org/3/library/pathlib.html",
    "langgraph_graph_api": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "langgraph_workflows": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "langchain_agents": "https://docs.langchain.com/oss/python/langchain/agents",
    "langchain_supervisor": "https://docs.langchain.com/oss/python/langchain/supervisor",
    "neo4j_cypher": "https://neo4j.com/docs/cypher-manual/current/introduction/",
    "neo4j_python": "https://neo4j.com/docs/python-manual/current/",
}


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    title: str
    step_range: tuple[int, int]
    x: int
    width: int
    fill: str
    stroke: str
    title_color: str

    @property
    def y(self) -> int:
        return PHASE_Y

    @property
    def height(self) -> int:
        return PHASE_H


@dataclass(frozen=True)
class StepSpec:
    number: int
    phase_key: str
    title: str
    subtitle: str
    doc_url: str
    x: int
    y: int
    width: int
    height: int = STEP_H

    @property
    def step_id(self) -> str:
        return f"step-{self.number:02d}"


@dataclass(frozen=True)
class ArtifactSpec:
    step_number: int
    text: str
    x: int
    y: int
    width: int
    height: int = 52

    @property
    def artifact_id(self) -> str:
        return f"artifact-{self.step_number:02d}"


def wrap_label(text: str, limit: int) -> str:
    def expand_word(word: str) -> str:
        expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", word)
        expanded = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", expanded)
        return expanded

    wrapped_lines: list[str] = []
    for raw_line in text.split("\n"):
        words = []
        for raw_word in raw_line.split():
            words.extend(expand_word(raw_word).split())
        if not words:
            wrapped_lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= limit:
                current = candidate
            else:
                wrapped_lines.append(current)
                current = word
        wrapped_lines.append(current)
    return "\n".join(wrapped_lines)


PHASES = [
    PhaseSpec("setup", "Setup", (1, 2), 48, 280, "#f8fafc", "#94a3b8", "#334155"),
    PhaseSpec(
        "fanout",
        "Fan-Out Indexing",
        (3, 5),
        350,
        398,
        "#ecfdf5",
        "#34d399",
        "#065f46",
    ),
    PhaseSpec(
        "workers",
        "Workers",
        (6, 7),
        776,
        324,
        "#eff6ff",
        "#60a5fa",
        "#1d4ed8",
    ),
    PhaseSpec(
        "supervisor",
        "Supervisor Query",
        (8, 14),
        1120,
        604,
        "#f5f3ff",
        "#a78bfa",
        "#6d28d9",
    ),
]
PHASE_BY_KEY = {phase.key: phase for phase in PHASES}

STEPS = [
    StepSpec(1, "setup", "Environment Check", "runtime + imports", DOCS["uv"], 66, 278, 244),
    StepSpec(
        2,
        "setup",
        "LangGraph Mental Model",
        "fan-out + supervisor",
        DOCS["langgraph_workflows"],
        66,
        454,
        244,
    ),
    StepSpec(
        3,
        "fanout",
        "Discover Markdown Files",
        "professors/*.md",
        DOCS["python_pathlib"],
        378,
        224,
        344,
    ),
    StepSpec(
        4,
        "fanout",
        "Build Professor Tasks",
        "markdown -> task queue",
        DOCS["python_pathlib"],
        378,
        406,
        344,
    ),
    StepSpec(
        5,
        "fanout",
        "Send Plan",
        "one worker per markdown",
        DOCS["langgraph_workflows"],
        378,
        588,
        344,
    ),
    StepSpec(
        6,
        "workers",
        "Index Markdown Worker",
        "markdown -> metadata",
        DOCS["langgraph_graph_api"],
        806,
        286,
        264,
    ),
    StepSpec(
        7,
        "workers",
        "Fan In + Build Chroma",
        "reducer -> corpus ready",
        DOCS["langgraph_graph_api"],
        806,
        520,
        264,
    ),
    StepSpec(
        8,
        "supervisor",
        "Minimal Query State",
        "question, route, result",
        DOCS["langgraph_graph_api"],
        1154,
        216,
        536,
    ),
    StepSpec(
        9,
        "supervisor",
        "StudentAgent Supervisor",
        "route -> final answer",
        DOCS["langchain_supervisor"],
        1154,
        344,
        536,
    ),
    StepSpec(
        10,
        "supervisor",
        "ProfessorQueryAgent",
        "filtered RAG facts",
        DOCS["langchain_agents"],
        1154,
        518,
        168,
    ),
    StepSpec(
        11,
        "supervisor",
        "StudentProfessorAgent",
        "topic-match RAG",
        DOCS["langchain_agents"],
        1338,
        518,
        168,
    ),
    StepSpec(
        12,
        "supervisor",
        "AddTeacherAgent",
        "crawl + OCR + add",
        DOCS["langchain_agents"],
        1522,
        518,
        168,
    ),
    StepSpec(
        13,
        "supervisor",
        "Run Supervisor Scenarios",
        "routed graph",
        DOCS["langgraph_workflows"],
        1176,
        692,
        494,
    ),
    StepSpec(
        14,
        "supervisor",
        "Wrap-Up",
        "Lab 3 -> swarm",
        DOCS["langchain_supervisor"],
        1478,
        790,
        184,
        80,
    ),
]
STEP_BY_NUMBER = {step.number: step for step in STEPS}

ARTIFACTS = [
    ArtifactSpec(3, "Markdown files", 442, 340, 216, 48),
    ArtifactSpec(4, "Per-professor tasks", 432, 524, 236, 48),
    ArtifactSpec(5, "Send batch", 454, 708, 192, 48),
    ArtifactSpec(6, "dossier metadata", 854, 414, 168, 52),
    ArtifactSpec(7, "Markdown index + Chroma", 820, 636, 236, 52),
    ArtifactSpec(10, "Professor chunks", 1154, 624, 168, 52),
    ArtifactSpec(11, "Topic matches", 1338, 624, 168, 52),
    ArtifactSpec(12, "Add summary", 1522, 624, 168, 52),
    ArtifactSpec(13, "Final answer", 1286, 806, 176, 40),
]
ARTIFACT_BY_STEP = {artifact.step_number: artifact for artifact in ARTIFACTS}

# Each connector becomes visible once the destination step is reached.
CONNECTORS: list[tuple[int, int, str]] = [
    (1, 2, "vertical"),
    (2, 3, "handoff"),
    (3, 4, "vertical"),
    (4, 5, "vertical"),
    (5, 6, "handoff"),
    (6, 7, "vertical"),
    (7, 8, "handoff"),
    (8, 9, "vertical"),
    (9, 10, "branch_left"),
    (9, 11, "vertical"),
    (9, 12, "branch_right"),
    (10, 13, "merge_left"),
    (11, 13, "vertical"),
    (12, 13, "merge_right"),
    (13, 14, "footer_right"),
]


class IdFactory:
    def __init__(self) -> None:
        self.seed = 1000
        self.nonce = 5000

    def next_seed(self) -> int:
        self.seed += 97
        return self.seed

    def next_nonce(self) -> int:
        self.nonce += 4099
        return self.nonce


def base_shape(element_id: str, element_type: str, x: float, y: float, width: float, height: float, factory: IdFactory) -> dict:
    roundness = {"type": 3} if element_type in {"rectangle", "ellipse"} else None
    return {
        "id": element_id,
        "type": element_type,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#475569",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": roundness,
        "seed": factory.next_seed(),
        "version": 1,
        "versionNonce": factory.next_nonce(),
        "isDeleted": False,
        "boundElements": [],
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
    }


def make_rect(element_id: str, x: float, y: float, width: float, height: float, *, stroke: str, fill: str, factory: IdFactory, stroke_width: float = 2, stroke_style: str = "solid", link: str | None = None) -> dict:
    element = base_shape(element_id, "rectangle", x, y, width, height, factory)
    element["strokeColor"] = stroke
    element["backgroundColor"] = fill
    element["strokeWidth"] = stroke_width
    element["strokeStyle"] = stroke_style
    element["link"] = link
    return element


def make_ellipse(element_id: str, x: float, y: float, width: float, height: float, *, stroke: str, fill: str, factory: IdFactory, link: str | None = None) -> dict:
    element = base_shape(element_id, "ellipse", x, y, width, height, factory)
    element["strokeColor"] = stroke
    element["backgroundColor"] = fill
    element["strokeWidth"] = 2
    element["link"] = link
    element["roundness"] = {"type": 2}
    return element


def make_text(
    element_id: str,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
    font_size: int,
    font_family: int,
    text_align: str,
    container_id: str | None,
    factory: IdFactory,
    link: str | None = None,
) -> dict:
    element = base_shape(element_id, "text", x, y, width, height, factory)
    element["roundness"] = None
    element["strokeColor"] = color
    element["backgroundColor"] = "transparent"
    element["strokeWidth"] = 1
    element["roughness"] = 1
    element["boundElements"] = None
    element["text"] = text
    element["fontSize"] = font_size
    element["fontFamily"] = font_family
    element["textAlign"] = text_align
    element["verticalAlign"] = "middle"
    element["containerId"] = container_id
    element["originalText"] = text
    element["autoResize"] = False
    element["lineHeight"] = 1.25
    element["baseline"] = max(12, height - 6)
    element["link"] = link
    return element


def make_arrow(
    element_id: str,
    x: float,
    y: float,
    points: list[list[float]],
    *,
    stroke: str,
    stroke_width: float,
    factory: IdFactory,
    stroke_style: str = "solid",
    start_binding: dict | None = None,
    end_binding: dict | None = None,
) -> dict:
    width = max(abs(point[0]) for point in points)
    height = max(abs(point[1]) for point in points)
    element = base_shape(element_id, "arrow", x, y, width, height, factory)
    element["strokeColor"] = stroke
    element["backgroundColor"] = "transparent"
    element["strokeWidth"] = stroke_width
    element["strokeStyle"] = stroke_style
    element["roughness"] = 0
    element["roundness"] = None
    element["points"] = points
    element["lastCommittedPoint"] = points[-1]
    element["startBinding"] = start_binding
    element["endBinding"] = end_binding
    element["startArrowhead"] = None
    element["endArrowhead"] = "arrow"
    element["elbowed"] = True
    element["boundElements"] = None
    return element


def edge_point(step: StepSpec | ArtifactSpec, edge: str, focus: float = 0.5) -> tuple[float, float]:
    if edge == "top":
        return step.x + step.width * focus, step.y
    if edge == "bottom":
        return step.x + step.width * focus, step.y + step.height
    if edge == "left":
        return step.x, step.y + step.height * focus
    if edge == "right":
        return step.x + step.width, step.y + step.height * focus
    raise ValueError(f"Unsupported edge: {edge}")


def add_binding(shapes: dict[str, dict], element_id: str, arrow_id: str) -> None:
    if element_id in shapes and isinstance(shapes[element_id].get("boundElements"), list):
        shapes[element_id]["boundElements"].append({"type": "arrow", "id": arrow_id})


def connector_points(source: StepSpec, target: StepSpec, kind: str) -> tuple[tuple[float, float], list[list[float]], dict, dict]:
    if kind == "vertical":
        focus = 0.5 if source.width > 260 or target.width > 260 else 0.5
        start = edge_point(source, "bottom", focus)
        end = edge_point(target, "top", focus)
        points = [[0, 0], [0, end[1] - start[1]]]
        return (
            start,
            points,
            {"elementId": source.step_id, "focus": 0, "gap": 1, "fixedPoint": [focus, 1]},
            {"elementId": target.step_id, "focus": 0, "gap": 1, "fixedPoint": [focus, 0]},
        )

    if kind == "handoff":
        start = edge_point(source, "right", 0.5)
        end = edge_point(target, "left", 0.5)
        source_phase = PHASE_BY_KEY[source.phase_key]
        target_phase = PHASE_BY_KEY[target.phase_key]
        gutter_x = (source_phase.x + source_phase.width + target_phase.x) / 2
        elbow_dx = gutter_x - start[0]
        points = [
            [0, 0],
            [elbow_dx, 0],
            [elbow_dx, end[1] - start[1]],
            [end[0] - start[0], end[1] - start[1]],
        ]
        return (
            start,
            points,
            {"elementId": source.step_id, "focus": 0, "gap": 1, "fixedPoint": [1, 0.5]},
            {"elementId": target.step_id, "focus": 0, "gap": 1, "fixedPoint": [0, 0.5]},
        )

    if kind in {"branch_left", "branch_right"}:
        start_focus = 0.36 if kind == "branch_left" else 0.64
        start = edge_point(source, "bottom", start_focus)
        end = edge_point(target, "top", 0.5)
        spine_y = start[1] + 40
        points = [
            [0, 0],
            [0, spine_y - start[1]],
            [end[0] - start[0], spine_y - start[1]],
            [end[0] - start[0], end[1] - start[1]],
        ]
        return (
            start,
            points,
            {"elementId": source.step_id, "focus": 0, "gap": 1, "fixedPoint": [start_focus, 1]},
            {"elementId": target.step_id, "focus": 0, "gap": 1, "fixedPoint": [0.5, 0]},
        )

    if kind in {"merge_left", "merge_right"}:
        end_focus = 0.32 if kind == "merge_left" else 0.68
        start = edge_point(source, "bottom", 0.5)
        end = edge_point(target, "top", end_focus)
        spine_y = end[1] - 26
        points = [
            [0, 0],
            [0, spine_y - start[1]],
            [end[0] - start[0], spine_y - start[1]],
            [end[0] - start[0], end[1] - start[1]],
        ]
        return (
            start,
            points,
            {"elementId": source.step_id, "focus": 0, "gap": 1, "fixedPoint": [0.5, 1]},
            {"elementId": target.step_id, "focus": 0, "gap": 1, "fixedPoint": [end_focus, 0]},
        )

    if kind == "footer_right":
        # Route the wrap-up connector from the bottom of the scenarios card so
        # it enters the final card cleanly instead of crossing through it.
        source_focus = min(max((target.x + target.width * 0.5 - source.x) / source.width, 0.2), 0.8)
        start = edge_point(source, "bottom", source_focus)
        end = edge_point(target, "top", 0.5)
        points = [[0, 0], [end[0] - start[0], end[1] - start[1]]]
        return (
            start,
            points,
            {"elementId": source.step_id, "focus": 0, "gap": 1, "fixedPoint": [source_focus, 1]},
            {"elementId": target.step_id, "focus": 0, "gap": 1, "fixedPoint": [0.5, 0]},
        )

    raise ValueError(f"Unsupported connector kind: {kind}")


def artifact_points(step: StepSpec, artifact: ArtifactSpec) -> tuple[tuple[float, float], list[list[float]], dict, dict]:
    start = edge_point(step, "bottom", 0.5)
    end = edge_point(artifact, "top", 0.5)
    points = [[0, 0], [0, end[1] - start[1]]]
    return (
        start,
        points,
        {"elementId": step.step_id, "focus": 0, "gap": 1, "fixedPoint": [0.5, 1]},
        {"elementId": artifact.artifact_id, "focus": 0, "gap": 1, "fixedPoint": [0.5, 0]},
    )


def build_diagram_elements() -> list[dict]:
    factory = IdFactory()
    elements: list[dict] = []
    shape_index: dict[str, dict] = {}

    title = make_text(
        "title",
        "Lab 2 Workflow: Markdown Corpus Indexing + Supervisor Agents",
        52,
        HEADER_Y,
        1120,
        40,
        color=TEXT_DARK,
        font_size=30,
        font_family=8,
        text_align="left",
        container_id=None,
        factory=factory,
    )
    subtitle = make_text(
        "subtitle",
        "Notebook-first LangGraph view of the BIT flow: instructor-only page-by-page OCR builds the markdown corpus ahead of class, the notebook fans out over local professor markdown to build a persistent Chroma collection, and the supervisor routes questions to retrieval-first workers plus one add-by-URL ingestion worker.",
        52,
        96,
        1320,
        26,
        color=TEXT_MUTED,
        font_size=16,
        font_family=5,
        text_align="left",
        container_id=None,
        factory=factory,
    )
    elements.extend([title, subtitle])

    for phase in PHASES:
        phase_rect = make_rect(
            f"phase-{phase.key}",
            phase.x,
            phase.y,
            phase.width,
            phase.height,
            stroke=phase.stroke,
            fill=phase.fill,
            factory=factory,
            stroke_width=2.2,
        )
        phase_text = make_text(
            f"phase-{phase.key}-text",
            phase.title,
            phase.x + 18,
            phase.y + 18,
            phase.width - 36,
            28,
            color=phase.title_color,
            font_size=20,
            font_family=6,
            text_align="left",
            container_id=f"phase-{phase.key}",
            factory=factory,
        )
        phase_rect["boundElements"] = [{"type": "text", "id": phase_text["id"]}]
        shape_index[phase_rect["id"]] = phase_rect
        elements.extend([phase_rect, phase_text])

    for step in STEPS:
        phase = PHASE_BY_KEY[step.phase_key]
        compact_badge_layout = step.number in COMPACT_BADGE_STEPS
        title_limit = 17 if compact_badge_layout else 18 if step.width <= 220 else 22 if step.width <= 280 else 28
        subtitle_limit = 18 if compact_badge_layout else 20 if step.width <= 220 else 24 if step.width <= 280 else 34
        text_font_size = 14 if step.width <= 220 else 15
        label_text = f"{wrap_label(step.title, title_limit)}\n{wrap_label(step.subtitle, subtitle_limit)}"
        step_rect = make_rect(
            step.step_id,
            step.x,
            step.y,
            step.width,
            step.height,
            stroke=phase.stroke,
            fill=CARD_FILL,
            factory=factory,
            stroke_width=2.2,
            link=step.doc_url,
        )
        if compact_badge_layout:
            badge_x = step.x + step.width - 38
            badge_y = step.y - 12
            text_x = step.x + 12
            text_y = step.y + 12
            text_width = step.width - 24
            text_height = step.height - 20
            text_align = "center"
        else:
            badge_x = step.x + 16
            badge_y = step.y + 18
            text_x = step.x + 52
            text_y = step.y + 12
            text_width = step.width - 70
            text_height = step.height - 20
            text_align = "left"
        badge = make_ellipse(
            f"{step.step_id}-badge",
            badge_x,
            badge_y,
            26,
            26,
            stroke=phase.stroke,
            fill=phase.stroke,
            factory=factory,
            link=step.doc_url,
        )
        badge_text = make_text(
            f"{step.step_id}-badge-text",
            f"{step.number}",
            badge_x,
            badge_y,
            26,
            26,
            color="#ffffff",
            font_size=13,
            font_family=6,
            text_align="center",
            container_id=f"{step.step_id}-badge",
            factory=factory,
            link=step.doc_url,
        )
        step_text = make_text(
            f"{step.step_id}-text",
            label_text,
            text_x,
            text_y,
            text_width,
            text_height,
            color=TEXT_DARK,
            font_size=text_font_size,
            font_family=6,
            text_align=text_align,
            container_id=step.step_id,
            factory=factory,
            link=step.doc_url,
        )
        step_rect["boundElements"] = [{"type": "text", "id": step_text["id"]}]
        badge["boundElements"] = [{"type": "text", "id": badge_text["id"]}]
        shape_index[step_rect["id"]] = step_rect
        shape_index[badge["id"]] = badge
        elements.extend([step_rect, badge, badge_text, step_text])

        artifact = ARTIFACT_BY_STEP.get(step.number)
        if artifact:
            artifact_text_value = wrap_label(artifact.text, 22 if artifact.width <= 220 else 30)
            artifact_rect = make_rect(
                artifact.artifact_id,
                artifact.x,
                artifact.y,
                artifact.width,
                artifact.height,
                stroke=phase.stroke,
                fill=ARTIFACT_FILL,
                factory=factory,
                stroke_width=1.8,
                stroke_style="dashed",
            )
            artifact_text = make_text(
                f"{artifact.artifact_id}-text",
                artifact_text_value,
                artifact.x + 14,
                artifact.y + 10,
                artifact.width - 28,
                artifact.height - 18,
                color=ARTIFACT_TEXT,
                font_size=14,
                font_family=5,
                text_align="left",
                container_id=artifact.artifact_id,
                factory=factory,
            )
            artifact_rect["boundElements"] = [{"type": "text", "id": artifact_text["id"]}]
            shape_index[artifact_rect["id"]] = artifact_rect
            elements.extend([artifact_rect, artifact_text])

            arrow_start, arrow_points, start_binding, end_binding = artifact_points(step, artifact)
            artifact_arrow = make_arrow(
                f"{artifact.artifact_id}-arrow",
                arrow_start[0],
                arrow_start[1],
                arrow_points,
                stroke=phase.stroke,
                stroke_width=1.8,
                factory=factory,
                stroke_style="dashed",
                start_binding=start_binding,
                end_binding=end_binding,
            )
            add_binding(shape_index, step.step_id, artifact_arrow["id"])
            add_binding(shape_index, artifact.artifact_id, artifact_arrow["id"])
            elements.append(artifact_arrow)

    for source_number, target_number, kind in CONNECTORS:
        source = STEP_BY_NUMBER[source_number]
        target = STEP_BY_NUMBER[target_number]
        start, points, start_binding, end_binding = connector_points(source, target, kind)
        connector = make_arrow(
            f"connector-{source_number:02d}-{target_number:02d}",
            start[0],
            start[1],
            points,
            stroke=CONNECTOR_STROKE,
            stroke_width=2.4 if kind == "handoff" else 2.0,
            factory=factory,
            start_binding=start_binding,
            end_binding=end_binding,
        )
        add_binding(shape_index, source.step_id, connector["id"])
        add_binding(shape_index, target.step_id, connector["id"])
        elements.append(connector)

    return sorted(elements, key=element_layer_key)


def build_master_excalidraw(path: Path) -> list[dict]:
    elements = build_diagram_elements()
    validate_elements(elements)
    document = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://openai.com/codex",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return elements


def validate_elements(elements: list[dict]) -> None:
    ids = [element["id"] for element in elements]
    duplicates = sorted({element_id for element_id in ids if ids.count(element_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate element IDs: {duplicates}")

    index = {element["id"]: element for element in elements}
    expected_steps = {f"step-{number:02d}" for number in range(1, 15)}
    actual_steps = {element_id for element_id in index if element_id.startswith("step-") and element_id.count("-") == 1}
    if expected_steps != actual_steps:
        missing = sorted(expected_steps - actual_steps)
        extra = sorted(actual_steps - expected_steps)
        raise ValueError(f"Step mismatch. Missing: {missing} Extra: {extra}")

    for element in elements:
        bound = element.get("boundElements")
        if isinstance(bound, list):
            for ref in bound:
                target = index.get(ref["id"])
                if target is None:
                    raise ValueError(f"{element['id']} references missing bound element {ref['id']}")
                if ref["type"] == "text" and target.get("containerId") != element["id"]:
                    raise ValueError(f"Text binding mismatch for {element['id']} -> {ref['id']}")

        if element["type"] != "arrow":
            continue

        if element.get("roughness") != 0 or element.get("roundness") is not None or not element.get("elbowed"):
            raise ValueError(f"Arrow {element['id']} does not satisfy elbow routing rules")

        if not element.get("startBinding") or not element.get("endBinding"):
            raise ValueError(f"Arrow {element['id']} is missing bindings")

        if element["id"].startswith("artifact-"):
            step_number = int(element["id"].split("-")[1])
            if step_number not in ARTIFACT_BY_STEP:
                raise ValueError(f"Artifact arrow {element['id']} is not mapped to a known artifact")

        if element["id"].startswith("connector-"):
            _, source, target = element["id"].split("-")
            source_number = int(source)
            target_number = int(target)
            if source_number not in STEP_BY_NUMBER or target_number not in STEP_BY_NUMBER:
                raise ValueError(f"Connector {element['id']} references a missing step")


def classify_element(element_id: str) -> tuple[str, int | str | None]:
    if element_id in {"title", "subtitle"}:
        return "context", None
    if element_id.startswith("phase-"):
        parts = element_id.split("-")
        return "phase", parts[1]
    if element_id.startswith("step-"):
        parts = element_id.split("-")
        return "step", int(parts[1])
    if element_id.startswith("artifact-"):
        parts = element_id.split("-")
        return "artifact", int(parts[1])
    if element_id.startswith("connector-"):
        _, source, target = element_id.split("-")
        return "connector", int(target)
    return "context", None


def element_layer_key(element: dict) -> int:
    kind, _ = classify_element(element["id"])
    if element["type"] == "rectangle" and kind == "phase":
        return 0
    if element["type"] == "arrow":
        return 1
    if element["type"] in {"rectangle", "ellipse"}:
        return 2
    if element["type"] == "text":
        return 3
    return 4


def opacity_for(element: dict, current_step: int | None) -> float:
    if current_step is None:
        return 1.0
    kind, marker = classify_element(element["id"])
    if kind in {"context", "phase"}:
        return 1.0
    if kind in {"step", "artifact"}:
        return 1.0 if int(marker) <= current_step else DIM_OPACITY
    if kind == "connector":
        return 1.0 if int(marker) <= current_step else DIM_OPACITY
    return 1.0


def current_phase_key(current_step: int | None) -> str | None:
    if current_step is None:
        return None
    return STEP_BY_NUMBER[current_step].phase_key


def current_step_number(element_id: str) -> int | None:
    kind, marker = classify_element(element_id)
    return int(marker) if kind == "step" else None


def svg_defs() -> str:
    return "\n".join(
        [
            "<defs>",
            '  <marker id="arrow-dark" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">',
            '    <path d="M 0 1 L 9 5 L 0 9 Z" fill="#64748b"/>',
            "  </marker>",
            '  <marker id="arrow-current" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">',
            '    <path d="M 0 1 L 9 5 L 0 9 Z" fill="#e8590c"/>',
            "  </marker>",
            '  <filter id="current-shadow" x="-10%" y="-10%" width="140%" height="140%">',
            '    <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#fed7aa" flood-opacity="0.95"/>',
            "  </filter>",
            "</defs>",
        ]
    )


def render_rect(element: dict, current_step: int | None) -> str:
    opacity = opacity_for(element, current_step)
    stroke = element["strokeColor"]
    fill = element["backgroundColor"]
    stroke_width = element["strokeWidth"]
    extra = ""
    kind, marker = classify_element(element["id"])
    if kind == "phase" and current_step is not None and marker == current_phase_key(current_step):
        stroke_width = max(stroke_width, 2.8)
    if kind == "step" and current_step is not None and int(marker) == current_step:
        stroke = CURRENT_STROKE
        fill = CURRENT_FILL
        stroke_width = 4
        extra = ' filter="url(#current-shadow)"'
    if kind == "artifact" and current_step is not None and int(marker) == current_step:
        stroke = CURRENT_STROKE
        stroke_width = 2.4

    dash = ' stroke-dasharray="8 6"' if element.get("strokeStyle") == "dashed" else ""
    svg = (
        f'<rect x="{element["x"]}" y="{element["y"]}" width="{element["width"]}" height="{element["height"]}" '
        f'rx="18" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}{extra}/>'
    )
    return wrap_link(element.get("link"), f'<g opacity="{opacity:.2f}">{svg}</g>')


def render_ellipse(element: dict, current_step: int | None) -> str:
    opacity = opacity_for(element, current_step)
    stroke = element["strokeColor"]
    fill = element["backgroundColor"]
    extra = ""
    kind, marker = classify_element(element["id"])
    if kind == "step" and current_step is not None and int(marker) == current_step:
        stroke = CURRENT_STROKE
        fill = CURRENT_STROKE
        extra = ' filter="url(#current-shadow)"'
    cx = element["x"] + element["width"] / 2
    cy = element["y"] + element["height"] / 2
    rx = element["width"] / 2
    ry = element["height"] / 2
    svg = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{element["strokeWidth"]}"{extra}/>'
    return wrap_link(element.get("link"), f'<g opacity="{opacity:.2f}">{svg}</g>')


def render_text(element: dict, current_step: int | None) -> str:
    opacity = opacity_for(element, current_step)
    text = element.get("text", "")
    lines = text.split("\n")
    font_size = element["fontSize"]
    line_height = font_size * 1.28
    x = element["x"]
    y = element["y"]
    width = element.get("width", 0)
    height = element.get("height", 0)
    align = element.get("textAlign", "left")
    if align == "center":
        anchor = "middle"
        tx = x + width / 2
    elif align == "right":
        anchor = "end"
        tx = x + width
    else:
        anchor = "start"
        tx = x
    content_height = font_size + line_height * (len(lines) - 1)
    first_y = y + max((height - content_height) / 2, 0)
    tspans = []
    for idx, line in enumerate(lines):
        line_y = first_y + idx * line_height
        tspans.append(f'<tspan x="{tx:.1f}" y="{line_y:.1f}" dominant-baseline="hanging">{escape(line)}</tspan>')

    color = element["strokeColor"]
    if current_step is not None:
        kind, marker = classify_element(element["id"])
        if kind == "step" and int(marker) == current_step and element["id"].endswith("badge-text"):
            color = "#ffffff"

    svg = (
        f'<text font-family="{FONT_FAMILY}" font-size="{font_size}" fill="{color}" text-anchor="{anchor}">'
        + "".join(tspans)
        + "</text>"
    )
    return wrap_link(element.get("link"), f'<g opacity="{opacity:.2f}">{svg}</g>')


def render_arrow(element: dict, current_step: int | None) -> str:
    opacity = opacity_for(element, current_step)
    points = [(element["x"] + point[0], element["y"] + point[1]) for point in element["points"]]
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    stroke = element["strokeColor"]
    stroke_width = element["strokeWidth"]
    marker = "arrow-dark"
    kind, marker_value = classify_element(element["id"])
    if current_step is not None and kind in {"artifact", "connector"} and int(marker_value) == current_step:
        stroke = CURRENT_STROKE
        stroke_width = max(stroke_width, 2.6)
        marker = "arrow-current"
    dash = ' stroke-dasharray="8 6"' if element.get("strokeStyle") == "dashed" else ""
    svg = (
        f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{marker})"{dash}/>'
    )
    return f'<g opacity="{opacity:.2f}">{svg}</g>'


def wrap_link(link: str | None, svg: str) -> str:
    if not link:
        return svg
    return f'<a href="{escape(link)}" target="_blank">{svg}</a>'


def render_svg(elements: list[dict], *, current_step: int | None) -> str:
    title = "Lab 2 Workflow: Markdown Corpus Indexing + Supervisor Agents"
    subtitle = "Master workflow" if current_step is None else f"Step {current_step:02d} focus"
    header_note = (
        "One notebook-friendly workflow view for the full lab."
        if current_step is None
        else "Past steps stay visible, the current step is highlighted, and future steps are dimmed."
    )

    body: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        svg_defs(),
        f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{CANVAS_FILL}" stroke="{CANVAS_STROKE}" stroke-width="1"/>',
        f'<text x="52" y="56" font-family="{FONT_FAMILY}" font-size="30" font-weight="700" fill="{TEXT_DARK}">{escape(title)}</text>',
        f'<text x="52" y="86" font-family="{FONT_FAMILY}" font-size="16" font-weight="500" fill="{TEXT_MUTED}">{escape(subtitle)}</text>',
        f'<text x="52" y="112" font-family="{FONT_FAMILY}" font-size="15" font-weight="400" fill="{TEXT_SOFT}">{escape(header_note)}</text>',
    ]

    for element in elements:
        if element["id"] in {"title", "subtitle"}:
            continue
        if element["type"] == "rectangle":
            body.append(render_rect(element, current_step))
        elif element["type"] == "ellipse":
            body.append(render_ellipse(element, current_step))
        elif element["type"] == "text":
            body.append(render_text(element, current_step))
        elif element["type"] == "arrow":
            body.append(render_arrow(element, current_step))

    body.append("</svg>")
    return "\n".join(body)


def render_master_and_steps(excalidraw_path: Path, master_svg_path: Path, steps_dir: Path) -> None:
    data = json.loads(excalidraw_path.read_text(encoding="utf-8"))
    elements = [element for element in data["elements"] if not element.get("isDeleted", False)]
    validate_elements(elements)
    master_svg_path.write_text(render_svg(elements, current_step=None), encoding="utf-8")
    steps_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 15):
        svg = render_svg(elements, current_step=number)
        (steps_dir / f"step_{number:02d}.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the Lab 2 workflow diagram and notebook SVG assets.")
    parser.add_argument("--excalidraw", type=Path, default=MASTER_EXCALIDRAW, help="Path to the editable Excalidraw file.")
    parser.add_argument("--master-svg", type=Path, default=MASTER_SVG, help="Path to the master workflow SVG.")
    parser.add_argument("--steps-dir", type=Path, default=STEPS_DIR, help="Directory for the per-step SVG files.")
    args = parser.parse_args()

    build_master_excalidraw(args.excalidraw)
    render_master_and_steps(args.excalidraw, args.master_svg, args.steps_dir)
    print(f"Wrote {args.excalidraw}")
    print(f"Wrote {args.master_svg}")
    print(f"Wrote {args.steps_dir}/step_01.svg ... step_14.svg")


if __name__ == "__main__":
    main()
