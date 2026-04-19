"""
workflow_render.py - Build and render the Lab 1 workflow diagrams.

This script maintains one diagram model and writes:
  - lab_1_workflow.svg
  - workflow_steps/step_01.svg ... workflow_steps/step_13.svg

The SVG files are notebook-friendly renderings derived from the same model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent
MASTER_SVG = ROOT / "lab_1_workflow.svg"
STEPS_DIR = ROOT / "workflow_steps"

CANVAS_W = 1680
CANVAS_H = 900
HEADER_Y = 54
PHASE_Y = 170
PHASE_H = 650
STEP_H = 88
STEP_SIDE_MARGIN = 18
STEP_ARTIFACT_GAP = 18
PHASE_CONTENT_TOP_OFFSET = 56
PHASE_CONTENT_BOTTOM_OFFSET = 40
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


DOCS = {
    "uv": "https://docs.astral.sh/uv/getting-started/",
    "docker_compose": "https://docs.docker.com/compose/gettingstarted/",
    "docker_compose_up": "https://docs.docker.com/reference/cli/docker/compose/up/",
    "gradio": "https://www.gradio.app/guides/quickstart",
    "chat_openai": "https://docs.langchain.com/oss/python/integrations/chat/openai",
    "messages": "https://docs.langchain.com/oss/python/langchain/messages",
    "lcel": "https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.Runnable.html",
    "runnable_sequence": "https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableSequence.html",
    "requests": "https://requests.readthedocs.io/en/latest/user/quickstart/",
    "bs4": "https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
    "neo4j": "https://neo4j.com/docs/python-manual/current/connect/",
    "langchain_mcp": "https://docs.langchain.com/oss/python/langchain/mcp",
    "langchain_learn": "https://docs.langchain.com/oss/python/learn",
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
    wrapped_lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split()
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
    PhaseSpec("setup", "Setup", (1, 4), 48, 268, "#f8fafc", "#94a3b8", "#334155"),
    PhaseSpec("data", "Data Acquisition", (5, 6), 344, 268, "#ecfdf5", "#34d399", "#065f46"),
    PhaseSpec("langchain", "Graph Input Prep", (7, 8), 640, 268, "#eff6ff", "#60a5fa", "#1d4ed8"),
    PhaseSpec("kg", "Typed Graph Prep", (9, 10), 936, 268, "#f5f3ff", "#a78bfa", "#6d28d9"),
    PhaseSpec("agent", "Agent Delivery", (11, 13), 1232, 400, "#fff7ed", "#fb923c", "#c2410c"),
]
PHASE_BY_KEY = {phase.key: phase for phase in PHASES}

STEP_LAYOUTS = [
    (1, "setup", "Environment Check", "runtime + imports", DOCS["uv"]),
    (2, "setup", "Services + Configuration", ".env, model, Neo4j", DOCS["chat_openai"]),
    (3, "setup", "Base LangChain Model", "shared model + tools", DOCS["chat_openai"]),
    (4, "setup", "Tiny LCEL Chain", "prompt | model | parser", DOCS["lcel"]),
    (5, "data", "Crawl Listing Page", "BIT -> detail URLs", DOCS["requests"]),
    (6, "data", "Default 5-Professor Working Set", "first five links -> working set", DOCS["bs4"]),
    (7, "langchain", "Page-by-Page OCR", "image pages -> notes", DOCS["messages"]),
    (8, "langchain", "Merge OCR Into Review Markdown", "clean OCR text per professor", DOCS["runnable_sequence"]),
    (9, "kg", "Parallel Structured Extraction", "one typed JSON profile per professor", DOCS["chat_openai"]),
    (10, "kg", "Verify Typed Neo4j After Batch Load", "counts + sample facts", DOCS["neo4j"]),
    (11, "agent", "Short MCP Epilogue", "create_agent + Neo4j tools", DOCS["langchain_mcp"]),
    (12, "agent", "Appendix: Inline Gradio UI", "same agent in the notebook", DOCS["gradio"]),
    (13, "agent", "Wrap-Up", "concepts + design habits", DOCS["langchain_learn"]),
]

ARTIFACT_LAYOUTS: dict[int, tuple[str, int]] = {
    5: ("Professor detail URLs", 196),
    6: ("Default 5-professor set", 196),
    7: ("Page notes (debug)", 196),
    8: ("Reviewed OCR markdown", 196),
    9: ("structured_output/*.json", 196),
    10: ("Neo4j graph store", 196),
    11: ("Answer + tool trace", 308),
}


def phase_content_bounds(phase: PhaseSpec) -> tuple[int, int]:
    return phase.y + PHASE_CONTENT_TOP_OFFSET, phase.y + phase.height - PHASE_CONTENT_BOTTOM_OFFSET


def step_width_for_phase(phase: PhaseSpec) -> int:
    return phase.width - STEP_SIDE_MARGIN * 2


def layout_phase_steps(phase: PhaseSpec) -> tuple[list[StepSpec], list[ArtifactSpec]]:
    phase_steps = [layout for layout in STEP_LAYOUTS if layout[1] == phase.key]
    content_top, content_bottom = phase_content_bounds(phase)
    group_heights = [
        STEP_H + (STEP_ARTIFACT_GAP + 52 if number in ARTIFACT_LAYOUTS else 0)
        for number, *_ in phase_steps
    ]
    available_height = content_bottom - content_top
    gap = (available_height - sum(group_heights)) / (len(phase_steps) + 1)
    if gap <= 0:
        raise ValueError(f"Phase {phase.key} does not have enough height for its steps")

    steps: list[StepSpec] = []
    artifacts: list[ArtifactSpec] = []
    cursor_y = content_top + gap
    step_width = step_width_for_phase(phase)
    step_x = int(round(phase.x + (phase.width - step_width) / 2))

    for number, phase_key, title, subtitle, doc_url in phase_steps:
        step_y = int(round(cursor_y))
        steps.append(StepSpec(number, phase_key, title, subtitle, doc_url, step_x, step_y, step_width))

        artifact_layout = ARTIFACT_LAYOUTS.get(number)
        if artifact_layout:
            artifact_text, artifact_width = artifact_layout
            artifact_x = int(round(phase.x + (phase.width - artifact_width) / 2))
            artifact_y = step_y + STEP_H + STEP_ARTIFACT_GAP
            artifacts.append(ArtifactSpec(number, artifact_text, artifact_x, artifact_y, artifact_width))
            cursor_y += STEP_H + STEP_ARTIFACT_GAP + 52 + gap
        else:
            cursor_y += STEP_H + gap

    return steps, artifacts


STEPS: list[StepSpec] = []
ARTIFACTS: list[ArtifactSpec] = []
for phase in PHASES:
    phase_steps, phase_artifacts = layout_phase_steps(phase)
    STEPS.extend(phase_steps)
    ARTIFACTS.extend(phase_artifacts)

STEP_BY_NUMBER = {step.number: step for step in STEPS}
ARTIFACT_BY_STEP = {artifact.step_number: artifact for artifact in ARTIFACTS}

# Each connector becomes visible once the destination step is reached.
CONNECTORS: list[tuple[int, int, str]] = [
    (1, 2, "vertical"),
    (2, 3, "vertical"),
    (3, 4, "vertical"),
    (4, 5, "handoff"),
    (5, 6, "vertical"),
    (6, 7, "handoff"),
    (7, 8, "vertical"),
    (8, 9, "handoff"),
    (9, 10, "vertical"),
    (10, 11, "handoff"),
    (11, 12, "vertical"),
    (12, 13, "vertical"),
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
        # Route non-setup vertical connectors near the right edge so they clear the dashed artifact cards.
        focus = 0.5 if source.phase_key == "setup" else 0.96
        start = edge_point(source, "bottom", focus)
        end = edge_point(target, "top", focus)
        points = [[0, 0], [0, end[1] - start[1]]]
        return (
            start,
            points,
            {"elementId": source.step_id, "focus": 0, "gap": 1, "fixedPoint": [focus, 1]},
            {"elementId": target.step_id, "focus": 0, "gap": 1, "fixedPoint": [focus, 0]},
        )

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
        "Lab 1 Workflow: BIT Professor Pipeline",
        52,
        HEADER_Y,
        840,
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
        "Notebook-first view of the BIT pipeline: crawl the site, run page-by-page OCR, merge the slices into a deterministic markdown transcript, build the graph, then finish with a short MCP epilogue.",
        52,
        96,
        1120,
        24,
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
        title_limit = 20 if step.width <= 240 else 24
        subtitle_limit = 22 if step.width <= 240 else 28
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
        badge = make_ellipse(
            f"{step.step_id}-badge",
            step.x + 16,
            step.y + 18,
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
            step.x + 16,
            step.y + 18,
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
            step.x + 52,
            step.y + 12,
            step.width - 70,
            60,
            color=TEXT_DARK,
            font_size=15,
            font_family=6,
            text_align="left",
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

    return elements


def build_master_elements() -> list[dict]:
    elements = build_diagram_elements()
    validate_elements(elements)
    return elements


def validate_elements(elements: list[dict]) -> None:
    ids = [element["id"] for element in elements]
    duplicates = sorted({element_id for element_id in ids if ids.count(element_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate element IDs: {duplicates}")

    index = {element["id"]: element for element in elements}
    expected_steps = {f"step-{number:02d}" for number in range(1, 14)}
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
    title = "Lab 1 Workflow: BIT Professor Pipeline"
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


def render_master_and_steps(elements: list[dict], master_svg_path: Path, steps_dir: Path) -> None:
    master_svg_path.write_text(render_svg(elements, current_step=None), encoding="utf-8")
    steps_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 14):
        svg = render_svg(elements, current_step=number)
        (steps_dir / f"step_{number:02d}.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the Lab 1 workflow SVG assets.")
    parser.add_argument("--master-svg", type=Path, default=MASTER_SVG, help="Path to the master workflow SVG.")
    parser.add_argument("--steps-dir", type=Path, default=STEPS_DIR, help="Directory for the per-step SVG files.")
    args = parser.parse_args()

    render_master_and_steps(build_master_elements(), args.master_svg, args.steps_dir)
    print(f"Wrote {args.master_svg}")
    print(f"Wrote {args.steps_dir}/step_01.svg ... step_13.svg")


if __name__ == "__main__":
    main()
