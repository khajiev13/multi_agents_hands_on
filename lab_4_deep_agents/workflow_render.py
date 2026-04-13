"""
workflow_render.py - Build and render the Lab 4 workflow diagrams.

This script maintains one diagram model and writes:
  - lab_4_workflow.excalidraw
  - lab_4_workflow.svg
  - workflow_steps/step_01.svg ... workflow_steps/step_06.svg
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent
MASTER_EXCALIDRAW = ROOT / "lab_4_workflow.excalidraw"
MASTER_SVG = ROOT / "lab_4_workflow.svg"
STEPS_DIR = ROOT / "workflow_steps"

CANVAS_W = 1480
CANVAS_H = 860
FONT_FAMILY = "Inter, Arial, sans-serif"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"
DIM_OPACITY = 0.20


@dataclass(frozen=True)
class Box:
    element_id: str
    label: str
    x: int
    y: int
    width: int
    height: int
    fill: str
    stroke: str
    shape: str = "rectangle"
    dashed: bool = False

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


TITLE = "Lab 4: Pure Deep Agent Workspace"
SUBTITLE = "One deep agent manages files, delegates one add-by-URL specialist, and uses prompt engineering plus a skill."

BOXES = [
    Box("user", "Student", 70, 200, 150, 86, fill="#e7f5ff", stroke="#1971c2", shape="ellipse"),
    Box(
        "main-agent",
        "ProfessorWorkspaceAgent\ncreate_deep_agent(...)",
        300,
        150,
        330,
        128,
        fill="#fff7ed",
        stroke="#e8590c",
    ),
    Box(
        "subagent",
        "AddProfessorSubagent\n(task tool)",
        780,
        105,
        270,
        118,
        fill="#eff6ff",
        stroke="#1d4ed8",
    ),
    Box(
        "skill",
        "SKILL.md\nadd-professor-from-url",
        1125,
        88,
        250,
        82,
        fill="#f8fafc",
        stroke="#64748b",
        dashed=True,
    ),
    Box(
        "ocr-tools",
        "Small helper tools\nextract_image_urls\nocr_images_to_professor_markdown\nrebuild_professors_index",
        1090,
        220,
        305,
        150,
        fill="#f5f3ff",
        stroke="#7c3aed",
    ),
    Box(
        "workspace",
        "FilesystemBackend runtime root",
        305,
        415,
        760,
        265,
        fill="#f8fafc",
        stroke="#94a3b8",
        dashed=True,
    ),
    Box(
        "index",
        "/professors.md\ncompact index",
        380,
        485,
        240,
        96,
        fill="#ecfdf5",
        stroke="#16a34a",
    ),
    Box(
        "dossiers",
        "/professors/*.md\nfull dossiers",
        720,
        485,
        250,
        96,
        fill="#eef2ff",
        stroke="#4f46e5",
    ),
    Box(
        "appendix",
        "Appendix only\nSandbox backend -> execute",
        1110,
        540,
        290,
        92,
        fill="#fffbeb",
        stroke="#f59e0b",
        dashed=True,
    ),
]
BOX_BY_ID = {box.element_id: box for box in BOXES}

ARROWS = [
    Arrow("user-to-main", [BOX_BY_ID["user"].right_center, BOX_BY_ID["main-agent"].left_center], stroke="#1971c2"),
    Arrow(
        "main-to-index",
        [
            BOX_BY_ID["main-agent"].bottom_center,
            (BOX_BY_ID["main-agent"].bottom_center[0], 360),
            (BOX_BY_ID["index"].top_center[0], 360),
            BOX_BY_ID["index"].top_center,
        ],
        stroke="#e8590c",
    ),
    Arrow(
        "main-to-dossiers",
        [
            (560, 278),
            (560, 390),
            (845, 390),
            BOX_BY_ID["dossiers"].top_center,
        ],
        stroke="#e8590c",
        dashed=True,
    ),
    Arrow(
        "main-to-subagent",
        [BOX_BY_ID["main-agent"].right_center, BOX_BY_ID["subagent"].left_center],
        stroke="#fb923c",
        dashed=True,
    ),
    Arrow(
        "subagent-to-skill",
        [BOX_BY_ID["subagent"].right_center, BOX_BY_ID["skill"].left_center],
        stroke="#64748b",
        dashed=True,
    ),
    Arrow(
        "subagent-to-ocr",
        [
            BOX_BY_ID["subagent"].bottom_center,
            (BOX_BY_ID["subagent"].bottom_center[0], 245),
            (1135, 245),
            BOX_BY_ID["ocr-tools"].left_center,
        ],
        stroke="#1d4ed8",
    ),
    Arrow(
        "ocr-to-dossiers",
        [
            BOX_BY_ID["ocr-tools"].left_center,
            (1020, 295),
            (1020, 533),
            BOX_BY_ID["dossiers"].right_center,
        ],
        stroke="#7c3aed",
    ),
    Arrow(
        "dossiers-to-index",
        [BOX_BY_ID["dossiers"].left_center, BOX_BY_ID["index"].right_center],
        stroke="#16a34a",
        dashed=True,
    ),
]
ARROW_BY_ID = {arrow.element_id: arrow for arrow in ARROWS}

STEPS = [
    StepSpec(
        1,
        "Why Deep Agents",
        "One agent owns the open-ended task and works over a file workspace.",
        {"user", "main-agent", "workspace", "index", "dossiers", "user-to-main", "main-to-index"},
    ),
    StepSpec(
        2,
        "Prompt Layers",
        "System prompts stay small; the longer add workflow lives in a skill.",
        {"main-agent", "subagent", "skill", "main-to-subagent", "subagent-to-skill"},
    ),
    StepSpec(
        3,
        "Workspace First",
        "Broad questions read /professors.md first and open dossiers only when needed.",
        {"workspace", "index", "dossiers", "main-agent", "main-to-index", "main-to-dossiers"},
    ),
    StepSpec(
        4,
        "Small Tool Layer",
        "The subagent uses OCR helpers and the main agent uses a deterministic index rebuild helper.",
        {"subagent", "ocr-tools", "dossiers", "main-to-subagent", "subagent-to-ocr", "ocr-to-dossiers"},
    ),
    StepSpec(
        5,
        "General Questions",
        "The main agent answers from local files without OCR or custom graph orchestration.",
        {"user", "main-agent", "workspace", "index", "dossiers", "user-to-main", "main-to-index", "main-to-dossiers"},
    ),
    StepSpec(
        6,
        "Add and Rebuild",
        "Add by URL delegates to the specialist, then the main agent rebuilds the compact index.",
        {
            "user",
            "main-agent",
            "subagent",
            "skill",
            "ocr-tools",
            "workspace",
            "index",
            "dossiers",
            "user-to-main",
            "main-to-subagent",
            "subagent-to-skill",
            "subagent-to-ocr",
            "ocr-to-dossiers",
            "dossiers-to-index",
        },
    ),
]


def text_dimensions(label: str) -> tuple[int, int]:
    lines = label.split("\n")
    width = max(int(len(line) * 8.1) for line in lines) if lines else 0
    height = 20 * max(len(lines), 1)
    return width, height


def excalidraw_shape(box: Box, timestamp: int) -> list[dict]:
    width, height = text_dimensions(box.label)
    text_x = box.center_x - width / 2
    text_y = box.center_y - height / 2
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
        "strokeWidth": 2.6 if not box.dashed else 2.2,
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
        "boundElements": [{"type": "text", "id": box.text_id}],
        "updated": timestamp,
        "link": None,
        "locked": False,
    }
    shape = {**common, "type": box.shape}
    text = {
        "id": box.text_id,
        "type": "text",
        "x": text_x,
        "y": text_y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": TEXT_DARK,
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
        "boundElements": None,
        "updated": timestamp,
        "link": None,
        "locked": False,
        "text": box.label,
        "fontSize": 22 if box.shape == "ellipse" else 20,
        "fontFamily": 1,
        "textAlign": "center",
        "verticalAlign": "middle",
        "baseline": height - 6,
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
        "boundElements": None,
        "updated": timestamp,
        "link": None,
        "locked": False,
        "points": relative_points,
        "lastCommittedPoint": relative_points[-1],
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": True,
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
    width, height = text_dimensions(box.label)
    text_x = box.center_x
    text_y = box.center_y - (height / 2) + 20
    opacity = DIM_OPACITY if dimmed else 1.0
    shape_tag = (
        f'<ellipse cx="{box.center_x}" cy="{box.center_y}" rx="{box.width / 2}" ry="{box.height / 2}" '
        if box.shape == "ellipse"
        else f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="24" '
    )
    dash = ' stroke-dasharray="8 8"' if box.dashed else ""
    lines = [
        shape_tag
        + f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="2.6"{dash} opacity="{opacity}"/>'
    ]
    for index, line in enumerate(box.label.split("\n")):
        dy = (index - (len(box.label.split("\n")) - 1) / 2) * 22
        lines.append(
            f'<text x="{text_x}" y="{box.center_y + dy + 7}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="20" fill="{TEXT_DARK}" opacity="{opacity}">{escape(line)}</text>'
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
            f'<path d="{" ".join(path_parts)}" fill="none" stroke="{arrow.stroke}" stroke-width="2.4"{dash} opacity="{opacity}"/>',
            f'<polygon points="{" ".join(f"{x},{y}" for x, y in arrow_head)}" fill="{arrow.stroke}" opacity="{opacity}"/>',
        ]
    )


def render_svg(*, highlight_ids: set[str] | None, title: str, subtitle: str, output_path: Path) -> None:
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        '<rect x="22" y="22" width="1436" height="816" rx="28" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>',
        f'<text x="72" y="86" font-family="{FONT_FAMILY}" font-size="34" font-weight="700" fill="{TEXT_DARK}">{escape(title)}</text>',
        f'<text x="72" y="122" font-family="{FONT_FAMILY}" font-size="19" fill="{TEXT_MUTED}">{escape(subtitle)}</text>',
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
    parser = argparse.ArgumentParser(description="Render Lab 4 workflow diagrams.")
    parser.parse_args()

    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    write_excalidraw(MASTER_EXCALIDRAW)
    render_svg(highlight_ids=None, title=TITLE, subtitle=SUBTITLE, output_path=MASTER_SVG)

    for step in STEPS:
        render_svg(
            highlight_ids=step.highlight_ids,
            title=f"Step {step.number:02d}: {step.title}",
            subtitle=step.subtitle,
            output_path=STEPS_DIR / f"step_{step.number:02d}.svg",
        )

    print(f"Wrote {MASTER_EXCALIDRAW}")
    print(f"Wrote {MASTER_SVG}")
    print(f"Wrote {STEPS_DIR}/step_01.svg ... step_06.svg")


if __name__ == "__main__":
    main()
