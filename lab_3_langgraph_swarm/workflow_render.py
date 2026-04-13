from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent
EXCALIDRAW_PATH = ROOT / "lab_3_swarm.excalidraw"
SVG_PATH = ROOT / "lab_3_swarm.svg"

CANVAS_W = 1280
CANVAS_H = 720
FONT_FAMILY = "Inter, Arial, sans-serif"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"


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


TITLE = "Lab 3: Minimal Swarm Handoffs"
SUBTITLE = "Front desk transfers ownership, specialists keep control, and active_agent persists by thread_id."

BOXES = [
    Box(
        "user",
        "Student",
        60,
        270,
        140,
        80,
        fill="#e7f5ff",
        stroke="#1971c2",
        shape="ellipse",
    ),
    Box(
        "front-desk",
        "FrontDeskAgent",
        250,
        250,
        240,
        120,
        fill="#fff7ed",
        stroke="#e8590c",
    ),
    Box(
        "professor",
        "ProfessorLookupAgent",
        590,
        110,
        280,
        120,
        fill="#eff6ff",
        stroke="#1d4ed8",
    ),
    Box(
        "research",
        "ResearchMatchAgent",
        590,
        390,
        280,
        120,
        fill="#ecfdf5",
        stroke="#15803d",
    ),
    Box(
        "neo4j",
        "Neo4jQueryService",
        970,
        250,
        240,
        120,
        fill="#f0fdf4",
        stroke="#16a34a",
    ),
    Box(
        "memory-note",
        "InMemorySaver\nactive_agent + thread_id",
        470,
        585,
        360,
        80,
        fill="#fffbeb",
        stroke="#f59e0b",
        dashed=True,
    ),
    Box(
        "handoff-note",
        "Handoff tools update\nactive_agent",
        520,
        255,
        180,
        70,
        fill="#fff7ed",
        stroke="#fb923c",
        dashed=True,
    ),
]
BOX_BY_ID = {box.element_id: box for box in BOXES}

ARROWS = [
    Arrow(
        "user-to-front-desk",
        [BOX_BY_ID["user"].right_center, BOX_BY_ID["front-desk"].left_center],
    ),
    Arrow(
        "front-desk-to-professor",
        [
            BOX_BY_ID["front-desk"].right_center,
            (540, BOX_BY_ID["front-desk"].right_center[1]),
            (540, BOX_BY_ID["professor"].left_center[1]),
            BOX_BY_ID["professor"].left_center,
        ],
        stroke="#e8590c",
    ),
    Arrow(
        "front-desk-to-research",
        [
            BOX_BY_ID["front-desk"].right_center,
            (540, BOX_BY_ID["front-desk"].right_center[1]),
            (540, BOX_BY_ID["research"].left_center[1]),
            BOX_BY_ID["research"].left_center,
        ],
        stroke="#e8590c",
    ),
    Arrow(
        "professor-to-neo4j",
        [
            BOX_BY_ID["professor"].right_center,
            (920, BOX_BY_ID["professor"].right_center[1]),
            (920, BOX_BY_ID["neo4j"].left_center[1]),
            BOX_BY_ID["neo4j"].left_center,
        ],
        stroke="#1d4ed8",
    ),
    Arrow(
        "research-to-neo4j",
        [
            BOX_BY_ID["research"].right_center,
            (920, BOX_BY_ID["research"].right_center[1]),
            (920, BOX_BY_ID["neo4j"].left_center[1]),
            BOX_BY_ID["neo4j"].left_center,
        ],
        stroke="#15803d",
    ),
    Arrow(
        "professor-to-research",
        [
            (680, 230),
            (680, 310),
            (680, 390),
        ],
        stroke="#fb923c",
        dashed=True,
    ),
    Arrow(
        "research-to-professor",
        [
            (780, 390),
            (780, 310),
            (780, 230),
        ],
        stroke="#fb923c",
        dashed=True,
    ),
]


def text_dimensions(label: str) -> tuple[int, int]:
    lines = label.split("\n")
    width = max(int(len(line) * 8.2) for line in lines) if lines else 0
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
    shape = {
        **common,
        "type": box.shape,
    }
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
        "boundElements": [],
        "updated": timestamp,
        "link": None,
        "locked": False,
        "text": box.label,
        "fontSize": 20 if box.shape == "ellipse" else 18,
        "fontFamily": 1,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": box.element_id,
        "originalText": box.label,
        "autoResize": True,
        "lineHeight": 1.25,
    }
    return [shape, text]


def excalidraw_arrow(arrow: Arrow, timestamp: int) -> dict:
    start_x, start_y = arrow.points[0]
    relative_points = [
        [point_x - start_x, point_y - start_y]
        for point_x, point_y in arrow.points
    ]
    width = int(max(point[0] for point in relative_points) - min(point[0] for point in relative_points))
    height = int(max(point[1] for point in relative_points) - min(point[1] for point in relative_points))
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
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": len(relative_points) > 2,
    }


def build_excalidraw() -> dict:
    timestamp = int(time.time() * 1000)
    elements: list[dict] = []
    for box in BOXES:
        elements.extend(excalidraw_shape(box, timestamp))
    for arrow in ARROWS:
        elements.append(excalidraw_arrow(arrow, timestamp))
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://openai.com/codex",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#f8fafc",
        },
        "files": {},
    }


def svg_rect(box: Box) -> str:
    dash = ' stroke-dasharray="8 6"' if box.dashed else ""
    if box.shape == "ellipse":
        return (
            f'<ellipse cx="{box.center_x}" cy="{box.center_y}" '
            f'rx="{box.width / 2}" ry="{box.height / 2}" '
            f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="2.6"{dash}/>'
        )
    return (
        f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="22" '
        f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="2.6"{dash}/>'
    )


def svg_text(box: Box) -> str:
    lines = [escape(line) for line in box.label.split("\n")]
    start_y = box.center_y - (len(lines) - 1) * 12
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else 24
        tspans.append(f'<tspan x="{box.center_x}" dy="{dy}">{line}</tspan>')
    return (
        f'<text x="{box.center_x}" y="{start_y}" fill="{TEXT_DARK}" '
        f'font-family="{FONT_FAMILY}" font-size="18" text-anchor="middle" '
        'dominant-baseline="middle">'
        + "".join(tspans)
        + "</text>"
    )


def svg_polyline(arrow: Arrow) -> str:
    points = " ".join(f"{x},{y}" for x, y in arrow.points)
    dash = ' stroke-dasharray="8 6"' if arrow.dashed else ""
    return (
        f'<polyline points="{points}" fill="none" stroke="{arrow.stroke}" '
        f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" '
        f'marker-end="url(#arrowhead)"{dash}/>'
    )


def build_svg() -> str:
    shapes = "\n  ".join(svg_rect(box) for box in BOXES)
    labels = "\n  ".join(svg_text(box) for box in BOXES)
    connectors = "\n  ".join(svg_polyline(arrow) for arrow in ARROWS)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}" fill="none">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" fill="#f8fafc"/>
  <text x="60" y="66" fill="{TEXT_DARK}" font-family="{FONT_FAMILY}" font-size="34" font-weight="700">{escape(TITLE)}</text>
  <text x="60" y="102" fill="{TEXT_MUTED}" font-family="{FONT_FAMILY}" font-size="18">{escape(SUBTITLE)}</text>
  {connectors}
  {shapes}
  {labels}
</svg>
"""


def main() -> None:
    EXCALIDRAW_PATH.write_text(
        json.dumps(build_excalidraw(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {EXCALIDRAW_PATH.relative_to(ROOT.parent)}")
    print(f"Wrote {SVG_PATH.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
