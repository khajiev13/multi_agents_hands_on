"""Render Lab 4 teaching diagrams as SVG files.

Four focused diagrams are emitted under this directory:

  01_deep_agent_anatomy             - inside one deep agent (model <-> tools loop, what is bundled)
  02_professor_workspace_agent      - ProfessorWorkspaceAgent: prompt, file tools, custom tool, subagent
  03_add_professor_subagent         - AddProfessorSubagent + skill + add_professor_from_url.py pipeline
  04_lab4_composition               - student -> deep agent -> backend, subagent, skills, checkpointer

Run:
    uv run python lab_4_deep_agents/teaching_diagrams/render.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent

FONT_FAMILY = "Inter, Arial, sans-serif"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"


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
    font_size: int = 16
    stroke_width: float = 3.0
    text_color: str = TEXT_DARK

    @property
    def text_id(self) -> str:
        return f"{self.element_id}-text"

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def left(self) -> tuple[float, float]:
        return (self.x, self.cy)

    @property
    def right(self) -> tuple[float, float]:
        return (self.x + self.width, self.cy)

    @property
    def top(self) -> tuple[float, float]:
        return (self.cx, self.y)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.cx, self.y + self.height)


@dataclass(frozen=True)
class Arrow:
    element_id: str
    points: list[tuple[float, float]]
    stroke: str = "#64748b"
    dashed: bool = False
    label: str | None = None
    label_offset: tuple[int, int] = (0, -10)


@dataclass
class Diagram:
    name: str
    title: str
    subtitle: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)


def text_dims(label: str, font_size: int) -> tuple[int, int]:
    lines = label.split("\n")
    line_gap = font_size + 6
    width = max(int(len(line) * font_size * 0.56) for line in lines) if lines else 0
    height = line_gap * max(len(lines), 1)
    return width, height


def excalidraw_shape(box: Box, ts: int) -> list[dict]:
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
        "versionNonce": abs(hash(f"{box.element_id}-n")) % 100000,
        "isDeleted": False,
        "updated": ts,
        "link": None,
        "locked": False,
    }
    shape = {**common, "type": box.shape}
    if box.label is None:
        shape["boundElements"] = []
        return [shape]
    width, height = text_dims(box.label, box.font_size)
    text_x = box.cx - width / 2
    text_y = box.cy - height / 2
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
        "versionNonce": abs(hash(f"{box.text_id}-n")) % 100000,
        "isDeleted": False,
        "boundElements": [],
        "updated": ts,
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


def excalidraw_arrow(arrow: Arrow, ts: int) -> dict:
    sx, sy = arrow.points[0]
    rel = [[round(px - sx, 2), round(py - sy, 2)] for px, py in arrow.points]
    xs = [p[0] for p in rel]
    ys = [p[1] for p in rel]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return {
        "id": arrow.element_id,
        "type": "arrow",
        "x": sx,
        "y": sy,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": arrow.stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2.0,
        "strokeStyle": "dashed" if arrow.dashed else "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": abs(hash(arrow.element_id)) % 100000,
        "version": 1,
        "versionNonce": abs(hash(f"{arrow.element_id}-n")) % 100000,
        "isDeleted": False,
        "boundElements": [],
        "updated": ts,
        "link": None,
        "locked": False,
        "points": rel,
        "lastCommittedPoint": rel[-1],
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": len(rel) > 2,
    }


def write_excalidraw(diagram: Diagram, path: Path) -> None:
    ts = int(time.time() * 1000)
    elements: list[dict] = []
    for box in diagram.boxes:
        elements.extend(excalidraw_shape(box, ts))
    for arrow in diagram.arrows:
        elements.append(excalidraw_arrow(arrow, ts))
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


def svg_box(box: Box) -> str:
    if box.shape == "ellipse":
        shape = (
            f'<ellipse cx="{box.cx}" cy="{box.cy}" rx="{box.width / 2}" ry="{box.height / 2}" '
        )
    else:
        shape = f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="18" '
    dash = ' stroke-dasharray="8 6"' if box.dashed else ""
    out = [
        shape
        + f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="{box.stroke_width}"{dash}/>'
    ]
    if box.label is None:
        return "\n".join(out)
    lines = box.label.split("\n")
    gap = box.font_size + 6
    for i, line in enumerate(lines):
        dy = (i - (len(lines) - 1) / 2) * gap
        out.append(
            f'<text x="{box.cx}" y="{box.cy + dy + box.font_size * 0.35}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="{box.font_size}" font-weight="600" '
            f'fill="{box.text_color}">{escape(line)}</text>'
        )
    return "\n".join(out)


def svg_arrow(arrow: Arrow) -> str:
    pts = arrow.points
    path = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    dash = ' stroke-dasharray="8 6"' if arrow.dashed else ""
    end_x, end_y = pts[-1]
    prev_x, prev_y = pts[-2]
    dx = end_x - prev_x
    dy = end_y - prev_y
    if abs(dx) > abs(dy):
        head = [
            (end_x, end_y),
            (end_x - 11 if dx > 0 else end_x + 11, end_y - 6),
            (end_x - 11 if dx > 0 else end_x + 11, end_y + 6),
        ]
    else:
        head = [
            (end_x, end_y),
            (end_x - 6, end_y - 11 if dy > 0 else end_y + 11),
            (end_x + 6, end_y - 11 if dy > 0 else end_y + 11),
        ]
    out = [
        f'<path d="{path}" fill="none" stroke="{arrow.stroke}" stroke-width="2.8"{dash}/>',
        f'<polygon points="{" ".join(f"{x},{y}" for x, y in head)}" fill="{arrow.stroke}"/>',
    ]
    if arrow.label:
        mid_idx = len(pts) // 2
        if len(pts) % 2 == 0:
            ax, ay = pts[mid_idx - 1]
            bx, by = pts[mid_idx]
            mx, my = (ax + bx) / 2, (ay + by) / 2
        else:
            mx, my = pts[mid_idx]
        ox, oy = arrow.label_offset
        text_w = len(arrow.label) * 8 + 16
        out.append(
            f'<rect x="{mx + ox - text_w / 2}" y="{my + oy - 13}" width="{text_w}" height="24" '
            f'rx="6" fill="#ffffff" stroke="{arrow.stroke}" stroke-width="1.4"/>'
        )
        out.append(
            f'<text x="{mx + ox}" y="{my + oy + 5}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="14" font-weight="700" '
            f'fill="{TEXT_DARK}">{escape(arrow.label)}</text>'
        )
    return "\n".join(out)


def render_svg(diagram: Diagram, path: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{diagram.width}" height="{diagram.height}" '
        f'viewBox="0 0 {diagram.width} {diagram.height}">',
        f'<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        f'<rect x="20" y="20" width="{diagram.width - 40}" height="{diagram.height - 40}" '
        f'rx="24" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>',
        f'<text x="56" y="74" font-family="{FONT_FAMILY}" font-size="34" font-weight="800" '
        f'fill="{TEXT_DARK}">{escape(diagram.title)}</text>',
        f'<text x="56" y="110" font-family="{FONT_FAMILY}" font-size="18" font-weight="600" '
        f'fill="{TEXT_MUTED}">{escape(diagram.subtitle)}</text>',
    ]
    for arrow in diagram.arrows:
        parts.append(svg_arrow(arrow))
    for box in diagram.boxes:
        parts.append(svg_box(box))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def emit(diagram: Diagram) -> None:
    svg = ROOT / f"{diagram.name}.svg"
    render_svg(diagram, svg)
    print(f"  wrote {svg.name}")


# ---------------------------------------------------------------------------
# Diagram builders
# ---------------------------------------------------------------------------


def build_deep_agent_anatomy() -> Diagram:
    d = Diagram(
        name="01_deep_agent_anatomy",
        title="Anatomy of a Deep Agent",
        subtitle="create_deep_agent wires one model into a tool loop. The 'tools' node bundles file tools, execute, custom tools, subagent dispatchers, and a skills resolver.",
        width=1720,
        height=1040,
    )

    # State (top-left)
    d.boxes.append(Box(
        "state",
        "DeepAgentState\n\n"
        "• messages: list[BaseMessage]\n"
        "• files: dict[str, str]\n"
        "    (virtual FS — bypassed when\n"
        "     LocalShellBackend writes\n"
        "     to the real workspace)\n"
        "• todos: list[Todo]",
        60, 150, 380, 230,
        fill="#eff6ff", stroke="#2563eb", font_size=16,
    ))

    # create_deep_agent signature (top-right)
    d.boxes.append(Box(
        "build-fn",
        "create_deep_agent(\n"
        "  model              = build_model(settings),\n"
        "  tools              = [rebuild_professors_index_tool],\n"
        "  system_prompt      = MAIN_SYSTEM_PROMPT,\n"
        "  backend            = LocalShellBackend(root_dir=...),\n"
        "  subagents          = [add_professor_subagent],\n"
        "  checkpointer       = InMemorySaver(),\n"
        "  name               = 'ProfessorWorkspaceAgent',\n"
        ")\n"
        "→ compiled LangGraph: START → model → tools → … → END",
        1100, 150, 580, 230,
        fill="#f8fafc", stroke="#475569", font_size=16, dashed=True,
    ))

    # Center loop
    start = Box(
        "n-start", "START",
        780, 430, 140, 56,
        fill="#e0f2fe", stroke="#0284c7", shape="ellipse", font_size=18,
    )
    model = Box(
        "n-model", "model\n(LLM call)",
        730, 540, 240, 90,
        fill="#eef2ff", stroke="#4f46e5", font_size=18,
    )
    tools = Box(
        "n-tools", "tools\n(bundled ToolNode)",
        1080, 540, 280, 90,
        fill="#ede9fe", stroke="#7c3aed", font_size=18,
    )
    end = Box(
        "n-end", "END",
        780, 740, 140, 56,
        fill="#fef3c7", stroke="#d97706", shape="ellipse", font_size=18,
    )
    d.boxes.extend([start, model, tools, end])

    # Left annotation: model behaviour
    d.boxes.append(Box(
        "ann-model",
        "model step:\n"
        " 1. read messages + system_prompt\n"
        " 2. emit AIMessage with optional tool_calls\n"
        " 3. if no tool_calls → END",
        60, 500, 640, 130,
        fill="#ffffff", stroke="#94a3b8", dashed=True, font_size=16,
    ))

    # Right annotation: routing
    d.boxes.append(Box(
        "ann-route",
        "routing rule:\n"
        " AIMessage.tool_calls? → 'tools'\n"
        " else                  → END",
        1400, 540, 280, 90,
        fill="#ffffff", stroke="#94a3b8", dashed=True, font_size=16,
    ))

    # Tool inventory bar (below tools node)
    d.boxes.append(Box(
        "ann-tools",
        "ToolNode runs every tool.tool_calls in parallel and appends ToolMessage(s) back into state.\n"
        "Deep Agents pre-bundle five categories of tools so the model can mix them in one turn:",
        420, 660, 1240, 60,
        fill="#faf5ff", stroke="#7c3aed", dashed=True, font_size=15,
    ))

    bundle_y = 740
    bundles = [
        ("bundle-fs",
         "Built-in file tools\n(ls / read_file / write_file /\nedit_file / glob / grep)\nbacked by LocalShellBackend",
         "#ecfeff", "#0891b2"),
        ("bundle-exec",
         "execute(cmd)\nshell command in\nbackend.root_dir\n(timeout, env, output cap)",
         "#fef9c3", "#ca8a04"),
        ("bundle-custom",
         "Custom @tool functions\nyou pass via tools=[...]\n(here: rebuild_professors_index_tool)",
         "#dbeafe", "#2563eb"),
        ("bundle-sub",
         "Subagent dispatchers\nauto-generated from\nsubagents=[...]\n(here: AddProfessorSubagent)",
         "#fee2e2", "#dc2626"),
        ("bundle-skill",
         "Skills resolver\nauto-loads /skills/ entries\nso the model can fetch\nSKILL.md on demand",
         "#dcfce7", "#16a34a"),
    ]
    bundle_w = 240
    bundle_h = 140
    bundle_gap = 20
    total_w = len(bundles) * bundle_w + (len(bundles) - 1) * bundle_gap
    bundle_x0 = (d.width - total_w) // 2
    for i, (eid, label, fill, stroke) in enumerate(bundles):
        d.boxes.append(Box(
            eid, label,
            bundle_x0 + i * (bundle_w + bundle_gap), bundle_y,
            bundle_w, bundle_h,
            fill=fill, stroke=stroke, font_size=14,
        ))

    # Bottom takeaway
    d.boxes.append(Box(
        "takeaway",
        "Same loop as Lab 3, richer toolbox. The 'tools' step can read a file, run a shell command, "
        "call a custom function, dispatch a subagent, or pull in a skill — all on the same turn.",
        100, 920, 1520, 60,
        fill="#fffbeb", stroke="#d97706", font_size=16,
    ))

    # Arrows
    d.arrows.extend([
        Arrow("a-start-model", [start.bottom, model.top], stroke="#0284c7"),
        Arrow("a-model-tools",
              [model.right, tools.left],
              stroke="#4f46e5", label="tool_calls"),
        Arrow("a-tools-model",
              [tools.top, (tools.cx, 490), (model.cx + 70, 490), (model.cx + 70, model.y)],
              stroke="#7c3aed", dashed=True, label="loop back"),
        Arrow("a-model-end",
              [(model.cx - 50, model.y + model.height), (model.cx - 50, end.y)],
              stroke="#d97706", label="no tool_calls"),
        Arrow("a-ann-model", [(700, 565), (model.x, 565)], stroke="#94a3b8", dashed=True),
        Arrow("a-ann-route", [(1400, 585), (tools.x + tools.width, 585)], stroke="#94a3b8", dashed=True),
    ])

    # Fan-out arrows from tools node down to the five bundle boxes
    fan_anchor_y = 720
    for i, (eid, *_rest) in enumerate(bundles):
        bx_cx = bundle_x0 + i * (bundle_w + bundle_gap) + bundle_w / 2
        # vary the source x along the bottom of the tools node
        src_x = tools.x + 40 + i * ((tools.width - 80) / (len(bundles) - 1))
        d.arrows.append(Arrow(
            f"a-tools-{eid}",
            [(src_x, tools.y + tools.height), (src_x, fan_anchor_y), (bx_cx, fan_anchor_y), (bx_cx, bundle_y)],
            stroke="#7c3aed", dashed=True,
        ))

    return d


def build_professor_workspace_agent() -> Diagram:
    d = Diagram(
        name="02_professor_workspace_agent",
        title="ProfessorWorkspaceAgent — internals & runtime workspace",
        subtitle="System prompt + built-in file tools + one custom helper + one subagent dispatcher, all rooted in lab_4_deep_agents/sandbox/.",
        width=1720,
        height=1180,
    )

    # System prompt (top, full width)
    d.boxes.append(Box(
        "prompt",
        "System prompt (MAIN_SYSTEM_PROMPT) — rules:\n"
        "• Work only inside the runtime workspace.\n"
        "• Broad question → start with professors.md.   Named-professor question → open one dossier only if needed.\n"
        "• If the user asks to add a professor from an official BIT CSAT detail URL → delegate to AddProfessorSubagent.\n"
        "• When the subagent returns status=added → call rebuild_professors_index_tool, then reply with a short confirmation.\n"
        "• Recover silently from file-tool errors; only return the final useful answer.",
        60, 140, 1600, 150,
        fill="#dbeafe", stroke="#2563eb", font_size=15,
    ))

    # === Left lane — compiled deep agent ===
    loop_lane = Box(
        "loop-lane", None,
        60, 320, 460, 600,
        fill="#eff6ff", stroke="#60a5fa", dashed=True,
    )
    d.boxes.append(loop_lane)
    d.boxes.append(Box(
        "loop-header", "Compiled deep agent",
        88, 340, 404, 40,
        fill="#bfdbfe", stroke="#2563eb", font_size=18,
    ))
    model = Box("pwa-model", "model\n(LLM call)", 110, 420, 160, 80,
                fill="#eef2ff", stroke="#4f46e5", font_size=18)
    tools = Box("pwa-tools", "tools\n(ToolNode)", 320, 420, 160, 80,
                fill="#ede9fe", stroke="#7c3aed", font_size=18)
    d.boxes.extend([model, tools])
    d.boxes.append(Box(
        "pwa-construct",
        "create_deep_agent(\n"
        "  model               = build_model(settings),\n"
        "  tools               = [rebuild_professors_index_tool],\n"
        "  system_prompt       = MAIN_SYSTEM_PROMPT,\n"
        "  backend             = LocalShellBackend(\n"
        "                          root_dir=runtime_root,\n"
        "                          virtual_mode=False,\n"
        "                          env=SHELL_ENV),\n"
        "  subagents           = [add_professor_subagent],\n"
        "  checkpointer        = InMemorySaver(),\n"
        "  name='ProfessorWorkspaceAgent',\n"
        ")",
        90, 540, 400, 240,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=13,
    ))
    d.boxes.append(Box(
        "pwa-state",
        "Returns: { messages: [response],\n           files: {…}, todos: […] }\n"
        "thread_id keeps it across turns.",
        90, 800, 400, 100,
        fill="#fffbeb", stroke="#d97706", dashed=True, font_size=14,
    ))

    # === Center lane — bundled tools ===
    tools_lane = Box(
        "tools-lane", None,
        555, 320, 580, 600,
        fill="#fefce8", stroke="#eab308", dashed=True,
    )
    d.boxes.append(tools_lane)
    d.boxes.append(Box(
        "tools-header", "Tools the model can call",
        583, 340, 524, 40,
        fill="#fef9c3", stroke="#ca8a04", font_size=18,
    ))

    t1 = Box(
        "tool-fs",
        "Built-in file tools (Deep Agents bundle)\n\n"
        "• ls(path)             • read_file(path, offset, limit)\n"
        "• write_file(path, …)  • edit_file(path, old, new)\n"
        "• glob('**/*.md')      • grep(pattern, glob, output)\n\n"
        "All paths are relative to runtime_root.",
        575, 400, 540, 170,
        fill="#ecfeff", stroke="#0891b2", font_size=14,
    )
    t2 = Box(
        "tool-custom",
        "@tool rebuild_professors_index_tool()\n\n"
        "Reads every professors/*.md, builds one summary line\nper dossier with build_cached_summary_line, and rewrites\nprofessors.md sorted case-insensitively.\n\n"
        "Returns: { index_path, professor_count, preview }",
        575, 585, 540, 160,
        fill="#dbeafe", stroke="#2563eb", font_size=14,
    )
    t3 = Box(
        "tool-sub",
        "Subagent dispatcher: AddProfessorSubagent\n\n"
        "Auto-generated from subagents=[…]. Calling it hands off\nthe task with its own system prompt and the /skills/ tool,\nthen merges the subagent's final reply back into messages.",
        575, 760, 540, 150,
        fill="#fee2e2", stroke="#dc2626", font_size=14,
    )
    d.boxes.extend([t1, t2, t3])

    # === Right lane — runtime workspace files ===
    fs_lane = Box(
        "fs-lane", None,
        1170, 320, 490, 600,
        fill="#f0fdf4", stroke="#16a34a", dashed=True,
    )
    d.boxes.append(fs_lane)
    d.boxes.append(Box(
        "fs-header", "Runtime workspace (root_dir)",
        1198, 340, 434, 40,
        fill="#dcfce7", stroke="#15803d", font_size=18,
    ))
    d.boxes.append(Box(
        "fs-tree",
        "lab_4_deep_agents/sandbox/\n"
        "├── professors.md            ← compact index\n"
        "├── professors/\n"
        "│    ├── che-haiying.md      ← starter dossiers\n"
        "│    ├── cheng-cheng.md\n"
        "│    ├── …\n"
        "│    └── tang-haijing.md     ← added at runtime\n"
        "├── skills/\n"
        "│    └── add-professor-from-url/\n"
        "│         ├── SKILL.md\n"
        "│         └── scripts/add_professor_from_url.py\n"
        "└── incoming/                ← subagent OCR cache\n"
        "     ├── <slug>-ocr.md\n"
        "     └── <slug>-profile.json",
        1190, 400, 460, 380,
        fill="#ffffff", stroke="#16a34a", font_size=13,
    ))
    d.boxes.append(Box(
        "fs-note",
        "LocalShellBackend(virtual_mode=False)\nso every read_file/write_file hits the\nreal filesystem, not state['files'].",
        1190, 800, 460, 100,
        fill="#ecfdf5", stroke="#16a34a", dashed=True, font_size=14,
    ))

    # === Bottom — example flow for a broad question ===
    d.boxes.append(Box(
        "flow-header",
        "Example turn — \"Which professors should I look at for human-computer interaction?\"",
        60, 940, 1600, 40,
        fill="#1e293b", stroke="#0f172a", font_size=16, text_color="#ffffff",
    ))
    step_y = 1000
    step_w = 380
    step_h = 130
    step_gap = 30
    steps = [
        ("step-1",
         "1. model reads system_prompt\n + last user message\n → emits tool_call:\nread_file('professors.md')",
         "#ecfeff", "#0891b2"),
        ("step-2",
         "2. ToolNode runs read_file →\n returns the index lines.\n model picks the matching\n names from one short read.",
         "#ecfeff", "#0891b2"),
        ("step-3",
         "3. model composes the answer\n from the index alone — no\n dossier read needed for a\n broad topic question.",
         "#eef2ff", "#4f46e5"),
        ("step-4",
         "4. AIMessage has no tool_calls\n → END. State persisted under\n the same thread_id so the\n next turn keeps context.",
         "#fffbeb", "#d97706"),
    ]
    for i, (eid, label, fill, stroke) in enumerate(steps):
        d.boxes.append(Box(
            eid, label,
            60 + i * (step_w + step_gap), step_y, step_w, step_h,
            fill=fill, stroke=stroke, font_size=14,
        ))

    # === Arrows ===
    d.arrows.extend([
        # loop
        Arrow("a-pwa-mt", [model.right, tools.left],
              stroke="#4f46e5", label="tool_calls"),
        Arrow("a-pwa-tm",
              [tools.top, (tools.cx, 400), (model.cx, 400), model.top],
              stroke="#7c3aed", dashed=True),
        # tools → each tool box (fan-out from right edge of tools)
        Arrow("a-tools-fs",
              [(tools.x + tools.width, tools.cy - 20), (540, tools.cy - 20), (540, t1.cy), t1.left],
              stroke="#0891b2"),
        Arrow("a-tools-custom",
              [(tools.x + tools.width, tools.cy), (550, tools.cy), (550, t2.cy), t2.left],
              stroke="#2563eb"),
        Arrow("a-tools-sub",
              [(tools.x + tools.width, tools.cy + 20), (560, tools.cy + 20), (560, t3.cy), t3.left],
              stroke="#dc2626"),
        # tool → fs (file tools and rebuild touch the workspace)
        Arrow("a-fs-fs", [t1.right, (1140, t1.cy), (1140, t1.cy), (1170, t1.cy)],
              stroke="#0891b2", dashed=True),
        Arrow("a-rebuild-fs", [t2.right, (1140, t2.cy), (1140, t2.cy), (1170, t2.cy)],
              stroke="#2563eb", dashed=True),
        Arrow("a-sub-fs", [t3.right, (1140, t3.cy), (1140, t3.cy), (1170, t3.cy)],
              stroke="#dc2626", dashed=True),
        # bottom flow
        Arrow("a-step-1-2", [(60 + step_w, step_y + step_h / 2),
                             (60 + step_w + step_gap, step_y + step_h / 2)],
              stroke="#475569"),
        Arrow("a-step-2-3", [(60 + 2 * step_w + step_gap, step_y + step_h / 2),
                             (60 + 2 * step_w + 2 * step_gap, step_y + step_h / 2)],
              stroke="#475569"),
        Arrow("a-step-3-4", [(60 + 3 * step_w + 2 * step_gap, step_y + step_h / 2),
                             (60 + 3 * step_w + 3 * step_gap, step_y + step_h / 2)],
              stroke="#475569"),
    ])

    return d


def build_add_professor_subagent() -> Diagram:
    d = Diagram(
        name="03_add_professor_subagent",
        title="AddProfessorSubagent — skill workflow & live add pipeline",
        subtitle="Subagent prompt → /skills/add-professor-from-url/SKILL.md → one execute() call → add_professor_from_url.py crawl + OCR + structured extract.",
        width=1720,
        height=1200,
    )

    # Subagent prompt (top)
    d.boxes.append(Box(
        "subprompt",
        "Subagent prompt (ADD_SUBAGENT_PROMPT) — rules:\n"
        "• Inspect professors/ first.\n"
        "• Accept only official BIT CSAT detail URLs.\n"
        "• Use execute exactly once to run the skill-owned script.   • Do not rebuild professors.md.\n"
        "• Return one compact final line: { status, professor_name, slug, markdown_path, page_count }.",
        60, 140, 1600, 130,
        fill="#fee2e2", stroke="#dc2626", font_size=15,
    ))

    # === Left lane — subagent loop ===
    sub_lane = Box(
        "sub-lane", None,
        60, 300, 420, 580,
        fill="#fef2f2", stroke="#fca5a5", dashed=True,
    )
    d.boxes.append(sub_lane)
    d.boxes.append(Box(
        "sub-header", "Subagent compiled graph",
        88, 320, 364, 40,
        fill="#fee2e2", stroke="#dc2626", font_size=18,
    ))
    s_model = Box("sub-model", "model\n(LLM call)", 90, 400, 160, 80,
                  fill="#eef2ff", stroke="#4f46e5", font_size=18)
    s_tools = Box("sub-tools", "tools\n(ToolNode)", 290, 400, 160, 80,
                  fill="#ede9fe", stroke="#7c3aed", font_size=18)
    d.boxes.extend([s_model, s_tools])
    d.boxes.append(Box(
        "sub-tools-list",
        "Tools available to the subagent\n\n"
        "• Built-in file tools (ls, read_file, write_file,\n   edit_file, glob, grep)\n"
        "• execute(cmd) — bash, capped output\n"
        "• /skills/ resolver — loads SKILL.md\n\n"
        "(no rebuild_professors_index_tool —\n only the parent has it)",
        80, 510, 380, 200,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=14,
    ))
    d.boxes.append(Box(
        "sub-return",
        "Final reply (one line) is appended\nto the parent's messages and the\nsubgraph hits END.",
        80, 730, 380, 110,
        fill="#fffbeb", stroke="#d97706", dashed=True, font_size=14,
    ))

    # === Center lane — SKILL.md required workflow ===
    skill_lane = Box(
        "skill-lane", None,
        510, 300, 480, 580,
        fill="#fefce8", stroke="#eab308", dashed=True,
    )
    d.boxes.append(skill_lane)
    d.boxes.append(Box(
        "skill-header", "/skills/add-professor-from-url/SKILL.md",
        538, 320, 424, 40,
        fill="#fef9c3", stroke="#ca8a04", font_size=16,
    ))
    skill_steps = [
        ("sk-1", "1. Inspect professors/*.md to learn\n    the current workspace state.",
         "#ecfeff", "#0891b2"),
        ("sk-2", "2. Accept only URLs shaped like\n    https://isc.bit.edu.cn/schools/csat/\n    knowingprofessors5/b<NNNN>.htm",
         "#ecfeff", "#0891b2"),
        ("sk-3", "3. Run exactly one execute() command:\n    python skills/add-professor-from-url/\n    scripts/add_professor_from_url.py\n    --detail-url \"<url>\"\n    --professors-dir professors",
         "#fef9c3", "#ca8a04"),
        ("sk-4", "4. Read the script's stdout JSON.",
         "#dcfce7", "#16a34a"),
        ("sk-5", "5. status=added or duplicate_found →\n    return one summary line and stop.",
         "#dcfce7", "#16a34a"),
        ("sk-6", "6. status=failed → return one\n    failed summary line and stop.",
         "#fee2e2", "#dc2626"),
    ]
    sk_y = 380
    for i, (eid, label, fill, stroke) in enumerate(skill_steps):
        d.boxes.append(Box(
            eid, label,
            530, sk_y + i * 80, 440, 70,
            fill=fill, stroke=stroke, font_size=13,
        ))

    # === Right lane — script pipeline ===
    pipe_lane = Box(
        "pipe-lane", None,
        1010, 300, 650, 580,
        fill="#f0fdf4", stroke="#16a34a", dashed=True,
    )
    d.boxes.append(pipe_lane)
    d.boxes.append(Box(
        "pipe-header", "add_professor_from_url.py — pipeline",
        1038, 320, 594, 40,
        fill="#dcfce7", stroke="#15803d", font_size=18,
    ))
    pipeline = [
        ("p-1", "validate URL\n(regex: knowingprofessors5/b<digits>.htm)", "#ecfeff", "#0891b2"),
        ("p-2", "find_professor_listing_by_detail_url(url)\nresolves canonical name + listing", "#ecfeff", "#0891b2"),
        ("p-3", "duplicate check: professors/<slug>.md\nor any *.md whose detail_url matches", "#fef9c3", "#ca8a04"),
        ("p-4", "extract_image_urls(detail_url)\n→ poster image URLs", "#ede9fe", "#7c3aed"),
        ("p-5", "extract_professor_poster_markdown(...)\nQwen-VL OCR over each poster image\n→ writes incoming/<slug>-ocr.md", "#ede9fe", "#7c3aed"),
        ("p-6", "extract_structured_professor_profile(...)\nLLM → typed Pydantic profile\n→ writes incoming/<slug>-profile.json", "#dbeafe", "#2563eb"),
        ("p-7", "structured_profile_to_dossier(...) +\nrender_professor_markdown(...) +\nvalidate_professor_dossier(...)", "#dbeafe", "#2563eb"),
        ("p-8", "write professors/<slug>.md\nemit JSON: status=added", "#dcfce7", "#16a34a"),
    ]
    pp_y = 380
    pp_h = 60
    pp_gap = 5
    for i, (eid, label, fill, stroke) in enumerate(pipeline):
        d.boxes.append(Box(
            eid, label,
            1030, pp_y + i * (pp_h + pp_gap), 610, pp_h,
            fill=fill, stroke=stroke, font_size=13,
        ))

    # === Bottom — outcomes / example flow ===
    d.boxes.append(Box(
        "out-header",
        "Three outcomes the script can return — and how the subagent reports each",
        60, 920, 1600, 40,
        fill="#1e293b", stroke="#0f172a", font_size=16, text_color="#ffffff",
    ))
    outcomes = [
        ("out-added",
         "status=added\n\nNew dossier written to\nprofessors/<slug>.md.\nSubagent returns one line;\nparent then calls\nrebuild_professors_index_tool.",
         "#dcfce7", "#16a34a"),
        ("out-dup",
         "status=duplicate_found\n\nThe slug or detail URL is\nalready present.\nSubagent returns one line;\nparent skips the rebuild and\nreplies that we already have it.",
         "#fef9c3", "#ca8a04"),
        ("out-fail",
         "status=failed\n\nValidation, crawl, OCR, or\nstructured extraction raised.\nSubagent returns one failed\nline with the error; parent\nrecovers silently.",
         "#fee2e2", "#dc2626"),
    ]
    out_w = 510
    out_h = 180
    out_gap = 25
    out_x0 = (d.width - (3 * out_w + 2 * out_gap)) // 2
    for i, (eid, label, fill, stroke) in enumerate(outcomes):
        d.boxes.append(Box(
            eid, label,
            out_x0 + i * (out_w + out_gap), 980, out_w, out_h,
            fill=fill, stroke=stroke, font_size=14,
        ))

    # === Arrows ===
    d.arrows.extend([
        # subagent loop
        Arrow("a-sub-mt", [s_model.right, s_tools.left],
              stroke="#4f46e5", label="tool_calls"),
        Arrow("a-sub-tm",
              [s_tools.top, (s_tools.cx, 380), (s_model.cx, 380), s_model.top],
              stroke="#7c3aed", dashed=True),
        # subagent → skill (loads SKILL.md via /skills/ resolver)
        Arrow("a-sub-skill",
              [(s_tools.x + s_tools.width, s_tools.cy + 30),
               (s_tools.x + s_tools.width + 20, s_tools.cy + 30),
               (s_tools.x + s_tools.width + 20, 340),
               (530, 340)],
              stroke="#ca8a04", label="loads SKILL.md"),
        # skill step 3 → pipeline (execute() call hands off to script)
        Arrow("a-sk3-pipe",
              [(530 + 440, 380 + 2 * 80 + 35), (1030, 380 + 2 * 80 + 35)],
              stroke="#7c3aed", label="execute()"),
    ])
    # vertical pipeline arrows
    for i in range(len(pipeline) - 1):
        y1 = pp_y + i * (pp_h + pp_gap) + pp_h
        y2 = pp_y + (i + 1) * (pp_h + pp_gap)
        d.arrows.append(Arrow(
            f"a-pipe-{i}",
            [(1030 + 305, y1), (1030 + 305, y2)],
            stroke="#475569",
        ))
    # vertical skill arrows
    for i in range(len(skill_steps) - 1):
        y1 = sk_y + i * 80 + 70
        y2 = sk_y + (i + 1) * 80
        d.arrows.append(Arrow(
            f"a-sk-{i}",
            [(530 + 220, y1), (530 + 220, y2)],
            stroke="#475569",
        ))
    # pipeline → outcomes
    for i, eid in enumerate(["out-added", "out-dup", "out-fail"]):
        cx = out_x0 + i * (out_w + out_gap) + out_w / 2
        # last pipeline step bottom
        last_y = pp_y + len(pipeline) * (pp_h + pp_gap) - pp_gap
        d.arrows.append(Arrow(
            f"a-pipe-{eid}",
            [(1030 + 305, last_y), (1030 + 305, 940), (cx, 940), (cx, 980)],
            stroke="#475569", dashed=True,
        ))

    return d


def build_lab4_composition() -> Diagram:
    d = Diagram(
        name="04_lab4_composition",
        title="Lab 4 composition — deep agent + subagent + skills + workspace",
        subtitle="One create_deep_agent call wires the model to a LocalShellBackend, an InMemorySaver, one custom tool, and one subagent that owns the live add workflow.",
        width=1720,
        height=1200,
    )

    # === Top row: Student → parent agent → checkpointer ===
    student = Box(
        "student",
        "Student\n\nconfig = { configurable:\n  { thread_id: 'lab4-…' } }",
        60, 140, 260, 130,
        fill="#e7f5ff", stroke="#1971c2", font_size=15,
    )
    parent = Box(
        "parent",
        "ProfessorWorkspaceAgent  ← create_deep_agent(...)\n"
        "  model           = build_model(settings)\n"
        "  tools           = [rebuild_professors_index_tool]\n"
        "  subagents       = [add_professor_subagent]\n"
        "  backend         = LocalShellBackend(root_dir=runtime_root, virtual_mode=False, env=SHELL_ENV)\n"
        "  system_prompt   = MAIN_SYSTEM_PROMPT",
        380, 140, 820, 130,
        fill="#fff7ed", stroke="#ea580c", font_size=14, stroke_width=3.0,
    )
    checkpointer = Box(
        "ckpt",
        "InMemorySaver()\n\nthread_id → StateSnapshot {\n  messages: [...],\n  files:    {...},\n  todos:    [...] }",
        1240, 140, 420, 130,
        fill="#fffbeb", stroke="#d97706", font_size=14,
    )
    d.boxes.extend([student, parent, checkpointer])

    # === Middle: parent loop and subagent loop side-by-side ===
    parent_lane = Box(
        "p-lane", None,
        60, 320, 660, 460,
        fill="#eff6ff", stroke="#60a5fa", dashed=True,
    )
    parent_header = Box(
        "p-header", "ProfessorWorkspaceAgent (compiled deep agent)",
        90, 340, 600, 50,
        fill="#dbeafe", stroke="#2563eb", font_size=18,
    )
    p_model = Box("p-model", "model", 130, 420, 160, 80,
                  fill="#eef2ff", stroke="#4f46e5", font_size=20)
    p_tools = Box("p-tools", "tools (ToolNode)", 360, 420, 240, 80,
                  fill="#ede9fe", stroke="#7c3aed", font_size=18)
    p_inventory = Box(
        "p-inv",
        "Bound on this node:\n"
        "• Built-in file tools (ls / read_file / write_file / edit_file / glob / grep)\n"
        "• execute(cmd)               • rebuild_professors_index_tool\n"
        "• Subagent dispatcher: AddProfessorSubagent\n"
        "• /skills/ resolver",
        90, 540, 600, 220,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=14,
    )
    d.boxes.extend([parent_lane, parent_header, p_model, p_tools, p_inventory])

    sub_lane = Box(
        "s-lane", None,
        870, 320, 530, 460,
        fill="#fef2f2", stroke="#fca5a5", dashed=True,
    )
    sub_header = Box(
        "s-header", "AddProfessorSubagent (compiled subgraph)",
        900, 340, 470, 50,
        fill="#fee2e2", stroke="#dc2626", font_size=18,
    )
    s_model = Box("s-model", "model", 920, 420, 150, 80,
                  fill="#eef2ff", stroke="#4f46e5", font_size=20)
    s_tools = Box("s-tools", "tools (ToolNode)", 1140, 420, 220, 80,
                  fill="#ede9fe", stroke="#7c3aed", font_size=18)
    s_inventory = Box(
        "s-inv",
        "Bound on this node:\n"
        "• Built-in file tools\n"
        "• execute(cmd)\n"
        "• /skills/ resolver\n"
        "  (no rebuild — only parent has it)\n\n"
        "Returns one summary line\nback to parent.messages.",
        890, 540, 490, 220,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=14,
    )
    d.boxes.extend([sub_lane, sub_header, s_model, s_tools, s_inventory])

    # === Right column: backend + skills + workspace ===
    backend_box = Box(
        "be",
        "LocalShellBackend\nroot_dir = lab_4_deep_agents/sandbox\nvirtual_mode = False  (writes hit disk)\ntimeout = 300s, output cap = 100KB\nenv = SHELL_ENV (LLM/OCR keys)",
        1430, 320, 240, 180,
        fill="#fefce8", stroke="#ca8a04", font_size=12,
    )
    skill_box = Box(
        "sk",
        "/skills/\n  add-professor-from-url/\n    SKILL.md\n    scripts/add_professor_from_url.py\n\nResolved by the deep agent's\nbuilt-in skills loader.",
        1430, 520, 240, 180,
        fill="#dcfce7", stroke="#16a34a", font_size=12,
    )
    d.boxes.extend([backend_box, skill_box])

    # Internal loop arrows (parent + subagent)
    d.arrows.extend([
        Arrow("a-p-mt", [p_model.right, p_tools.left], stroke="#4f46e5"),
        Arrow("a-p-tm",
              [p_tools.top, (p_tools.cx, 400), (p_model.cx, 400), p_model.top],
              stroke="#7c3aed", dashed=True),
        Arrow("a-s-mt", [s_model.right, s_tools.left], stroke="#4f46e5"),
        Arrow("a-s-tm",
              [s_tools.top, (s_tools.cx, 400), (s_model.cx, 400), s_model.top],
              stroke="#7c3aed", dashed=True),
    ])

    # Handoff arrows between lanes
    handoff_label = Box(
        "h-label",
        "AddProfessorSubagent dispatcher\n→ runs the subagent subgraph,\n   then merges its final reply",
        730, 410, 130, 100,
        fill="#fff7ed", stroke="#ea580c", dashed=True, font_size=12,
    )
    d.boxes.append(handoff_label)
    d.arrows.extend([
        Arrow("a-handoff",
              [(p_tools.x + p_tools.width, p_tools.cy - 10),
               (730, p_tools.cy - 10),
               (730, parent_header.cy + 30),
               (sub_header.x, parent_header.cy + 30),
               (sub_header.x, sub_header.cy)],
              stroke="#ea580c"),
        Arrow("a-return",
              [(s_inventory.x, s_inventory.cy + 70),
               (730, s_inventory.cy + 70),
               (730, p_inventory.cy + 70),
               (p_inventory.x + p_inventory.width, p_inventory.cy + 70)],
              stroke="#16a34a", dashed=True, label="final summary line"),
    ])

    # Top wiring
    d.arrows.extend([
        Arrow("a-stu-parent", [student.right, parent.left],
              stroke="#1971c2", label="invoke / stream"),
        Arrow("a-parent-ckpt", [parent.right, checkpointer.left],
              stroke="#ea580c", label="reads + writes state"),
        Arrow("a-parent-plane",
              [(parent.x + 200, parent.y + parent.height),
               (parent.x + 200, 310),
               (parent_header.cx, 310),
               parent_header.top],
              stroke="#2563eb"),
        Arrow("a-parent-slane",
              [(parent.x + 600, parent.y + parent.height),
               (parent.x + 600, 305),
               (sub_header.cx, 305),
               sub_header.top],
              stroke="#dc2626"),
    ])

    # tools → backend / skills
    d.arrows.extend([
        Arrow("a-ptools-be",
              [(p_tools.x + p_tools.width, p_tools.cy + 20),
               (700, p_tools.cy + 20),
               (700, backend_box.cy - 30),
               (backend_box.x, backend_box.cy - 30)],
              stroke="#ca8a04", dashed=True, label="every file tool / execute"),
        Arrow("a-stools-be",
              [(s_tools.x + s_tools.width, s_tools.cy - 20),
               (1410, s_tools.cy - 20),
               (1410, backend_box.cy + 30),
               (backend_box.x, backend_box.cy + 30)],
              stroke="#ca8a04", dashed=True),
        Arrow("a-stools-sk",
              [(s_tools.x + s_tools.width, s_tools.cy + 20),
               (1410, s_tools.cy + 20),
               (1410, skill_box.cy),
               (skill_box.x, skill_box.cy)],
              stroke="#16a34a", dashed=True, label="loads SKILL.md"),
    ])

    # === Bottom: workspace + memory timeline ===
    ws_header = Box(
        "ws-header",
        "Same workspace + same thread_id across turns — files persist between calls",
        60, 810, 1600, 40,
        fill="#1e293b", stroke="#0f172a", font_size=16, text_color="#ffffff",
    )
    d.boxes.append(ws_header)

    turns = [
        ("turn-1",
         "Turn 1 — broad question\n\nparent.model emits read_file('professors.md').\nLocalShellBackend reads the index from disk.\nReply uses just the index lines.",
         "#ecfeff", "#0891b2"),
        ("turn-2",
         "Turn 2 — named professor\n\nparent.model reads professors.md, then opens\nprofessors/<slug>.md only when more detail is\nneeded. Same thread_id, so prior context stays.",
         "#dbeafe", "#2563eb"),
        ("turn-3",
         "Turn 3 — add from URL\n\nparent dispatches AddProfessorSubagent.\nSubagent loads SKILL.md, runs execute() once.\nNew dossier appears under professors/.",
         "#fee2e2", "#dc2626"),
        ("turn-4",
         "Turn 4 — index + confirm\n\nOn status=added the parent calls\nrebuild_professors_index_tool, then replies\nwith the final confirmation. State is persisted.",
         "#dcfce7", "#16a34a"),
    ]
    tw = 380
    th = 200
    tg = 25
    tx0 = (d.width - (4 * tw + 3 * tg)) // 2
    for i, (eid, label, fill, stroke) in enumerate(turns):
        d.boxes.append(Box(
            eid, label,
            tx0 + i * (tw + tg), 880, tw, th,
            fill=fill, stroke=stroke, font_size=14,
        ))
    for i in range(3):
        x1 = tx0 + i * (tw + tg) + tw
        x2 = tx0 + (i + 1) * (tw + tg)
        d.arrows.append(Arrow(
            f"a-turn-{i}",
            [(x1, 980), (x2, 980)],
            stroke="#475569",
        ))

    return d


def main() -> None:
    print("Rendering Lab 4 teaching diagrams...")
    for builder in (
        build_deep_agent_anatomy,
        build_professor_workspace_agent,
        build_add_professor_subagent,
        build_lab4_composition,
    ):
        emit(builder())
    print("Done.")


if __name__ == "__main__":
    main()
