"""Render Lab 3 teaching diagrams as SVG files.

Four focused diagrams are emitted under this directory:

  01_specialist_anatomy        - inside one specialist subgraph (START -> llm -> ToolNode -> llm)
  02_professor_lookup_agent    - ProfessorLookupAgent: tools, prompt, Neo4j queries
  03_research_match_agent      - ResearchMatchAgent: tools, prompt, Neo4j query
  04_swarm_composition         - create_swarm parent, handoffs, active_agent, checkpointer

Run:
    uv run python lab_3_langgraph_swarm/teaching_diagrams/render.py
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
        text_w = len(arrow.label) * 7 + 12
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


def build_specialist_anatomy() -> Diagram:
    d = Diagram(
        name="01_specialist_anatomy",
        title="Anatomy of a Specialist Subgraph",
        subtitle="Each specialist is a tiny LangGraph: START -> llm -> ToolNode -> llm, until the model stops calling tools.",
        width=1500,
        height=940,
    )

    # State schema (top-left)
    d.boxes.append(Box(
        "state",
        "Lab3State (extends SwarmState)\n\n• messages: list[BaseMessage]\n• active_agent: str\n\n(only two fields persist)",
        60, 150, 380, 170,
        fill="#eff6ff", stroke="#2563eb", font_size=16,
    ))

    # Code snippet (top-right)
    d.boxes.append(Box(
        "build-fn",
        "build_specialist_agent(name, prompt, tools):\n  bound_model = model.bind_tools(tools)\n  tool_node  = ToolNode(tools)\n  workflow   = StateGraph(Lab3State)\n  workflow.add_node('llm',   call_model)\n  workflow.add_node('tools', tool_node)\n  workflow.add_edge(START, 'llm')\n  workflow.add_conditional_edges('llm', route_after_model)\n  workflow.add_edge('tools', 'llm')\n  return workflow.compile(name=name)",
        1000, 150, 460, 230,
        fill="#f8fafc", stroke="#475569", font_size=16, dashed=True,
    ))

    # Center subgraph nodes
    start = Box(
        "n-start", "START",
        660, 410, 140, 56,
        fill="#e0f2fe", stroke="#0284c7", shape="ellipse", font_size=18,
    )
    llm = Box(
        "n-llm", "llm\n(call_model)",
        620, 520, 220, 90,
        fill="#eef2ff", stroke="#4f46e5", font_size=18,
    )
    tools = Box(
        "n-tools", "tools\n(ToolNode)",
        920, 520, 220, 90,
        fill="#ede9fe", stroke="#7c3aed", font_size=18,
    )
    end = Box(
        "n-end", "END",
        660, 720, 140, 56,
        fill="#fef3c7", stroke="#d97706", shape="ellipse", font_size=18,
    )
    d.boxes.extend([start, llm, tools, end])

    # call_model annotation (left of llm)
    d.boxes.append(Box(
        "ann-call",
        "call_model(state, config):\n 1. emit teaching event\n 2. response = bound_model.invoke(\n      [SystemMessage(prompt), *state['messages']])\n 3. return {\n      messages: [response],\n      active_agent: agent_name }",
        60, 480, 520, 170,
        fill="#ffffff", stroke="#94a3b8", dashed=True, font_size=16,
    ))

    # route_after_model annotation (right of tools)
    d.boxes.append(Box(
        "ann-route",
        "route_after_model(state):\n  last = state['messages'][-1]\n  if AIMessage and last.tool_calls:\n      return 'tools'\n  return END",
        1190, 480, 270, 130,
        fill="#ffffff", stroke="#94a3b8", dashed=True, font_size=16,
    ))

    # ToolNode side note
    d.boxes.append(Box(
        "ann-tools",
        "ToolNode runs every tool the model called\nin parallel, then appends ToolMessage(s)\nback into state['messages'].",
        920, 640, 540, 60,
        fill="#faf5ff", stroke="#7c3aed", dashed=True, font_size=16,
    ))

    # Bottom takeaway
    d.boxes.append(Box(
        "takeaway",
        "The loop is the whole agent. It stops only when the model returns no tool_calls — then the subgraph hits END and control returns to the parent swarm.",
        100, 820, 1300, 60,
        fill="#fffbeb", stroke="#d97706", font_size=16,
    ))

    # Arrows
    d.arrows.extend([
        Arrow("a-start-llm", [start.bottom, llm.top], stroke="#0284c7"),
        Arrow("a-llm-tools",
              [llm.right, tools.left],
              stroke="#4f46e5", label="tool_calls"),
        Arrow("a-tools-llm",
              [tools.top, (tools.cx, 470), (llm.cx + 60, 470), (llm.cx + 60, llm.y)],
              stroke="#7c3aed", dashed=True, label="loop back"),
        Arrow("a-llm-end",
              [(llm.cx - 40, llm.y + llm.height), (llm.cx - 40, end.y)],
              stroke="#d97706", label="no tool_calls"),
        Arrow("a-call-llm", [(580, 565), (llm.x, 565)], stroke="#94a3b8", dashed=True),
        Arrow("a-route-llm", [(1190, 545), (tools.x + tools.width, 545)], stroke="#94a3b8", dashed=True),
    ])

    return d


def build_professor_lookup_agent() -> Diagram:
    d = Diagram(
        name="02_professor_lookup_agent",
        title="ProfessorLookupAgent — internals & Neo4j queries",
        subtitle="System prompt + bound tools + the Cypher each tool runs against the Lab 3 typed graph.",
        width=1720,
        height=1140,
    )

    # System prompt summary (top, full width)
    d.boxes.append(Box(
        "prompt",
        "System prompt: handle named-professor questions.\n"
        "Rules: call resolve_professor_tool first. If exactly one match, call get_professor_facts_tool before answering.\n"
        "Handoff: if the user asks for topic-based recommendations, call transfer_to_research_match.\n"
        "Style: ground every answer in tool output; never answer from prior knowledge.",
        60, 140, 1600, 120,
        fill="#dbeafe", stroke="#2563eb", font_size=16,
    ))

    # Left column - agent loop
    loop_lane = Box(
        "loop-lane", None,
        60, 300, 460, 580,
        fill="#eff6ff", stroke="#60a5fa", dashed=True,
    )
    d.boxes.append(loop_lane)
    d.boxes.append(Box(
        "loop-header", "Compiled subgraph",
        88, 320, 404, 40,
        fill="#bfdbfe", stroke="#2563eb", font_size=18,
    ))
    llm = Box("plk-llm", "llm\ncall_model", 110, 400, 160, 80,
              fill="#eef2ff", stroke="#4f46e5", font_size=18)
    tools = Box("plk-tools", "tools\nToolNode", 320, 400, 160, 80,
                fill="#ede9fe", stroke="#7c3aed", font_size=18)
    d.boxes.extend([llm, tools])
    d.boxes.append(Box(
        "plk-bind",
        "model.bind_tools([\n  resolve_professor_tool,\n  get_professor_facts_tool,\n  transfer_to_research_match,\n])",
        90, 530, 400, 130,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=16,
    ))
    d.boxes.append(Box(
        "plk-emit",
        "Each tool call also emits a teaching event:\n{ kind: 'tool_call', agent: 'ProfessorLookupAgent',\n  tool: <name>, input: {...} }",
        90, 690, 400, 90,
        fill="#fffbeb", stroke="#d97706", dashed=True, font_size=16,
    ))
    d.boxes.append(Box(
        "plk-state",
        "Returns: { messages: [response],\n           active_agent: 'ProfessorLookupAgent' }",
        90, 800, 400, 60,
        fill="#f1f5f9", stroke="#64748b", dashed=True, font_size=16,
    ))

    # Center column - the three tools
    tools_lane = Box(
        "tools-lane", None,
        555, 300, 480, 580,
        fill="#fefce8", stroke="#eab308", dashed=True,
    )
    d.boxes.append(tools_lane)
    d.boxes.append(Box(
        "tools-header", "Bound tools (3)",
        583, 320, 424, 40,
        fill="#fef9c3", stroke="#ca8a04", font_size=18,
    ))

    t1 = Box(
        "tool-resolve",
        "@tool resolve_professor_tool(name_hint: str)\n\n"
        "Goal: turn a fuzzy name into a ranked\nlist of Professor matches.\n\n"
        "Returns: { matches: [{name, url, score}] }",
        575, 380, 440, 150,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    )
    t2 = Box(
        "tool-facts",
        "@tool get_professor_facts_tool(professor_name: str)\n\n"
        "Goal: fetch grounded facts for one professor —\norgs, topics, experiences, publications.\n\n"
        "Returns: { facts: [{source, target, predicate}] }",
        575, 545, 440, 150,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    )
    t3 = Box(
        "tool-handoff",
        "@tool transfer_to_research_match()\n\n"
        "Goal: hand control to ResearchMatchAgent.\nReturns Command(goto=..., graph=PARENT,\nupdate={ active_agent: 'ResearchMatchAgent' }).\n\nNot a Neo4j call — pure routing.",
        575, 710, 440, 160,
        fill="#fff7ed", stroke="#ea580c", font_size=16,
    )
    d.boxes.extend([t1, t2, t3])

    # Right column - Cypher queries
    cypher_lane = Box(
        "cypher-lane", None,
        1070, 300, 590, 580,
        fill="#f0fdf4", stroke="#16a34a", dashed=True,
    )
    d.boxes.append(cypher_lane)
    d.boxes.append(Box(
        "cypher-header", "Neo4jQueryService — Cypher",
        1098, 320, 534, 40,
        fill="#dcfce7", stroke="#15803d", font_size=18,
    ))

    c1 = Box(
        "cy-resolve",
        "MATCH (p:Professor)\n"
        "WITH p, [v IN [p.name] + p.aliases\n        WHERE trim(v) <> ''] AS names\n"
        "WHERE any(v IN names\n         WHERE toLower(v) CONTAINS $needle)\n"
        "RETURN p.name, p.detail_url,\n       <exact|prefix|contains> AS score\n"
        "ORDER BY score LIMIT 5",
        1090, 380, 555, 160,
        fill="#ffffff", stroke="#16a34a", font_size=13,
    )
    c2 = Box(
        "cy-facts",
        "MATCH (p:Professor) WHERE p.name = $name\n"
        "CALL (p) {\n"
        "  MATCH (p)-[:AFFILIATED_WITH]->(o:Organization)         RETURN ...\n"
        "  UNION ALL MATCH (p)-[:HAS_RESEARCH_INTEREST]->(t)      RETURN ...\n"
        "  UNION ALL MATCH (p)-[:HAS_EXPERIENCE]->(e)             RETURN ...\n"
        "  UNION ALL MATCH (p)-[:AUTHORED]->(pub)                 RETURN ...\n"
        "  UNION ALL MATCH (p)-[:RECEIVED]->(a)                   RETURN ...\n"
        "  UNION ALL UNWIND p.emails / p.phones / p.websites      RETURN ...\n"
        "}\nRETURN source_name, target_name, relationship_type LIMIT 18",
        1090, 555, 555, 200,
        fill="#ffffff", stroke="#16a34a", font_size=12,
    )
    c3 = Box(
        "cy-handoff",
        "(no Cypher)\n\n"
        "return Command(\n"
        "  goto='ResearchMatchAgent',\n"
        "  graph=Command.PARENT,\n"
        "  update={'messages': [...ToolMessage],\n"
        "          'active_agent': 'ResearchMatchAgent'},\n)",
        1090, 770, 555, 170,
        fill="#ffffff", stroke="#ea580c", font_size=13,
    )
    d.boxes.extend([c1, c2, c3])

    # Bottom: example flow
    d.boxes.append(Box(
        "flow-header", "Example turn — \"What are CHENG Cheng's research interests?\"",
        60, 905, 1600, 40,
        fill="#1e293b", stroke="#0f172a", font_size=17, text_color="#ffffff",
    ))
    d.boxes.append(Box(
        "step-1",
        "1. llm sees question →\nemits tool_call:\nresolve_professor_tool(\n  name_hint='CHENG Cheng')",
        60, 960, 380, 130,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    ))
    d.boxes.append(Box(
        "step-2",
        "2. ToolNode runs Cypher,\nreturns 1 match →\nllm emits next tool_call:\nget_professor_facts_tool(\n  professor_name='CHENG Cheng')",
        470, 960, 380, 130,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    ))
    d.boxes.append(Box(
        "step-3",
        "3. ToolNode returns facts:\nresearch interests,\norganizations, publications →\nllm composes grounded answer",
        880, 960, 380, 130,
        fill="#eef2ff", stroke="#4f46e5", font_size=16,
    ))
    d.boxes.append(Box(
        "step-4",
        "4. AIMessage has\nno tool_calls →\nroute_after_model returns END.\nactive_agent stays\n'ProfessorLookupAgent'.",
        1290, 960, 370, 130,
        fill="#fffbeb", stroke="#d97706", font_size=16,
    ))

    # Arrows
    d.arrows.extend([
        Arrow("a-llm-tools",
              [llm.right, tools.left],
              stroke="#4f46e5", label="tool_calls"),
        Arrow("a-tools-llm",
              [tools.top, (tools.cx, 380), (llm.cx, 380), llm.top],
              stroke="#7c3aed", dashed=True),
        # tools -> each tool box
        Arrow("a-tn-resolve",
              [tools.right, (520, tools.cy), (520, t1.cy), t1.left],
              stroke="#0891b2"),
        Arrow("a-tn-facts",
              [tools.right, (530, tools.cy), (530, t2.cy), t2.left],
              stroke="#0891b2"),
        Arrow("a-tn-handoff",
              [tools.right, (540, tools.cy), (540, t3.cy), t3.left],
              stroke="#ea580c"),
        # tool -> cypher
        Arrow("a-resolve-cy", [t1.right, c1.left], stroke="#16a34a", dashed=True),
        Arrow("a-facts-cy", [t2.right, c2.left], stroke="#16a34a", dashed=True),
        Arrow("a-handoff-cy", [t3.right, c3.left], stroke="#ea580c", dashed=True),
        # bottom flow
        Arrow("a-step-1-2", [(440, 1025), (470, 1025)], stroke="#475569"),
        Arrow("a-step-2-3", [(850, 1025), (880, 1025)], stroke="#475569"),
        Arrow("a-step-3-4", [(1260, 1025), (1290, 1025)], stroke="#475569"),
    ])

    return d


def build_swarm_composition() -> Diagram:
    d = Diagram(
        name="04_swarm_composition",
        title="Swarm composition — handoffs, active_agent, and same-thread memory",
        subtitle="create_swarm wires the two specialists into one parent graph; a handoff returns Command(graph=PARENT) which atomically swaps the active_agent in checkpointed state.",
        width=1720,
        height=1180,
    )

    # === Top row: Student → swarm parent → checkpointer ===
    student = Box(
        "student",
        "Student\n\nconfig = { configurable:\n  { thread_id: 'lab3-xyz' } }",
        60, 140, 260, 130,
        fill="#e7f5ff", stroke="#1971c2", font_size=16,
    )
    parent = Box(
        "parent",
        "create_swarm(\n"
        "  [ ProfessorLookupAgent, ResearchMatchAgent ],\n"
        "  default_active_agent='ProfessorLookupAgent',\n"
        "  state_schema=Lab3State,\n"
        ")",
        380, 140, 760, 130,
        fill="#fff7ed", stroke="#ea580c", font_size=16, stroke_width=3.0,
    )
    checkpointer = Box(
        "ckpt",
        ".compile(checkpointer=InMemorySaver())\n\n"
        "thread_id  →  StateSnapshot {\n"
        "  messages:     [ ... ],\n"
        "  active_agent: '<one specialist>' }",
        1200, 140, 460, 130,
        fill="#fffbeb", stroke="#d97706", font_size=16,
    )
    d.boxes.extend([student, parent, checkpointer])

    # === Middle: two specialist lanes ===
    plk_lane = Box(
        "plk-lane", None,
        60, 330, 720, 460,
        fill="#eff6ff", stroke="#60a5fa", dashed=True,
    )
    plk_header = Box(
        "plk-header", "ProfessorLookupAgent\n(compiled subgraph)",
        90, 350, 660, 70,
        fill="#dbeafe", stroke="#2563eb", font_size=20,
    )
    plk_llm = Box("p-llm", "llm", 130, 470, 160, 80,
                  fill="#eef2ff", stroke="#4f46e5", font_size=20)
    plk_tools = Box("p-tools", "tools (ToolNode)", 360, 470, 240, 80,
                    fill="#ede9fe", stroke="#7c3aed", font_size=18)
    plk_tool_list = Box(
        "p-tool-list",
        "Bound tools:\n"
        "• resolve_professor_tool   — Cypher:\n"
        "    MATCH (:Professor) WHERE alias CONTAINS …\n"
        "• get_professor_facts_tool — Cypher:\n"
        "    MATCH (p)-[*]->(...) UNION ALL ...\n"
        "• transfer_to_research_match — handoff:\n"
        "    Command(goto, graph=PARENT)",
        90, 600, 660, 180,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=13,
    )
    d.boxes.extend([plk_lane, plk_header, plk_llm, plk_tools, plk_tool_list])

    rm_lane = Box(
        "rm-lane", None,
        940, 330, 720, 460,
        fill="#ecfdf5", stroke="#22c55e", dashed=True,
    )
    rm_header = Box(
        "rm-header", "ResearchMatchAgent\n(compiled subgraph)",
        970, 350, 660, 70,
        fill="#bbf7d0", stroke="#15803d", font_size=20,
    )
    rm_llm = Box("r-llm", "llm", 1010, 470, 160, 80,
                 fill="#ecfdf5", stroke="#16a34a", font_size=20)
    rm_tools = Box("r-tools", "tools (ToolNode)", 1240, 470, 240, 80,
                   fill="#ede9fe", stroke="#7c3aed", font_size=18)
    rm_tool_list = Box(
        "r-tool-list",
        "Bound tools:\n"
        "• find_professors_by_topics_tool — Cypher:\n"
        "    MATCH (:Professor)-[:HAS_RESEARCH_INTEREST]\n"
        "          ->(:ResearchTopic)\n"
        "• transfer_to_professor_lookup — handoff:\n"
        "    Command(goto, graph=PARENT)",
        970, 600, 660, 180,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=13,
    )
    d.boxes.extend([rm_lane, rm_header, rm_llm, rm_tools, rm_tool_list])

    # Internal loop arrows for each specialist
    d.arrows.extend([
        Arrow("a-p-llm-tools", [plk_llm.right, plk_tools.left], stroke="#4f46e5"),
        Arrow("a-p-tools-llm",
              [plk_tools.top, (plk_tools.cx, 450), (plk_llm.cx, 450), plk_llm.top],
              stroke="#7c3aed", dashed=True),
        Arrow("a-r-llm-tools", [rm_llm.right, rm_tools.left], stroke="#16a34a"),
        Arrow("a-r-tools-llm",
              [rm_tools.top, (rm_tools.cx, 450), (rm_llm.cx, 450), rm_llm.top],
              stroke="#7c3aed", dashed=True),
    ])

    # Handoff arrows between lanes (centered between them, stacked)
    handoff_label_p_to_r = Box(
        "h1-label",
        "transfer_to_research_match\n→ Command(goto='ResearchMatchAgent',\n   graph=PARENT,\n   update={ active_agent: 'ResearchMatchAgent' })",
        790, 380, 140, 110,
        fill="#fff7ed", stroke="#ea580c", dashed=True, font_size=16,
    )
    handoff_label_r_to_p = Box(
        "h2-label",
        "transfer_to_professor_lookup\n→ Command(goto='ProfessorLookupAgent',\n   graph=PARENT,\n   update={ active_agent: 'ProfessorLookupAgent' })",
        790, 660, 140, 110,
        fill="#fff7ed", stroke="#ea580c", dashed=True, font_size=16,
    )
    d.boxes.extend([handoff_label_p_to_r, handoff_label_r_to_p])

    d.arrows.extend([
        Arrow("a-handoff-pr",
              [(plk_header.x + plk_header.width, plk_header.cy),
               (rm_header.x, rm_header.cy)],
              stroke="#ea580c"),
        Arrow("a-handoff-rp",
              [(rm_tool_list.x, rm_tool_list.cy),
               (plk_tool_list.x + plk_tool_list.width, plk_tool_list.cy)],
              stroke="#ea580c"),
    ])

    # Top connectors
    d.arrows.extend([
        Arrow("a-student-parent", [student.right, parent.left],
              stroke="#1971c2", label="invoke / stream"),
        Arrow("a-parent-ckpt", [parent.right, checkpointer.left],
              stroke="#ea580c", label="reads + writes state"),
        Arrow("a-parent-plk",
              [(parent.x + 180, parent.y + parent.height),
               (parent.x + 180, 320),
               (plk_header.cx, 320),
               plk_header.top],
              stroke="#2563eb", label="goto active_agent"),
        Arrow("a-parent-rm",
              [(parent.x + 580, parent.y + parent.height),
               (parent.x + 580, 320),
               (rm_header.cx, 320),
               rm_header.top],
              stroke="#16a34a", label="goto active_agent"),
    ])

    # === Bottom: state / memory timeline ===
    timeline_header = Box(
        "tl-header",
        "Same-thread memory: how active_agent persists across turns",
        60, 830, 1600, 44,
        fill="#1e293b", stroke="#0f172a", font_size=18, text_color="#ffffff",
    )
    d.boxes.append(timeline_header)

    snap1 = Box(
        "snap1",
        "Turn 1 — start\n\n"
        "active_agent = 'ProfessorLookupAgent'\n"
        "messages = [ HumanMessage('which profs match\n             computer vision?') ]",
        60, 890, 380, 180,
        fill="#eef2ff", stroke="#4f46e5", font_size=16,
    )
    snap2 = Box(
        "snap2",
        "Turn 1 — after handoff\n\n"
        "active_agent = 'ResearchMatchAgent'\n"
        "messages += [ AIMessage(tool_calls=[transfer_to_…]),\n"
        "              ToolMessage('Successfully transferred…'),\n"
        "              AIMessage(<grounded answer>) ]",
        470, 890, 440, 180,
        fill="#ecfdf5", stroke="#16a34a", font_size=16,
    )
    snap3 = Box(
        "snap3",
        "Turn 2 — same thread_id\n\n"
        "Swarm reads checkpoint → starts at\nResearchMatchAgent (no supervisor needed).\n"
        "If user pivots to one named professor,\nResearchMatchAgent calls\ntransfer_to_professor_lookup.",
        940, 890, 380, 180,
        fill="#fff7ed", stroke="#ea580c", font_size=16,
    )
    snap4 = Box(
        "snap4",
        "New thread_id ⇒ fresh state.\n\n"
        "default_active_agent kicks in:\nthe conversation begins again at\nProfessorLookupAgent with empty messages.",
        1350, 890, 310, 180,
        fill="#fffbeb", stroke="#d97706", font_size=16,
    )
    d.boxes.extend([snap1, snap2, snap3, snap4])
    d.arrows.extend([
        Arrow("a-snap-1-2", [(440, 980), (470, 980)], stroke="#475569"),
        Arrow("a-snap-2-3", [(910, 980), (940, 980)], stroke="#475569"),
        Arrow("a-snap-3-4", [(1320, 980), (1350, 980)], stroke="#475569"),
    ])

    return d


def build_research_match_agent() -> Diagram:
    d = Diagram(
        name="03_research_match_agent",
        title="ResearchMatchAgent — internals & topic search",
        subtitle="Same loop shape as ProfessorLookupAgent, but a different prompt and a single Cypher query over (:Professor)-[:HAS_RESEARCH_INTEREST]->(:ResearchTopic).",
        width=1720,
        height=1140,
    )

    d.boxes.append(Box(
        "prompt",
        "System prompt: recommend professors for a research direction or topic.\n"
        "Rules: call find_professors_by_topics_tool with a short list of concrete keywords before answering.\n"
        "Handoff: if the user asks about one named professor, call transfer_to_professor_lookup.\n"
        "Style: ground every answer in tool output; if the graph has no match, say so clearly.",
        60, 140, 1600, 120,
        fill="#dcfce7", stroke="#16a34a", font_size=16,
    ))

    # Left column - agent loop
    d.boxes.append(Box(
        "loop-lane", None,
        60, 300, 460, 580,
        fill="#ecfdf5", stroke="#22c55e", dashed=True,
    ))
    d.boxes.append(Box(
        "loop-header", "Compiled subgraph",
        88, 320, 404, 40,
        fill="#bbf7d0", stroke="#15803d", font_size=18,
    ))
    llm = Box("rm-llm", "llm\ncall_model", 110, 400, 160, 80,
              fill="#ecfdf5", stroke="#16a34a", font_size=18)
    tools = Box("rm-tools", "tools\nToolNode", 320, 400, 160, 80,
                fill="#ede9fe", stroke="#7c3aed", font_size=18)
    d.boxes.extend([llm, tools])
    d.boxes.append(Box(
        "rm-bind",
        "model.bind_tools([\n  find_professors_by_topics_tool,\n  transfer_to_professor_lookup,\n])",
        90, 530, 400, 110,
        fill="#ffffff", stroke="#475569", dashed=True, font_size=16,
    ))
    d.boxes.append(Box(
        "rm-emit",
        "Tool calls emit teaching events:\n{ kind: 'tool_call',\n  agent: 'ResearchMatchAgent',\n  tool: <name>, input: {...} }",
        90, 660, 400, 110,
        fill="#fffbeb", stroke="#d97706", dashed=True, font_size=16,
    ))
    d.boxes.append(Box(
        "rm-state",
        "Returns: { messages: [response],\n           active_agent: 'ResearchMatchAgent' }",
        90, 800, 400, 60,
        fill="#f1f5f9", stroke="#64748b", dashed=True, font_size=16,
    ))

    # Center - tools
    d.boxes.append(Box(
        "tools-lane", None,
        555, 300, 480, 580,
        fill="#fefce8", stroke="#eab308", dashed=True,
    ))
    d.boxes.append(Box(
        "tools-header", "Bound tools (2)",
        583, 320, 424, 40,
        fill="#fef9c3", stroke="#ca8a04", font_size=18,
    ))

    t1 = Box(
        "tool-find",
        "@tool find_professors_by_topics_tool(\n  keywords: list[str])\n\n"
        "Goal: rank professors by overlap of their\nresearch topics with the supplied keywords.\n\n"
        "Returns: { matches: [{name, url,\n  matched_nodes, match_count}] }",
        575, 420, 440, 200,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    )
    t2 = Box(
        "tool-handoff",
        "@tool transfer_to_professor_lookup()\n\n"
        "Goal: hand control back to ProfessorLookupAgent\nwhen the user pivots to a single named professor.\n\n"
        "Returns Command(goto=..., graph=PARENT,\n  update={ active_agent:\n    'ProfessorLookupAgent' }).",
        575, 650, 440, 220,
        fill="#fff7ed", stroke="#ea580c", font_size=16,
    )
    d.boxes.extend([t1, t2])

    # Right - Cypher
    d.boxes.append(Box(
        "cypher-lane", None,
        1070, 300, 590, 580,
        fill="#f0fdf4", stroke="#16a34a", dashed=True,
    ))
    d.boxes.append(Box(
        "cypher-header", "Neo4jQueryService — Cypher",
        1098, 320, 534, 40,
        fill="#dcfce7", stroke="#15803d", font_size=18,
    ))

    c1 = Box(
        "cy-find",
        "MATCH (p:Professor)-[:HAS_RESEARCH_INTEREST]->(t:ResearchTopic)\n"
        "WITH p, t,\n"
        "     [k IN $keywords WHERE\n"
        "        toLower(t.name) CONTAINS k\n"
        "        OR toLower(t.normalized_name) CONTAINS k] AS hits\n"
        "WHERE size(hits) > 0\n"
        "RETURN p.name AS professor_name,\n"
        "       p.detail_url AS detail_url,\n"
        "       collect(DISTINCT t.name)[0..5] AS matched_nodes,\n"
        "       count(DISTINCT t)              AS match_count\n"
        "ORDER BY match_count DESC, professor_name\n"
        "LIMIT 5",
        1090, 420, 555, 220,
        fill="#ffffff", stroke="#16a34a", font_size=13,
    )
    c2 = Box(
        "cy-handoff",
        "(no Cypher)\n\n"
        "return Command(\n"
        "  goto='ProfessorLookupAgent',\n"
        "  graph=Command.PARENT,\n"
        "  update={\n"
        "    'messages': [...ToolMessage],\n"
        "    'active_agent': 'ProfessorLookupAgent',\n"
        "  },\n"
        ")",
        1090, 670, 555, 200,
        fill="#ffffff", stroke="#ea580c", font_size=13,
    )
    d.boxes.extend([c1, c2])

    # Bottom example
    d.boxes.append(Box(
        "flow-header", "Example turn — \"Which professors work on computer vision and multimedia content analysis?\"",
        60, 905, 1600, 40,
        fill="#1e293b", stroke="#0f172a", font_size=17, text_color="#ffffff",
    ))
    d.boxes.append(Box(
        "step-1",
        "1. llm extracts keywords:\n['computer vision',\n 'multimedia content']\nemits tool_call:\nfind_professors_by_topics_tool(...)",
        60, 960, 380, 130,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    ))
    d.boxes.append(Box(
        "step-2",
        "2. ToolNode runs Cypher,\nreturns ranked TopicMatch list\nwith matched ResearchTopic nodes",
        470, 960, 380, 130,
        fill="#ecfeff", stroke="#0891b2", font_size=16,
    ))
    d.boxes.append(Box(
        "step-3",
        "3. llm composes a grounded\nrecommendation citing matched\nresearch topics per professor",
        880, 960, 380, 130,
        fill="#ecfdf5", stroke="#16a34a", font_size=16,
    ))
    d.boxes.append(Box(
        "step-4",
        "4. No more tool_calls → END.\nactive_agent = 'ResearchMatchAgent'\nis remembered for the next turn\non this thread_id.",
        1290, 960, 370, 130,
        fill="#fffbeb", stroke="#d97706", font_size=16,
    ))

    d.arrows.extend([
        Arrow("a-llm-tools",
              [llm.right, tools.left],
              stroke="#16a34a", label="tool_calls"),
        Arrow("a-tools-llm",
              [tools.top, (tools.cx, 380), (llm.cx, 380), llm.top],
              stroke="#7c3aed", dashed=True),
        Arrow("a-tn-find",
              [tools.right, (525, tools.cy), (525, t1.cy), t1.left],
              stroke="#0891b2"),
        Arrow("a-tn-handoff",
              [tools.right, (540, tools.cy), (540, t2.cy), t2.left],
              stroke="#ea580c"),
        Arrow("a-find-cy", [t1.right, c1.left], stroke="#16a34a", dashed=True),
        Arrow("a-handoff-cy", [t2.right, c2.left], stroke="#ea580c", dashed=True),
        Arrow("a-step-1-2", [(440, 1025), (470, 1025)], stroke="#475569"),
        Arrow("a-step-2-3", [(850, 1025), (880, 1025)], stroke="#475569"),
        Arrow("a-step-3-4", [(1260, 1025), (1290, 1025)], stroke="#475569"),
    ])

    return d



def main() -> None:
    print("Rendering Lab 3 teaching diagrams...")
    for builder in (
        build_specialist_anatomy,
        build_professor_lookup_agent,
        build_research_match_agent,
        build_swarm_composition,
    ):
        diagram = builder()
        emit(diagram)
    print("Done.")


if __name__ == "__main__":
    main()
