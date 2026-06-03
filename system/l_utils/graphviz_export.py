from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import json
import logging


logger = logging.getLogger(__name__)


_GV_WARNED = False


def _gv():
    global _GV_WARNED
    try:
        from graphviz import Digraph
        return Digraph
    except Exception:
        if not _GV_WARNED:
            print("Graphviz python package not available.")
            _GV_WARNED = True
        return None


def _render(g, out_path: str, audit=None) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        g.render(out_path, format="png", cleanup=True)
        return out_path + ".png"
    except Exception as e:
        print("Graphviz render failed. Ensure system Graphviz is installed.")
        if audit:
            try:
                audit.record("graphviz", str(e))
            except Exception:
                logger.debug("graphviz_export: audit record failed", exc_info=True)
        return ""


def export_json_fallback(data: dict, out_path: str = "artifacts/audit/fallback.json", audit=None) -> str:
    try:
        if not isinstance(data, dict):
            raise ValueError("fallback data must be dict")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if audit:
            audit.record_json("graphviz_fallback", {"path": out_path})
        return out_path
    except Exception:
        return ""


def export_trace_graph(trace: List[Dict], out_path: str = "artifacts/audit/trace_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"trace": trace}, out_path="artifacts/audit/trace_fallback.json", audit=audit)
    g = Digraph("trace")
    g.attr(rankdir="LR")
    prev = None
    for i, t in enumerate(trace):
        node_id = f"n{i}"
        label = f"{t.get('rule', '')}\\n{t.get('conclusion', '')}"
        g.node(node_id, label=label, shape="box")
        if prev is not None:
            g.edge(prev, node_id)
        prev = node_id
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"trace": trace}, out_path="artifacts/audit/trace_fallback.json", audit=audit)
    return r


def export_execution_graph(plan: Dict, execution: Dict, out_path: str = "artifacts/audit/execution_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"plan": plan, "execution": execution}, out_path="artifacts/audit/execution_fallback.json", audit=audit)
    g = Digraph("execution")
    g.attr(rankdir="LR")
    g.node("start", "start", shape="circle")
    steps = plan.get("steps", [])
    prev = "start"
    for i, step in enumerate(steps):
        status = step.get("status", "")
        label = f"{step.get('detail', 'step')}\\n{status}"
        color = "lightgreen" if status == "done" else ("salmon" if status == "failed" else "lightgrey")
        node_id = f"s{i}"
        g.node(node_id, label=label, shape="box", style="filled", fillcolor=color)
        g.edge(prev, node_id)
        prev = node_id
    g.node("end", "end", shape="doublecircle")
    g.edge(prev, "end")
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"plan": plan, "execution": execution}, out_path="artifacts/audit/execution_fallback.json", audit=audit)
    return r


def export_trace_exec_graph(trace: List[Dict], execution: Dict, out_path: str = "artifacts/audit/trace_exec_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"trace": trace, "execution": execution}, out_path="artifacts/audit/trace_exec_fallback.json", audit=audit)
    g = Digraph("trace_exec")
    g.attr(rankdir="LR")
    prev = None
    for i, t in enumerate(trace):
        tid = f"t{i}"
        g.node(tid, f"{t.get('rule','')}\\n{t.get('conclusion','')}", shape="box")
        if prev is not None:
            g.edge(prev, tid)
        prev = tid
    prev_exec = None
    for i, step in enumerate(execution.get("steps", [])):
        sid = f"s{i}"
        status = step.get("status", "")
        color = "lightgreen" if status == "done" else ("salmon" if status == "failed" else "lightgrey")
        g.node(sid, f"{step.get('detail','')}\\n{status}", shape="box", style="filled", fillcolor=color)
        if prev_exec is not None:
            g.edge(prev_exec, sid, style="dashed")
        prev_exec = sid
    if trace and execution.get("steps"):
        g.edge("t0", "s0", color="blue")
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"trace": trace, "execution": execution}, out_path="artifacts/audit/trace_exec_fallback.json", audit=audit)
    return r


def export_combined_graph(trace: List[Dict], plan: Dict, out_path: str = "artifacts/audit/combined_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"trace": trace, "plan": plan}, out_path="artifacts/audit/combined_fallback.json", audit=audit)
    g = Digraph("combined")
    g.attr(rankdir="LR")
    with g.subgraph(name="cluster_trace") as c:
        c.attr(label="trace")
        prev = None
        for i, t in enumerate(trace):
            node_id = f"t{i}"
            label = f"{t.get('rule', '')}\\n{t.get('conclusion', '')}"
            c.node(node_id, label=label, shape="box")
            if prev is not None:
                c.edge(prev, node_id)
            prev = node_id
    with g.subgraph(name="cluster_plan") as c:
        c.attr(label="plan")
        c.node("goal", plan.get("goal", "goal"), shape="oval")
        levels = plan.get("levels", [])
        prev_level_nodes = ["goal"]
        for li, level in enumerate(levels):
            current_nodes = []
            for si, step in enumerate(level):
                node_id = f"p{li}_{si}"
                label = step.get("detail", "step")
                c.node(node_id, label=label, shape="box")
                current_nodes.append(node_id)
                for p in prev_level_nodes:
                    c.edge(p, node_id)
            prev_level_nodes = current_nodes or prev_level_nodes
    if trace:
        g.edge("t0", "goal")
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"trace": trace, "plan": plan}, out_path="artifacts/audit/combined_fallback.json", audit=audit)
    return r


def export_memory_graph(short_mem: List[str], long_mem: List[str], out_path: str = "artifacts/audit/memory_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"short": short_mem, "long": long_mem}, out_path="artifacts/audit/memory_fallback.json", audit=audit)
    g = Digraph("memory")
    g.attr(rankdir="LR")
    prev = None
    for i, m in enumerate(short_mem):
        node_id = f"s{i}"
        g.node(node_id, m[:30], shape="box", style="filled", fillcolor="lightblue")
        if prev is not None:
            g.edge(prev, node_id)
        prev = node_id
    prev = None
    for i, m in enumerate(long_mem[-10:]):
        node_id = f"l{i}"
        g.node(node_id, m[:30], shape="box", style="filled", fillcolor="lightgrey")
        if prev is not None:
            g.edge(prev, node_id)
        prev = node_id
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"short": short_mem, "long": long_mem}, out_path="artifacts/audit/memory_fallback.json", audit=audit)
    return r


def export_global_graph(trace: List[Dict], plan: Dict, execution: Dict, short_mem: List[str], long_mem: List[str], out_path: str = "artifacts/audit/global_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"trace": trace, "plan": plan, "execution": execution, "memory": {"short": short_mem, "long": long_mem}}, out_path="artifacts/audit/global_fallback.json", audit=audit)
    g = Digraph("global")
    g.attr(rankdir="LR")
    with g.subgraph(name="cluster_trace") as c:
        c.attr(label="trace")
        prev = None
        for i, t in enumerate(trace):
            node_id = f"t{i}"
            label = f"{t.get('rule', '')}\\n{t.get('conclusion', '')}"
            c.node(node_id, label=label, shape="box")
            if prev is not None:
                c.edge(prev, node_id)
            prev = node_id
    with g.subgraph(name="cluster_plan") as c:
        c.attr(label="plan")
        c.node("goal", plan.get("goal", "goal"), shape="oval")
        levels = plan.get("levels", [])
        prev_level_nodes = ["goal"]
        for li, level in enumerate(levels):
            current_nodes = []
            for si, step in enumerate(level):
                node_id = f"p{li}_{si}"
                label = step.get("detail", "step")
                c.node(node_id, label=label, shape="box")
                current_nodes.append(node_id)
                for p in prev_level_nodes:
                    c.edge(p, node_id)
            prev_level_nodes = current_nodes or prev_level_nodes
    with g.subgraph(name="cluster_exec") as c:
        c.attr(label="execution")
        c.node("start", "start", shape="circle")
        prev = "start"
        for i, step in enumerate(execution.get("steps", [])):
            status = step.get("status", "")
            label = f"{step.get('detail', 'step')}\\n{status}"
            node_id = f"e{i}"
            c.node(node_id, label=label, shape="box")
            c.edge(prev, node_id)
            prev = node_id
        c.node("end", "end", shape="doublecircle")
        c.edge(prev, "end")
    with g.subgraph(name="cluster_mem") as c:
        c.attr(label="memory")
        prev = None
        for i, m in enumerate(short_mem):
            node_id = f"ms{i}"
            c.node(node_id, m[:20], shape="box")
            if prev is not None:
                c.edge(prev, node_id)
            prev = node_id
        prev = None
        for i, m in enumerate(long_mem[-5:]):
            node_id = f"ml{i}"
            c.node(node_id, m[:20], shape="box")
            if prev is not None:
                c.edge(prev, node_id)
            prev = node_id
    if trace:
        g.edge("t0", "goal")
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"trace": trace, "plan": plan, "execution": execution, "memory": {"short": short_mem, "long": long_mem}}, out_path="artifacts/audit/global_fallback.json", audit=audit)
    return r


def export_plan_graph(plan: Dict, out_path: str = "artifacts/audit/plan_graph", audit=None) -> str:
    Digraph = _gv()
    if Digraph is None:
        return export_json_fallback({"plan": plan}, out_path="artifacts/audit/plan_fallback.json", audit=audit)
    g = Digraph("plan")
    g.attr(rankdir="TB")
    goal = plan.get("goal", "goal")
    g.node("goal", goal, shape="oval")
    levels = plan.get("levels", [])
    prev_level_nodes = ["goal"]
    for li, level in enumerate(levels):
        current_nodes = []
        for si, step in enumerate(level):
            node_id = f"l{li}_s{si}"
            label = step.get("detail", "step")
            status = step.get("status", "")
            color = "lightgreen" if status == "done" else ("salmon" if status == "failed" else "lightgrey")
            g.node(node_id, label=f"{label}\\n{status}", shape="box", style="filled", fillcolor=color)
            current_nodes.append(node_id)
            for p in prev_level_nodes:
                g.edge(p, node_id)
        prev_level_nodes = current_nodes or prev_level_nodes
    r = _render(g, out_path, audit=audit)
    if not r:
        return export_json_fallback({"plan": plan}, out_path="artifacts/audit/plan_fallback.json", audit=audit)
    return r
