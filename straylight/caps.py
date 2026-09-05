"""Straylight capabilities as types -- Phase 3 static capability enforcement.

The prosecutor thesis applied to effects:
Straylight treats capabilities as a static effect system rather than dynamic permissions.
In v0.1:
1. The capability set is {"io"}. The only primitive requiring a capability is 'print' (requires "io").
2. Enforcement is a SEPARATE static compiler pass -- NOT runtime.
3. Static enforcement is FILE-WIDE: the granted set is the union of all top-level (grant ...) forms.
   Capabilities are not tracked per-function in v0.1 (upgrade path = per-function effect typing).
4. No-first-class-calls enables fully static analysis: Call.head is a plain symbol string.
5. All errors are aggregated across the translation unit with exact source positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import json

from straylight.ast_nodes import (
    And,
    Call,
    Def,
    Defn,
    Fn,
    Grant,
    If,
    Let,
    ListLit,
    Node,
    Or,
    Program,
    Sorry,
    Sym,
)

TABLE_PATH = Path(__file__).resolve().parent.parent / "spec" / "arity_table.json"

KNOWN_CAPABILITIES: set[str] = {"io"}

CAPABILITY_REQUIREMENTS: dict[str, set[str]] = {
    "print": {"io"},
}


def _derive_capabilities_from_table(path: Path = TABLE_PATH) -> tuple[set[str], dict[str, set[str]]]:
    """Derive known capabilities and primitive requirements from spec/arity_table.json."""
    if not path.exists():
        return KNOWN_CAPABILITIES, CAPABILITY_REQUIREMENTS
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = set(data.get("known_capabilities", {}).keys()) or set(KNOWN_CAPABILITIES)
        reqs: dict[str, set[str]] = {}
        for head, prim in data.get("primitives", {}).items():
            if "capabilities" in prim:
                reqs[head] = set(prim["capabilities"])
        return known, reqs
    except Exception:
        return KNOWN_CAPABILITIES, CAPABILITY_REQUIREMENTS


@dataclass(frozen=True)
class CapError:
    """Prosecutorial capability diagnostic record with exact source coordinates."""

    message: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"line {self.line}, col {self.col}: {self.message}"


def collect_grants(program: Program | Node) -> set[str]:
    """Collect all capability names declared via top-level (grant ...) forms."""
    granted: set[str] = set()
    if isinstance(program, Program):
        forms = program.forms
    elif isinstance(program, Grant):
        forms = [program]
    else:
        forms = []

    for form in forms:
        if isinstance(form, Grant):
            for cap in form.caps:
                name = cap.name if isinstance(cap, Sym) else str(cap)
                granted.add(name)
    return granted


def check_capabilities(program: Program | Node, granted: Optional[set[str]] = None) -> list[CapError]:
    """Statically verify that all capability-requiring operations have been granted.

    Recursively walks the AST and aggregates CapError instances for:
    - Any (grant ...) form declaring a capability not in KNOWN_CAPABILITIES.
    - Any Call whose head is in CAPABILITY_REQUIREMENTS and whose required
      capabilities are not a subset of `granted`.

    All violations are aggregated without early termination (parser-style collection).
    """
    if granted is None:
        granted = collect_grants(program)

    errors: list[CapError] = []

    def walk(node: Node) -> None:
        if isinstance(node, Program):
            for form in node.forms:
                walk(form)
        elif isinstance(node, Grant):
            for cap in node.caps:
                cap_name = cap.name if isinstance(cap, Sym) else str(cap)
                if cap_name not in KNOWN_CAPABILITIES:
                    known_str = ", ".join(sorted(KNOWN_CAPABILITIES))
                    c_line = getattr(cap, "line", 0) or getattr(node, "line", 0)
                    c_col = getattr(cap, "col", 0) or getattr(node, "col", 0)
                    errors.append(
                        CapError(
                            message=f"unknown capability '{cap_name}' — known capabilities: {known_str}",
                            line=c_line,
                            col=c_col,
                        )
                    )
        elif isinstance(node, Def):
            walk(node.value)
        elif isinstance(node, Defn):
            for p in node.params:
                walk(p)
            walk(node.body)
        elif isinstance(node, Fn):
            for p in node.params:
                walk(p)
            walk(node.body)
        elif isinstance(node, Let):
            walk(node.value)
            walk(node.body)
        elif isinstance(node, If):
            walk(node.cond)
            walk(node.then)
            walk(node.else_)
        elif isinstance(node, (And, Or)):
            walk(node.l)
            walk(node.r)
        elif isinstance(node, Call):
            if node.head in CAPABILITY_REQUIREMENTS:
                reqs = CAPABILITY_REQUIREMENTS[node.head]
                if not reqs.issubset(granted):
                    missing = sorted(reqs - granted)
                    if len(missing) == 1:
                        m = missing[0]
                        msg = f"capability '{m}' required by '{node.head}' but not granted — add (grant {m}) at top level"
                    else:
                        missing_str = ", ".join(f"'{m}'" for m in missing)
                        grant_str = " ".join(missing)
                        msg = f"capabilities {missing_str} required by '{node.head}' but not granted — add (grant {grant_str}) at top level"
                    errors.append(CapError(message=msg, line=node.line, col=node.col))
            for arg in node.args:
                walk(arg)
        elif isinstance(node, ListLit):
            for item in node.items:
                walk(item)
        elif isinstance(node, Sorry):
            # Sorry semantics untouched (Phase 4 scope)
            walk(node.reason)
        else:
            # Leaf literal / symbol nodes
            pass

    walk(program)
    return errors
