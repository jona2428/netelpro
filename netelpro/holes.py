"""Straylight typed holes & static definedness enforcement — Phase 4 prosecutor.

Compiler-as-prosecutor, Phase 4 (verified against the parser fiscal 2026-09-05):
1. THE PARSER is the primary silent-hole prosecutor: any call head that is not a
   primitive/special form or a top-level defn is a PARSE error ("unknown head 'X'"),
   including def-values, fn/defn params, and let bindings (no first-class calls, §9.1).
2. (sorry "reason") is the ONLY legal unimplemented path: compiles clean, is collected
   into a holes manifest (line, col, reason), raises StrayHoleError at runtime if executed.
   This pass IS the manifest collector — its primary v0.1 job.
3. Forward refs & mutual recursion among top-level defns are legal (parser registry-based).
4. Duplicate top-level defn = PARSE error ("duplicate defn 'f' (already defined at top level)").
   This pass keeps the check as defense-in-depth for direct-API consumers.
5. The lexical scope stack below is future-proofing: once first-class heads are legal
   (later phase), check_holes must resolve fn-param and let-bound heads. Today no
   parse-clean program can reach those branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from netelpro.ast_nodes import (
    And,
    BoolLit,
    Call,
    Def,
    Defn,
    FloatLit,
    Fn,
    Grant,
    If,
    IntLit,
    Let,
    ListLit,
    NilLit,
    Node,
    Or,
    Program,
    Sorry,
    StrLit,
    Sym,
)
from netelpro.evaluator import PRIMITIVES


@dataclass(frozen=True)
class HoleError:
    """Prosecutorial hole diagnostic record with exact source coordinates."""

    message: str
    line: int = 0
    col: int = 0

    def __str__(self) -> str:
        return f"line {self.line}, col {self.col}: {self.message}"


def collect_declarations(program: Program | Node | Sequence[Node]) -> dict[str, int]:
    """Collect name -> line of FIRST declaration, from TOP-LEVEL forms only.

    Parser law (fiscal): def/defn are only valid at top level; nested ones are
    rejected at parse time. So declaration collection is a top-level walk.
    """
    declarations: dict[str, int] = {}

    top_forms: list[Node]
    if isinstance(program, Program):
        top_forms = list(program.forms)
    elif isinstance(program, (list, tuple)):
        top_forms = [f for f in program if isinstance(f, Node)]
    elif isinstance(program, Node):
        top_forms = [program]
    else:
        top_forms = []

    for form in top_forms:
        if isinstance(form, (Def, Defn)):
            name = form.name.name if isinstance(form.name, Sym) else str(form.name)
            if name and name not in declarations:
                declarations[name] = form.line or getattr(form.name, "line", 0)
    return declarations


def check_holes(
    program: Program | Node | Sequence[Node],
    primitives: Optional[set[str]] = None,
) -> tuple[list[HoleError], list[dict[str, Any]]]:
    """Statically audit callable heads, duplicates, and sorry holes.

    Returns (errors, manifest):
    - errors: all HoleError records, ordered by (line, col).
    - manifest: all (sorry ...) holes as {'line', 'col', 'reason'} dicts.
    """
    prims = PRIMITIVES if primitives is None else set(primitives)
    declarations = collect_declarations(program)

    errors: list[HoleError] = []
    manifest: list[dict[str, Any]] = []

    # 1. Duplicate TOP-LEVEL def/defn of the same name.
    top_forms: list[Node]
    if isinstance(program, Program):
        top_forms = list(program.forms)
    elif isinstance(program, (list, tuple)):
        top_forms = [f for f in program if isinstance(f, Node)]
    elif isinstance(program, Node):
        top_forms = [program]
    else:
        top_forms = []

    top_level_seen: dict[str, int] = {}
    for form in top_forms:
        if isinstance(form, (Def, Defn)):
            name = form.name.name if isinstance(form.name, Sym) else str(form.name)
            if not name:
                continue
            f_line = form.line or getattr(form.name, "line", 0)
            f_col = form.col or getattr(form.name, "col", 0)
            if name in top_level_seen:
                errors.append(
                    HoleError(
                        message=f"'{name}' is already declared at line {top_level_seen[name]}",
                        line=f_line,
                        col=f_col,
                    )
                )
            else:
                top_level_seen[name] = f_line

    # 2. Head resolution with lexical scope stack (fn params + let bindings).
    def walk(node: Node, scope: list[set[str]]) -> None:
        if isinstance(node, Program):
            for form in node.forms:
                walk(form, [])
        elif isinstance(node, Def):
            walk(node.value, scope)
        elif isinstance(node, Defn):
            params = {p.name for p in node.params if isinstance(p, Sym)}
            walk(node.body, scope + [params])
        elif isinstance(node, Fn):
            params = {p.name for p in node.params}
            walk(node.body, scope + [params])
        elif isinstance(node, Let):
            walk(node.value, scope)
            let_name = node.name.name if isinstance(node.name, Sym) else ""
            walk(node.body, scope + [{let_name}] if let_name else scope)
        elif isinstance(node, If):
            walk(node.cond, scope)
            walk(node.then, scope)
            walk(node.else_, scope)
        elif isinstance(node, (And, Or)):
            walk(node.l, scope)
            walk(node.r, scope)
        elif isinstance(node, ListLit):
            for item in node.items:
                walk(item, scope)
        elif isinstance(node, Sorry):
            reason_node = node.reason
            if isinstance(reason_node, StrLit):
                reason_str = reason_node.value
            elif hasattr(reason_node, "value"):
                reason_str = str(reason_node.value)
            else:
                reason_str = str(reason_node)
            manifest.append(
                {
                    "line": node.line or getattr(reason_node, "line", 0),
                    "col": node.col or getattr(reason_node, "col", 0),
                    "reason": reason_str,
                }
            )
        elif isinstance(node, Call):
            head = node.head.name if isinstance(node.head, Sym) else str(node.head)
            in_scope = any(head in s for s in scope)
            if head not in prims and head not in declarations and not in_scope:
                errors.append(
                    HoleError(
                        message=(
                            f"undefined function '{head}' — declare it with "
                            f"(defn {head} ...) or admit the hole with (sorry ...)"
                        ),
                        line=node.line,
                        col=node.col,
                    )
                )
            for arg in node.args:
                walk(arg, scope)
        else:
            # Literals (IntLit, FloatLit, StrLit, BoolLit, NilLit, Sym leaf, Grant): nothing to check.
            pass

    if isinstance(program, Program):
        for form in program.forms:
            walk(form, [])
    elif isinstance(program, (list, tuple)):
        for item in program:
            if isinstance(item, Node):
                walk(item, [])
    elif isinstance(program, Node):
        walk(program, [])

    errors.sort(key=lambda e: (e.line, e.col))
    manifest.sort(key=lambda m: (m["line"], m["col"]))
    return errors, manifest


__all__ = [
    "HoleError",
    "check_holes",
    "collect_declarations",
]