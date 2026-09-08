"""Truth-table builder (spec 2026-09-08-truth-table-range-impl-spec.md §4; step 5 of Phase 1).

Generates, from the TruthTableSpec retained on a Defn plus the .sl source it
came from:

1. A standalone pure-Python fallback module (zero netelpro imports) that
   reproduces the table semantics: declared-order first-match rows under
   type-strict conditions, plus the mandatory final default as the closing
   return. Bool slots compile to ``is True`` / ``is False`` (identity, so the
   bool/int conflation is structurally impossible in the artifact); Int-enum
   slots compile to ``(type(x) is int) and (x == k)``.

2. A self-contained pytest matrix asserting the fallback reproduces the
   interpreter oracle over the FULL declared domain (wildcards expanded over
   the declared domains, product capped at 256 -- same discipline as the
   exhaustiveness prosecutor) plus out-of-domain probes for Int enums
   (first non-enum integer -> must hit the default).

The .sl source is the single source of truth: every expected value is
captured at build time by the real reference interpreter (first-match-wins
included), so the generated artifacts are derivations and never hand-edited.
This closes the bug class of the hand-written gate fallback (2026-09-07
aliasing incident).

Rendered expression subset (closed and honest): IntLit/FloatLit/StrLit/
BoolLit/NilLit, parameter Syms, If, And, Or, and Calls to a fixed operator /
callable map. Anything else raises BuilderError naming the node type or head
-- nothing renders silently.
"""
from __future__ import annotations

import hashlib
import itertools
import keyword

from netelpro.ast_nodes import (
    And,
    BoolLit,
    Call,
    Defn,
    FloatLit,
    If,
    IntLit,
    NilLit,
    Node,
    Or,
    ParamType,
    Program,
    StrLit,
    Sym,
    TruthTableSpec,
)

MAX_PRODUCT = 256

# Binary operators rendered as Python infix (arity checked at render time).
_INFIX: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
}
# Primitives rendered as plain Python calls with the same name.
_CALLABLE_UNARY: frozenset[str] = frozenset({"not", "abs", "len", "str", "int", "float"})
_CALLABLE_NARY: frozenset[str] = frozenset({"min", "max"})


class BuilderError(Exception):
    """A truth-table defn cannot be rendered into generated artifacts."""


# ---------------------------------------------------------------------------
# Python expression rendering (closed subset)
# ---------------------------------------------------------------------------


def _py_ident(name: str) -> str:
    """Deterministic Python identifier rendering of a Netelpro symbol.

    Netelpro allows kebab-case (``filter-rule``); Python does not. The mapping
    is ``-`` -> ``_`` and nothing else; anything that would not be a valid
    identifier after that is a BuilderError, never a silent mangle.
    """
    cand = name.replace("-", "_")
    if not cand.isidentifier() or keyword.iskeyword(cand):
        raise BuilderError(
            f"netelpro symbol {name!r} has no deterministic Python identifier rendering"
        )
    return cand


def _slot_condition(name: str, ptype: ParamType, slot: bool | int | None) -> str:
    if slot is None:
        return ""
    if ptype.kind == "bool":
        # Identity, not equality: 1 == True in Python, 1 is True is False.
        return f"({name} is {bool(slot)})"
    if ptype.kind == "int_enum":
        return f"((type({name}) is int) and ({name} == {int(slot)}))"
    raise BuilderError(f"unknown ParamType kind {ptype.kind!r}")


def _row_condition(
    spec: TruthTableSpec,
    slots: tuple[bool | int | None, ...],
    names_map: dict[str, str],
) -> str:
    parts: list[str] = []
    for (name, ptype), slot in zip(spec.params, slots):
        cond = _slot_condition(names_map[name], ptype, slot)
        if cond:
            parts.append(cond)
    return " and ".join(parts) if parts else "True"


def _render_expr(node: Node, names_map: dict[str, str]) -> str:
    if isinstance(node, IntLit):
        return str(int(node.value))
    if isinstance(node, FloatLit):
        return repr(float(node.value))
    if isinstance(node, StrLit):
        return repr(node.value)
    if isinstance(node, BoolLit):
        return str(bool(node.value))
    if isinstance(node, NilLit):
        return "None"
    if isinstance(node, Sym):
        if node.name not in names_map:
            raise BuilderError(
                f"row body references symbol {node.name!r} which is not a table parameter"
            )
        return names_map[node.name]
    if isinstance(node, If):
        return (
            f"(({_render_expr(node.then, names_map)}) "
            f"if ({_render_expr(node.cond, names_map)}) "
            f"else ({_render_expr(node.else_, names_map)}))"
        )
    if isinstance(node, And):
        return f"(({_render_expr(node.l, names_map)}) and ({_render_expr(node.r, names_map)}))"
    if isinstance(node, Or):
        return f"(({_render_expr(node.l, names_map)}) or ({_render_expr(node.r, names_map)}))"
    if isinstance(node, Call):
        args = [_render_expr(a, names_map) for a in node.args]
        if node.head in _INFIX and len(args) == 2:
            return f"({args[0]} {_INFIX[node.head]} {args[1]})"
        if node.head in _CALLABLE_UNARY and len(args) == 1:
            return f"{node.head}({args[0]})"
        if node.head in _CALLABLE_NARY and len(args) >= 2:
            return f"{node.head}({', '.join(args)})"
        raise BuilderError(
            f"unsupported call {node.head!r}/{len(args)} in truth-table body; "
            f"renderer covers infix {sorted(_INFIX)} and calls "
            f"{sorted(_CALLABLE_UNARY | _CALLABLE_NARY)}"
        )
    raise BuilderError(f"unsupported node type {type(node).__name__} in truth-table body")


# ---------------------------------------------------------------------------
# Fallback module
# ---------------------------------------------------------------------------


def source_sha8(source: str) -> str:
    """Deterministic provenance digest of the .sl source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def render_fallback(defn: Defn, source: str, source_path: str) -> str:
    spec = defn.truth_table
    if spec is None:
        raise BuilderError(
            f"{defn.name.name!r} has no truth_table spec; the builder only accepts "
            "truth-table definitions"
        )
    names_map = {n: _py_ident(n) for n, _ in spec.params}
    if len(set(names_map.values())) != len(names_map):
        raise BuilderError(
            f"parameter names collide after Python sanitization: "
            f"{[n for n, _ in spec.params]}"
        )
    names = tuple(names_map[n] for n, _ in spec.params)
    fn = _py_ident(defn.name.name)
    header = (
        f"# GENERATED FROM {source_path} AT {source_sha8(source)} BY "
        "netelpro.truth_table_builder -- DO NOT EDIT."
    )
    lines: list[str] = [
        header,
        "# Source of truth: the .sl truth-table. Regenerate; never hand-edit.",
        "",
        f"def {fn}({', '.join(names)}):",
    ]
    for slots, expr in spec.rows:
        lines.append(f"    if {_row_condition(spec, slots, names_map)}:")
        lines.append(f"        return {_render_expr(expr, names_map)}")
    lines.append(f"    return {_render_expr(spec.default_expr, names_map)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Build-time oracle (real reference interpreter)
# ---------------------------------------------------------------------------


def _declared_domain(ptype: ParamType) -> tuple[bool, ...] | tuple[int, ...]:
    if ptype.kind == "bool":
        return (True, False)
    if ptype.kind == "int_enum":
        return tuple(ptype.enum)
    raise BuilderError(f"unknown ParamType kind {ptype.kind!r}")


def _arg_node(value: bool | int, ptype: ParamType) -> Node:
    return BoolLit(bool(value)) if ptype.kind == "bool" else IntLit(int(value))


def _probe_value(ptype: ParamType) -> int:
    """First non-enum integer >= 0; exists for any enum under MAX_PRODUCT."""
    v = 0
    while v in ptype.enum:
        v += 1
    return v


def oracle_cases(defn: Defn, source: str) -> dict[tuple[bool | int, ...], object]:
    """Evaluate the real interpreter over the declared domain + OOD probes.

    First-match-wins and out-of-domain -> default are captured by the oracle
    itself: expected values are the semantics the .sl actually has.
    """
    spec = defn.truth_table
    if spec is None:
        raise BuilderError(
            f"{defn.name.name!r} has no truth_table spec; the builder only accepts "
            "truth-table definitions"
        )
    from netelpro.evaluator import Evaluator
    from netelpro.parser import parse

    res = parse(source)
    if res.errors:
        raise BuilderError(f"source does not parse: {res.errors[0]}")
    ev = Evaluator()
    ev.evaluate(res.program)

    domains = [_declared_domain(pt) for _, pt in spec.params]
    combos = list(itertools.product(*domains))
    if len(combos) > MAX_PRODUCT:
        raise BuilderError(
            f"declared domain product {len(combos)} exceeds cap {MAX_PRODUCT}"
        )
    fn = defn.name.name
    cases: dict[tuple[bool | int, ...], object] = {}
    for combo in combos:
        args = [_arg_node(v, pt) for v, (_, pt) in zip(combo, spec.params)]
        cases[combo] = ev.evaluate(Program([Call(fn, args)]))
    # Out-of-domain probes: first non-enum int >= 0, other params at domain min.
    for i, (_, pt) in enumerate(spec.params):
        if pt.kind != "int_enum":
            continue
        probe = _probe_value(pt)
        combo = tuple(
            probe if j == i else _declared_domain(pt2)[0]
            for j, (_, pt2) in enumerate(spec.params)
        )
        args = [_arg_node(v, pt2) for v, (_, pt2) in zip(combo, spec.params)]
        cases[combo] = ev.evaluate(Program([Call(fn, args)]))
    return cases


# ---------------------------------------------------------------------------
# Differential matrix (self-contained pytest, stdlib only)
# ---------------------------------------------------------------------------


def render_matrix(
    defn: Defn, source: str, source_path: str, fallback_name: str
) -> str:
    cases = oracle_cases(defn, source)
    fn = defn.name.name
    fn_ident = _py_ident(fn)
    header = (
        f"# GENERATED FROM {source_path} AT {source_sha8(source)} BY "
        "netelpro.truth_table_builder -- DO NOT EDIT."
    )
    subtitle = (
        f"# Differential matrix: generated fallback vs build-time interpreter "
        f"oracle ({len(cases)} cases)."
    )
    spec_line = (
        f'_SPEC = importlib.util.spec_from_file_location('
        f'"netelpro_tt_generated_fallback", _HERE / {fallback_name!r})'
    )
    lines: list[str] = [
        header,
        subtitle,
        "import importlib.util",
        "import pathlib",
        "",
        f"EXPECTED = {cases!r}",
        "",
        '_HERE = pathlib.Path(__file__).resolve().parent',
        spec_line,
        "_MOD = importlib.util.module_from_spec(_SPEC)",
        "_SPEC.loader.exec_module(_MOD)",
        "",
        "",
        "def test_generated_fallback_matches_build_time_oracle() -> None:",
        "    for args, expected in EXPECTED.items():",
        f"        got = _MOD.{fn_ident}(*args)",
        "        assert got == expected, (args, expected, got)",
    ]
    return "\n".join(lines) + "\n"


def build_truth_table_artifacts(
    defn: Defn,
    source: str,
    source_path: str,
    fallback_name: str | None = None,
) -> tuple[str, str]:
    """Returns (fallback_module_text, matrix_pytest_text)."""
    if fallback_name is None:
        fallback_name = f"tt_{_py_ident(defn.name.name)}_fallback.py"
    base = fallback_name[:-3] if fallback_name.endswith(".py") else fallback_name
    fallback = render_fallback(defn, source, source_path)
    matrix = render_matrix(defn, source, source_path, base + ".py")
    return fallback, matrix