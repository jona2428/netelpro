"""Netelpro effect inference and gate purity enforcement.

Static effect system:
1. Derives capability requirements from spec/arity_table.json via
   netelpro.caps._derive_capabilities_from_table (reusing the single source of truth).
2. Computes the transitive effect set for each user-defined function (defn)
   using monotonic fixpoint iteration over the static call graph (capped at 100 iterations).
3. Attributes effects of nested anonymous functions (Fn) to their enclosing Defn,
   since Netelpro v0.1 does not support first-class function calls.
4. Enforces gate rule purity: any 'filter-rule' defn must be completely pure.
   Impure gate rules report actionable diagnostics containing the exact call chain
   to the offending effectful primitive.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from netelpro.ast_nodes import (
    And,
    Call,
    Def,
    Defn,
    Fn,
    If,
    Let,
    ListLit,
    Node,
    Or,
    Program,
    Sorry,
    Sym,
)
from netelpro.caps import _derive_capabilities_from_table


@dataclass(frozen=True)
class EffectError(Exception):
    """Effect diagnostic record with exact source coordinates."""

    message: str
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"line {self.line}, col {self.col}: {self.message}"


def _collect_defn_info(
    defn_node: Defn,
    known_defns: set[str],
    cap_reqs: dict[str, set[str]],
) -> tuple[set[str], list[str], list[str]]:
    """Traverse defn body and collect direct capabilities, primitive calls, and callee defns.

    Decision on nested anonymous functions (Fn):
    Anonymous functions (Fn) nested inside a Defn contribute their effects to the enclosing Defn.
    Because Netelpro v0.1 does not support first-class dynamic function calls, any effect occurring
    within an anonymous function body is attributed statically to the enclosing top-level definition.

    Returns:
        (direct_effects, direct_primitive_calls, direct_defn_calls)
    """
    direct_effects: set[str] = set()
    direct_primitive_calls: list[str] = []
    direct_defn_calls: list[str] = []

    def walk(node: Node) -> None:
        if isinstance(node, Call):
            head = node.head
            if head in cap_reqs:
                direct_effects.update(cap_reqs[head])
                direct_primitive_calls.append(head)
            elif head in known_defns:
                direct_defn_calls.append(head)
            for arg in node.args:
                walk(arg)
        elif isinstance(node, (Defn, Fn)):
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
        elif isinstance(node, ListLit):
            for item in node.items:
                walk(item)
        elif isinstance(node, Def):
            walk(node.value)
        elif isinstance(node, Sorry):
            walk(node.reason)
        else:
            pass

    walk(defn_node.body)
    return direct_effects, direct_primitive_calls, direct_defn_calls


def infer_effects(
    program: Program,
    *,
    _max_iterations: int = 100,
) -> dict[str, frozenset[str]]:
    """Infer static effect sets (capabilities) for all defns in program.

    Effects propagate transitively through the static call graph.
    Cycles (direct and mutual recursion) are resolved via fixpoint iteration.
    Raises EffectError if inference does not converge within max iterations.
    """
    if isinstance(program, Program):
        forms = program.forms
    elif isinstance(program, Node):
        forms = [program]
    else:
        forms = []

    defns: dict[str, Defn] = {}
    for form in forms:
        if isinstance(form, Defn):
            name = form.name.name if isinstance(form.name, Sym) else str(form.name)
            defns[name] = form

    if not defns:
        return {}

    _, cap_reqs = _derive_capabilities_from_table()
    known_defns = set(defns.keys())

    direct_effects: dict[str, set[str]] = {}
    callees: dict[str, set[str]] = {}

    for name, defn_node in defns.items():
        effs, _, defn_calls = _collect_defn_info(defn_node, known_defns, cap_reqs)
        direct_effects[name] = effs
        callees[name] = set(defn_calls)

    current_effects: dict[str, set[str]] = {name: set(direct_effects[name]) for name in defns}

    converged = False
    for _ in range(_max_iterations):
        changed = False
        for name in defns:
            for callee in callees[name]:
                new_effs = current_effects[callee] - current_effects[name]
                if new_effs:
                    current_effects[name].update(new_effs)
                    changed = True
        if not changed:
            converged = True
            break

    if not converged:
        raise EffectError("effect inference did not converge")

    return {name: frozenset(current_effects[name]) for name in defns}


def _find_call_chain(
    start_defn: str,
    direct_primitives: dict[str, list[str]],
    direct_defn_calls: dict[str, list[str]],
    effects: dict[str, frozenset[str]],
) -> list[str]:
    """Find the shortest call chain from start_defn to an effect-inducing primitive.

    Uses BFS over the static call graph, pruning pure branches.
    """
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(start_defn, (start_defn,))])
    visited: set[str] = {start_defn}

    while queue:
        curr, path = queue.popleft()
        prims = direct_primitives.get(curr, [])
        if prims:
            return [*path, prims[0]]

        for callee in direct_defn_calls.get(curr, []):
            if effects.get(callee) and callee not in visited:
                visited.add(callee)
                queue.append((callee, (*path, callee)))

    return [start_defn]


def check_gate_purity(
    program: Program,
    gate_rule_name: str = "filter-rule",
) -> list[EffectError]:
    """Verify that gate rule definition is completely pure.

    Returns a list of EffectError diagnostics if gate rule has any effects.
    The diagnostic message details the exact call chain leading to the effect.
    """
    if isinstance(program, Program):
        forms = program.forms
    elif isinstance(program, Node):
        forms = [program]
    else:
        forms = []

    defns: dict[str, Defn] = {}
    for form in forms:
        if isinstance(form, Defn):
            name = form.name.name if isinstance(form.name, Sym) else str(form.name)
            defns[name] = form

    if gate_rule_name not in defns:
        return []

    effects = infer_effects(program)
    rule_effects = effects.get(gate_rule_name, frozenset())
    if not rule_effects:
        return []

    _, cap_reqs = _derive_capabilities_from_table()
    known_defns = set(defns.keys())

    direct_prims: dict[str, list[str]] = {}
    direct_calls: dict[str, list[str]] = {}

    for name, defn_node in defns.items():
        _, prims, calls = _collect_defn_info(defn_node, known_defns, cap_reqs)
        direct_prims[name] = prims
        direct_calls[name] = calls

    chain = _find_call_chain(gate_rule_name, direct_prims, direct_calls, effects)
    chain_str = " -> ".join(chain)
    effects_str = ", ".join(sorted(rule_effects))

    gate_defn = defns[gate_rule_name]
    line = gate_defn.line if gate_defn.line > 0 else getattr(gate_defn.name, "line", 0)
    col = gate_defn.col if gate_defn.col > 0 else getattr(gate_defn.name, "col", 0)

    msg = f"gate rule '{gate_rule_name}' must be pure (effects: {effects_str} via <chain: {chain_str}>)"
    return [EffectError(message=msg, line=line, col=col)]


def _primitive_effect_reqs() -> dict[str, list[tuple[str, str]]]:
    """Derive primitive effect requirements from arity_table.json (spec D3).

    Reuses the single source of truth (caps._derive_capabilities_from_table):
    each required capability string 'c' becomes the effect row (c "*") — the
    wildcard pattern means 'any pattern of this verb'. Future IO primitives
    (read-file, write-file, ...) will declare richer rows in the table.
    """
    _, cap_reqs = _derive_capabilities_from_table()
    return {head: [(c, "*") for c in sorted(caps)] for head, caps in cap_reqs.items()}


def _collect_calls(
    node: Node,
    prim_calls: list[tuple[str, int, int]],
    defn_calls: list[tuple[str, int, int]],
    known_defns: set[str],
) -> None:
    """Collect direct primitive and user-defn calls with exact coordinates."""
    if isinstance(node, Call):
        if node.head in known_defns:
            defn_calls.append((node.head, node.line, node.col))
        else:
            prim_calls.append((node.head, node.line, node.col))
        for arg in node.args:
            _collect_calls(arg, prim_calls, defn_calls, known_defns)
    elif isinstance(node, (Defn, Fn)):
        _collect_calls(node.body, prim_calls, defn_calls, known_defns)
    elif isinstance(node, Let):
        _collect_calls(node.value, prim_calls, defn_calls, known_defns)
        _collect_calls(node.body, prim_calls, defn_calls, known_defns)
    elif isinstance(node, If):
        _collect_calls(node.cond, prim_calls, defn_calls, known_defns)
        _collect_calls(node.then, prim_calls, defn_calls, known_defns)
        _collect_calls(node.else_, prim_calls, defn_calls, known_defns)
    elif isinstance(node, (And, Or)):
        _collect_calls(node.l, prim_calls, defn_calls, known_defns)
        _collect_calls(node.r, prim_calls, defn_calls, known_defns)
    elif isinstance(node, ListLit):
        for item in node.items:
            _collect_calls(item, prim_calls, defn_calls, known_defns)
    elif isinstance(node, Def):
        _collect_calls(node.value, prim_calls, defn_calls, known_defns)
    elif isinstance(node, Sorry):
        _collect_calls(node.reason, prim_calls, defn_calls, known_defns)


def _pattern_covers(declared: str, needed: str) -> bool:
    """F3-v2: does a declared row pattern cover a needed one?

    Glob semantics (conservative): '*' matches any run of characters that
    does NOT cross a '/' (path-segment scope, zone_rule_generator
    vocabulary).  A non-glob pattern covers only its literal self.
    Conservative on purpose: under-coverage fails closed (compile error),
    never the reverse."""
    if declared == needed:
        return True
    if declared.count("*") != 1:
        return False
    head, _, tail = declared.partition("*")
    if len(needed) < len(head) + len(tail):
        return False
    if not needed.startswith(head) or not needed.endswith(tail):
        return False
    middle = needed[len(head) : len(needed) - len(tail)]
    return "/" not in middle


def check_effect_rows(program: Program | Node) -> list[EffectError]:
    """Fase 3: verify declared effect rows cover every inferred effect (spec §2).

    Composition (D2): for every direct call edge caller -> callee, every declared
    row of the callee must appear LITERALLY in the caller's rows. Literal row-set
    subset checks are transitive, so chained and mutual calls need no fixpoint.
    Primitives (D3): required capability 'c' is the row (c "*"), satisfied by any
    declared row of the same verb. Top-level forms (D11) have no declaration
    site: calling an effectful defn from them is an error (legacy grants still
    own top-level io via caps.py — D10 coexistence).
    """
    forms: list[Node]
    if isinstance(program, Program):
        forms = list(program.forms)
    else:
        forms = [program]

    defns: dict[str, Defn] = {}
    for form in forms:
        if isinstance(form, Defn):
            name = form.name.name if isinstance(form.name, Sym) else str(form.name)
            defns[name] = form

    prim_reqs = _primitive_effect_reqs()
    errors: list[EffectError] = []

    for name, defn_node in defns.items():
        declared = {(r.verb, r.pattern) for r in defn_node.effects}
        declared_verbs = {v for (v, _) in declared}

        for r in defn_node.effects:
            if r.pattern.count("*") > 1:
                errors.append(
                    EffectError(
                        message=(
                            f"effect pattern '{r.pattern}' in '{name}' supports "
                            f"at most one '*'"
                        ),
                        line=r.line,
                        col=r.col,
                    )
                )

        prim_calls: list[tuple[str, int, int]] = []
        defn_calls: list[tuple[str, int, int]] = []
        _collect_calls(defn_node.body, prim_calls, defn_calls, set(defns))

        for head, line, col in prim_calls:
            for verb, _pat in prim_reqs.get(head, []):
                if verb not in declared_verbs:
                    msg = (
                        f"call to '{head}' requires effect ({verb} \"*\") not declared "
                        f"in function '{name}' — add : (effects ({verb} \"*\"))"
                    )
                    errors.append(EffectError(message=msg, line=line, col=col))

        for callee, line, col in defn_calls:
            callee_rows = {(r.verb, r.pattern) for r in defns[callee].effects}
            missing = [
                (v, p)
                for (v, p) in sorted(callee_rows)
                if not any(
                    dv == v and _pattern_covers(dp, p) for (dv, dp) in declared
                )
            ]
            if not missing:
                continue
            missing_str = " ".join(f'({v} "{p}")' for v, p in sorted(missing))
            if not declared:
                msg = (
                    f"call to '{callee}' brings effects {missing_str} "
                    f"but caller '{name}' declares none"
                )
            else:
                have_str = " ".join(
                    f'({r.verb} "{r.pattern}")' for r in defn_node.effects
                )
                msg = (
                    f"call to '{callee}' brings effects {missing_str} "
                    f"but caller '{name}' declares only {have_str} — "
                    f"add the missing effect rows"
                )
            errors.append(EffectError(message=msg, line=line, col=col))

    # Top-level (D11): no declaration site — effectful defn calls are rejected.
    for form in forms:
        if isinstance(form, Defn):
            continue
        top_prims: list[tuple[str, int, int]] = []
        top_calls: list[tuple[str, int, int]] = []
        _collect_calls(form, top_prims, top_calls, set(defns))
        for callee, line, col in top_calls:
            if defns[callee].effects:
                rows_str = " ".join(
                    f'({r.verb} "{r.pattern}")' for r in defns[callee].effects
                )
                msg = (
                    f"top-level call to '{callee}' brings effects {rows_str} — "
                    f"wrap it in a defn that declares : (effects {rows_str})"
                )
                errors.append(EffectError(message=msg, line=line, col=col))

    return errors
