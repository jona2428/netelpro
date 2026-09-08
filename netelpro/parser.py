"""Netelpro parser -- Phase 1 recursive-descent parser and AST constructor.

The prosecutor thesis:
Netelpro programs are validated mechanically against a single source of truth:
spec/arity_table.json. Every form (special forms, primitives, and user defns)
declares a fixed arity (with 'list' as the sole open-arity exception).

The parser serves as the structural prosecutor:
1. It counts parentheses and checks matching balance with exact source provenance.
2. It collects all user `defn` declarations upfront so forward references and recursive
   calls are verified mechanically without semantic analysis.
3. It audits every operand count against declared bounds with exact line/col provenance.
4. It enforces structural invariants:
   - def/defn/let target must be a SYMBOL.
   - defn/fn params must be a parenthesized group of SYMBOLs.
   - sorry argument must be a STRING literal.
   - grant is allowed ONLY at top level with SYMBOL operands.
   - reserved heads (special forms and primitives) cannot be redefined via def/defn.
   - unknown head is rejected.
5. It never fails fast on the first syntax error: all errors across the entire translation
   unit are accumulated in a ParseResult.
6. When well-formed, it maps the token stream into a frozen, typed abstract syntax tree
   (ast_nodes.py) preserving source provenance on every node.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
    ParamType,
    TruthTableSpec,
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
from netelpro.lexer import LexError, Tok, tokenize

TABLE_PATH = Path(__file__).resolve().parent.parent / "spec" / "arity_table.json"


@dataclass(frozen=True)
class Arity:
    """Declared arity bounds for a head symbol."""

    lo: int
    hi: int | None  # None = open arity, closed only by ')'


@dataclass(frozen=True)
class ParseError:
    """A frozen diagnostic record representing a prosecutorial syntax violation.

    The compiler-as-prosecutor thesis requires exact source provenance on every
    error so mechanical syntax verification can report the precise offense.
    """

    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}, col {self.col}: {self.message}"


@dataclass
class ParseResult:
    """The outcome of parsing a Netelpro translation unit.

    Carries the reconstructed AST (Program) and any accumulated prosecutorial errors.
    Exposes defn_registry mapping user function names to declared parameter counts.
    """

    program: Program
    errors: list[ParseError] = field(default_factory=list)
    defn_registry: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if parsing succeeded with zero prosecutorial errors."""
        return len(self.errors) == 0

    @property
    def defns(self) -> dict[str, int]:
        """Convenience alias for defn_registry."""
        return self.defn_registry


@dataclass
class Form:
    """Intermediate nested S-expression form used for structural auditing."""

    lparen: Tok
    items: list[Tok | Form] = field(default_factory=list)
    rparen: Tok | None = None


def load_table(path: str | Path = TABLE_PATH) -> dict[str, tuple[Arity, str]]:
    """Load arities and head groupings from arity_table.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    heads: dict[str, tuple[Arity, str]] = {}
    for group in ("special_forms", "primitives"):
        for name, item in data[group].items():
            lo, hi = item["arity"]
            heads[name] = (Arity(lo, hi), group)
    return heads


def expected_msg(a: Arity) -> str:
    """Format expected operand count matching tools/check_arity.py style."""
    if a.hi is None:
        return f"at least {a.lo}"
    if a.hi == a.lo:
        return str(a.lo)
    return f"{a.lo} to {a.hi}"


def _annotation_param_name(p: Tok | Form) -> str | None:
    """Parameter name of a plain symbol or an annotated '(name : TYPE)' param."""
    if isinstance(p, Tok) and p.kind == "SYMBOL":
        return p.value
    if (
        isinstance(p, Form)
        and len(p.items) == 3
        and isinstance(p.items[0], Tok)
        and p.items[0].kind == "SYMBOL"
        and isinstance(p.items[1], Tok)
        and p.items[1].kind == "COLON"
    ):
        return p.items[0].value
    return None


def _build_param_type(ty: Tok | Form) -> ParamType | None:
    """Construct a ParamType from a syntactically validated TYPE node, or None."""
    if isinstance(ty, Tok) and ty.kind == "SYMBOL" and ty.value == "Bool":
        return ParamType(kind="bool", enum=())
    if (
        isinstance(ty, Form)
        and ty.items
        and isinstance(ty.items[0], Tok)
        and ty.items[0].kind == "SYMBOL"
        and ty.items[0].value == "Int"
    ):
        lits = ty.items[1:]
        if not lits or not all(isinstance(t, Tok) and t.kind == "INT" for t in lits):
            return None
        return ParamType(kind="int_enum", enum=tuple(int(t.value) for t in lits if isinstance(t, Tok)))
    return None


def _check_type_annotation(ty: Tok | Form, errors: list[ParseError]) -> ParamType | None:
    """Validate a TYPE annotation, reporting a prosecutorial error if malformed."""
    pt = _build_param_type(ty)
    if pt is None:
        if isinstance(ty, Tok):
            errors.append(
                ParseError(
                    ty.line,
                    ty.col,
                    f"unsupported parameter type {ty.value!r}, expected Bool or (Int <int literals>)",
                )
            )
        elif isinstance(ty, Form):
            errors.append(
                ParseError(
                    ty.lparen.line,
                    ty.lparen.col,
                    "unsupported parameter type, expected Bool or (Int <int literals>)",
                )
            )
    return pt


def _check_annotated_param(
    form: Form, errors: list[ParseError]
) -> tuple[str, ParamType | None] | None:
    """Validate one annotated parameter '(name : TYPE)'. Returns (name, type)."""
    if (
        len(form.items) != 3
        or not (isinstance(form.items[0], Tok) and form.items[0].kind == "SYMBOL")
        or not (isinstance(form.items[1], Tok) and form.items[1].kind == "COLON")
    ):
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                "malformed parameter annotation, expected (name : TYPE)",
            )
        )
        return None
    name_tok = form.items[0]
    assert isinstance(name_tok, Tok)
    ty_item = form.items[2]
    assert isinstance(ty_item, (Tok, Form))
    ptype = _check_type_annotation(ty_item, errors)
    return (name_tok.value, ptype)


def _slot_value(t: Tok) -> bool | int | None:
    """Slot value of a literal slot token; None is also the wildcard marker."""
    if t.kind == "BOOL":
        return t.value == "true"
    if t.kind == "INT":
        return int(t.value)
    return None


def _fmt_slot(v: bool | int) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _check_call_site_literals(
    fname: str,
    operands: list[Tok | Form],
    ptypes: list[ParamType | None],
    errors: list[ParseError],
) -> None:
    """Type-strict literal checking at call sites of annotated functions (spec §1.1).

    Only literal arguments are checked in v1: annotations are declarations, not
    refinements. Non-literal arguments (variable SYMBOLs, nested forms) compile
    freely; the v0.6 bool/int strictness is pinned for literal arguments only.
    """
    for idx, (ptype, arg) in enumerate(zip(ptypes, operands)):
        if ptype is None or not isinstance(arg, Tok):
            continue
        if arg.kind == "BOOL":
            if ptype.kind != "bool":
                errors.append(
                    ParseError(
                        arg.line,
                        arg.col,
                        f"type-strict call: boolean literal passed to Int parameter #{idx + 1} of '{fname}'",
                    )
                )
        elif arg.kind == "INT":
            if ptype.kind == "bool":
                errors.append(
                    ParseError(
                        arg.line,
                        arg.col,
                        f"type-strict call: integer literal {arg.value} passed to Bool parameter #{idx + 1} of '{fname}'",
                    )
                )
            elif int(arg.value) not in set(ptype.enum):
                enum_s = ", ".join(str(x) for x in ptype.enum)
                errors.append(
                    ParseError(
                        arg.line,
                        arg.col,
                        f"literal {arg.value} outside declared enumeration ({enum_s}) of parameter #{idx + 1} of '{fname}'",
                    )
                )
        elif arg.kind in ("FLOAT", "STRING", "NIL"):
            errors.append(
                ParseError(
                    arg.line,
                    arg.col,
                    f"type-strict call: {arg.kind.lower()} literal passed to typed parameter #{idx + 1} of '{fname}'",
                )
            )


def _slot_condition(
    params: list[tuple[str, ParamType]],
    slots: tuple[bool | int | None, ...],
    line: int,
    col: int,
) -> Node:
    """Build the equality conjunction for one row's non-wildcard slots.

    Uses only existing primitives (==, and) so interpreter and LLVM backends
    execute the desugared body identically (spec §3.1: parity by construction).
    A row with no concrete slots (legal overlap) yields the literal true.
    """
    cond: Node | None = None
    for (pname, ptype), slot in zip(params, slots):
        if slot is None:
            continue
        lit: Node
        if ptype.kind == "bool":
            lit = BoolLit(bool(slot), line=line, col=col)
        else:
            lit = IntLit(int(slot), line=line, col=col)
        cmp_ = Call(head="==", args=[Sym(pname, line=line, col=col), lit], line=line, col=col)
        cond = cmp_ if cond is None else And(l=cond, r=cmp_, line=line, col=col)
    if cond is None:
        return BoolLit(True, line=line, col=col)
    return cond


def _partition_tt(
    operands: list[Tok | Form],
) -> tuple[Tok, list[Form], list[Form]] | None:
    """Partition truth-table operands into (NAME, PARAM forms, ROW forms).

    Discriminator: a PARAM form starts with a SYMBOL token; a ROW form starts
    with a Form (the slots group). Params must all precede rows. Returns None
    on any structural violation.
    """
    if not operands or not (isinstance(operands[0], Tok) and operands[0].kind == "SYMBOL"):
        return None
    name_tok = operands[0]
    assert isinstance(name_tok, Tok)
    params: list[Form] = []
    rows: list[Form] = []
    for op in operands[1:]:
        if isinstance(op, Form) and op.items and isinstance(op.items[0], Form):
            rows.append(op)
        elif isinstance(op, Form) and op.items and isinstance(op.items[0], Tok) and not rows:
            params.append(op)
        else:
            return None
    return (name_tok, params, rows)


def check_truth_table(
    form: Form,
    operands: list[Tok | Form],
    errors: list[ParseError],
    heads: dict[str, tuple[Arity, str]],
    user_defns: dict[str, int],
    typed_registry: dict[str, list[ParamType | None]],
    depth: int,
) -> None:
    """Prosecutor for the truth-table special form (spec §1.2, §2).

    Enforces: top-level only; symbol name not reserved; params are annotated
    and finite; rows are ((SLOT+) -> EXPR) with slot count == param count;
    slots are literals type-strict against the declared param domain; the last
    row is the all-wildcard default (D1/B4); literal row results are uniform;
    the declared product is <= 256 and fully covered by the non-default rows
    (B5). Row EXPR forms are walked with the standard structural validator.
    """
    if depth != 1:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                "'truth-table' is only valid at top level, not nested",
            )
        )
    if not operands or not (isinstance(operands[0], Tok) and operands[0].kind == "SYMBOL"):
        errors.append(
            ParseError(form.lparen.line, form.lparen.col, "'truth-table' requires a symbol name")
        )
        return
    name_tok = operands[0]
    assert isinstance(name_tok, Tok)
    if name_tok.value in heads:
        errors.append(
            ParseError(
                name_tok.line,
                name_tok.col,
                f"'{name_tok.value}' is a reserved head and cannot be redefined with 'truth-table'",
            )
        )
        return
    part = _partition_tt(operands)
    if part is None:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                "'truth-table' operands must be (name : TYPE) params followed by ((slots) -> expr) rows",
            )
        )
        return
    _, param_forms, row_forms = part
    if not param_forms:
        errors.append(
            ParseError(form.lparen.line, form.lparen.col, "'truth-table' requires at least one typed parameter")
        )
        return
    if not row_forms:
        errors.append(
            ParseError(form.lparen.line, form.lparen.col, "'truth-table' requires at least one row")
        )
        return

    params: list[tuple[str, ParamType | None]] = []
    for pf in param_forms:
        res = _check_annotated_param(pf, errors)
        if res is not None:
            params.append(res)
    if len(params) != len(param_forms):
        return  # malformed params already reported; nothing sound to check further
    n = len(params)
    typed_registry[name_tok.value] = [pt for _, pt in params]

    rows_ok: list[tuple[tuple[bool | int | None, ...], Tok | Form]] = []
    struct_rows_ok = True
    for rf in row_forms:
        assert isinstance(rf, Form)
        if (
            len(rf.items) != 3
            or not isinstance(rf.items[0], Form)
            or not (isinstance(rf.items[1], Tok) and rf.items[1].kind == "ARROW")
        ):
            errors.append(
                ParseError(rf.lparen.line, rf.lparen.col, "malformed row, expected ((SLOT+) -> EXPR)")
            )
            struct_rows_ok = False
            continue
        slots_form = rf.items[0]
        assert isinstance(slots_form, Form)
        if len(slots_form.items) != n:
            errors.append(
                ParseError(
                    slots_form.lparen.line,
                    slots_form.lparen.col,
                    f"row has {len(slots_form.items)} slot(s), the table declares {n} parameter(s)",
                )
            )
            struct_rows_ok = False
            continue
        slots: list[bool | int | None] = []
        row_ok = True
        for (pname, ptype), st in zip(params, slots_form.items):
            if isinstance(st, Tok) and st.kind == "SYMBOL" and st.value == "_":
                slots.append(None)
                continue
            if not isinstance(st, Tok):
                errors.append(
                    ParseError(
                        st.lparen.line if isinstance(st, Form) else 0,
                        st.lparen.col if isinstance(st, Form) else 0,
                        "slots must be literals or '_'",
                    )
                )
                row_ok = False
                continue
            assert ptype is not None
            if ptype.kind == "bool":
                if st.kind == "BOOL":
                    slots.append(st.value == "true")
                elif st.kind == "INT":
                    errors.append(
                        ParseError(
                            st.line,
                            st.col,
                            f"type-strict slot: integer literal {st.value!r} in Bool slot of parameter '{pname}'",
                        )
                    )
                    row_ok = False
                else:
                    errors.append(
                        ParseError(st.line, st.col, f"slot for '{pname}' must be true, false or '_'")
                    )
                    row_ok = False
            else:
                if st.kind == "INT":
                    v = int(st.value)
                    if v not in set(ptype.enum):
                        enum_s = ", ".join(str(x) for x in ptype.enum)
                        errors.append(
                            ParseError(
                                st.line,
                                st.col,
                                f"literal {v} outside declared enumeration ({enum_s}) of parameter '{pname}'",
                            )
                        )
                        row_ok = False
                    slots.append(v)
                elif st.kind == "BOOL":
                    errors.append(
                        ParseError(
                            st.line,
                            st.col,
                            f"type-strict slot: boolean literal in Int slot of parameter '{pname}'",
                        )
                    )
                    row_ok = False
                else:
                    errors.append(
                        ParseError(
                            st.line,
                            st.col,
                            f"slot for '{pname}' must be an Int literal from the declared enumeration or '_'",
                        )
                    )
                    row_ok = False
        if row_ok:
            rows_ok.append((tuple(slots), rf.items[2]))
        else:
            struct_rows_ok = False

    # B4: mandatory all-wildcard last row
    last_rf = row_forms[-1]
    assert isinstance(last_rf, Form)
    default_ok = False
    if (
        len(last_rf.items) == 3
        and isinstance(last_rf.items[0], Form)
        and len(last_rf.items[0].items) == n
        and all(
            isinstance(t, Tok) and t.kind == "SYMBOL" and t.value == "_"
            for t in last_rf.items[0].items
        )
    ):
        default_ok = True
    else:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                "missing default row: the last row must be the all-wildcard row (_ ... _)",
            )
        )

    # Uniform literal result type across rows (call-expression results are
    # not statically typed in v1 -- labeled strictness gap, spec §1.2)
    kinds: list[str] = []
    for _, expr_item in rows_ok:
        if isinstance(expr_item, Tok):
            if expr_item.kind == "BOOL":
                kinds.append("bool")
            elif expr_item.kind == "INT":
                kinds.append("int")
            elif expr_item.kind == "FLOAT":
                kinds.append("float")
            elif expr_item.kind == "STRING":
                kinds.append("str")
            elif expr_item.kind == "NIL":
                kinds.append("nil")
    if len(set(kinds)) > 1:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                f"truth-table rows have mixed literal result types ({' vs '.join(sorted(set(kinds)))})",
            )
        )

    # Row EXPRs still go through the standard structural walk
    for rf in row_forms:
        assert isinstance(rf, Form)
        if len(rf.items) == 3 and isinstance(rf.items[2], Form):
            expr_form = rf.items[2]
            assert isinstance(expr_form, Form)
            walk_and_validate(expr_form, heads, user_defns, errors, depth + 1)

    if not (default_ok and struct_rows_ok):
        return  # coverage is only meaningful on a structurally sound table

    # B5 / D1: coverage over non-default rows, product cap 256
    domains: list[list[bool | int]] = []
    for _, ptype in params:
        assert ptype is not None
        if ptype.kind == "bool":
            domains.append([True, False])
        else:
            domains.append(list(dict.fromkeys(ptype.enum)))
    total = 1
    for d in domains:
        total *= len(d)
    if total > 256:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                f"declared product {total} exceeds the 256-combination cap; reduce parameter ranges",
            )
        )
        return
    non_default = rows_ok[:-1]
    uncovered: list[tuple[bool | int, ...]] = []
    for combo in itertools.product(*domains):
        if not any(
            all(s is None or s == v for s, v in zip(slots, combo)) for slots, _ in non_default
        ):
            uncovered.append(combo)
    if uncovered:
        shown = uncovered[:8]
        detail = "; ".join("(" + ", ".join(_fmt_slot(v) for v in c) + ")" for c in shown)
        more = f" ... +{len(uncovered) - 8} more" if len(uncovered) > 8 else ""
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                f"truth-table '{name_tok.value}' does not cover the declared product; uncovered: {detail}{more}",
            )
        )


def parse_s_expressions(toks: Sequence[Tok]) -> tuple[list[Form], list[ParseError]]:
    """Group flat token stream into nested Form objects while auditing parens.

    Reports:
    - Stray ')' with no matching '('
    - Tokens outside any form at the top level
    - Unclosed '(' at end of token stream
    """
    errors: list[ParseError] = []
    forms: list[Form] = []
    stack: list[Form] = []

    for t in toks:
        if t.kind == "LPAREN":
            stack.append(Form(lparen=t))
        elif t.kind == "RPAREN":
            if not stack:
                errors.append(ParseError(t.line, t.col, "stray ')' with no matching '('"))
                continue
            f = stack.pop()
            f.rparen = t
            if stack:
                stack[-1].items.append(f)
            else:
                forms.append(f)
        else:
            if not stack:
                errors.append(
                    ParseError(
                        t.line,
                        t.col,
                        f"token {t.value!r} outside any form (top level must be forms)",
                    )
                )
                continue
            stack[-1].items.append(t)

    for f in stack:
        errors.append(
            ParseError(
                f.lparen.line,
                f.lparen.col,
                "'(' opened here is never closed (missing ')')",
            )
        )

    return forms, errors


def collect_defns(
    forms: list[Form], reserved: set[str]
) -> tuple[dict[str, int], list[ParseError], dict[str, list[ParamType | None]]]:
    """Register top-level user defn arities (name -> parameter count).

    Enables mechanical validation of user function calls regardless of
    definition order (forward references permitted in top-level declarations).
    Reserved heads and malformed definitions are excluded from registration.
    Duplicate top-level defns (same name declared twice) are NOT silently
    overwritten in silence: the duplicate is reported as a prosecutorial error
    and the LAST declaration wins the arity so downstream call checks remain
    mechanically deterministic.

    Also registers top-level 'truth-table' definitions (name -> param count) so
    calls to table-defined functions are verified like any defn, and returns a
    typed-registry (name -> list[ParamType | None]) for functions that declare
    parameter annotations, used for call-site literal type-checking.
    """
    out: dict[str, int] = {}
    typed: dict[str, list[ParamType | None]] = {}
    dup_errors: list[ParseError] = []
    for f in forms:
        head = f.items[0] if f.items else None
        if not (isinstance(head, Tok) and head.kind == "SYMBOL"):
            continue
        if (
            len(f.items) == 4
            and head.value == "defn"
            and isinstance(f.items[1], Tok)
            and f.items[1].kind == "SYMBOL"
            and f.items[1].value not in reserved
            and isinstance(f.items[2], Form)
        ):
            param_names = [_annotation_param_name(t) for t in f.items[2].items]
            # Parameter uniqueness is required for valid defn registration
            if all(n is not None for n in param_names) and len(set(param_names)) == len(param_names):
                name_tok = f.items[1]
                assert isinstance(name_tok, Tok)
                if name_tok.value in out:
                    dup_errors.append(
                        ParseError(
                            name_tok.line,
                            name_tok.col,
                            f"duplicate defn '{name_tok.value}' (already defined at top level)",
                        )
                    )
                out[name_tok.value] = len(f.items[2].items)
                if any(isinstance(t, Form) for t in f.items[2].items):
                    typed[name_tok.value] = [
                        _build_param_type(t.items[2])
                        if isinstance(t, Form) and len(t.items) == 3
                        else None
                        for t in f.items[2].items
                    ]
        elif head.value == "truth-table" and len(f.items) >= 2:
            tt_name_tok = f.items[1]
            if (
                isinstance(tt_name_tok, Tok)
                and tt_name_tok.kind == "SYMBOL"
                and tt_name_tok.value not in reserved
            ):
                part = _partition_tt(f.items[1:])
                if part is not None:
                    _, param_forms, _row_forms = part
                    if tt_name_tok.value in out:
                        dup_errors.append(
                            ParseError(
                                tt_name_tok.line,
                                tt_name_tok.col,
                                f"duplicate defn '{tt_name_tok.value}' (already defined at top level)",
                            )
                        )
                    out[tt_name_tok.value] = len(param_forms)
                    typed[tt_name_tok.value] = [
                        _build_param_type(pf.items[2])
                        if isinstance(pf, Form) and len(pf.items) == 3
                        else None
                        for pf in param_forms
                    ]
    return out, dup_errors, typed


def check_special(
    name: str,
    form: Form,
    operands: list[Tok | Form],
    errors: list[ParseError],
    depth: int,
    reserved: set[str],
) -> None:
    """Enforce structural invariants for Netelpro special forms."""
    if name in ("def", "defn") and depth != 1:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                f"'{name}' is only valid at top level, not nested",
            )
        )
        return
    if name in ("def", "defn", "let"):
        if not operands:
            return
        t = operands[0]
        if not (isinstance(t, Tok) and t.kind == "SYMBOL"):
            kind = t.kind if isinstance(t, Tok) else "nested form"
            errors.append(
                ParseError(
                    form.lparen.line,
                    form.lparen.col,
                    f"'{name}' requires a symbol name, found {kind}",
                )
            )
        elif name in ("def", "defn", "let") and t.value in reserved:
            errors.append(
                ParseError(
                    t.line,
                    t.col,
                    f"'{t.value}' is a reserved head and cannot be redefined with '{name}'",
                )
            )

    if name in ("defn", "fn"):
        params_idx = 1 if name == "defn" else 0
        if len(operands) > params_idx:
            params = operands[params_idx]
            if not isinstance(params, Form):
                errors.append(
                    ParseError(
                        form.lparen.line,
                        form.lparen.col,
                        f"'{name}' parameter list must be a parenthesized group",
                    )
                )
            else:
                seen: set[str] = set()
                for p in params.items:
                    pname = _annotation_param_name(p)
                    if pname is None:
                        kind = p.kind if isinstance(p, Tok) else "nested form"
                        errors.append(
                            ParseError(
                                form.lparen.line,
                                form.lparen.col,
                                f"parameters must be symbols or (name : TYPE) annotations, found {kind}",
                            )
                        )
                    elif pname in seen:
                        # Uniqueness of parameter names is strictly enforced in v0.1
                        errors.append(
                            ParseError(
                                p.line if isinstance(p, Tok) else form.lparen.line,
                                p.col if isinstance(p, Tok) else form.lparen.col,
                                f"duplicate parameter '{pname}'",
                            )
                        )
                    else:
                        seen.add(pname)

    elif name == "sorry":
        if operands:
            t = operands[0]
            if not (isinstance(t, Tok) and t.kind == "STRING"):
                errors.append(
                    ParseError(
                        form.lparen.line,
                        form.lparen.col,
                        "'sorry' requires a string literal reason",
                    )
                )

    elif name == "grant":
        if depth != 1:
            errors.append(
                ParseError(
                    form.lparen.line,
                    form.lparen.col,
                    "'grant' is only valid at top level, not nested",
                )
            )
        for op in operands:
            if not (isinstance(op, Tok) and op.kind == "SYMBOL"):
                kind = op.kind if isinstance(op, Tok) else "nested form"
                errors.append(
                    ParseError(
                        form.lparen.line,
                        form.lparen.col,
                        f"'grant' requires symbol operands, found {kind}",
                    )
                )


def walk_and_validate(
    form: Form,
    heads: dict[str, tuple[Arity, str]],
    user_defns: dict[str, int],
    errors: list[ParseError],
    depth: int,
    typed_registry: dict[str, list[ParamType | None]] | None = None,
) -> None:
    """Walk form tree recursively, auditing operand counts and structural rules."""
    if not form.items:
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                "empty form '()' is not valid",
            )
        )
        return

    head = form.items[0]
    operands = form.items[1:]

    if isinstance(head, Form):
        errors.append(
            ParseError(
                form.lparen.line,
                form.lparen.col,
                "form head must be a symbol, not a nested form (no first-class calls in v0.1)",
            )
        )
    elif head.kind != "SYMBOL":
        errors.append(
            ParseError(
                head.line,
                head.col,
                f"form head must be a symbol, found {head.kind} {head.value!r}",
            )
        )
    else:
        name = head.value
        got = len(operands)
        if name in heads:
            arity, _group = heads[name]
            if got < arity.lo or (arity.hi is not None and got > arity.hi):
                errors.append(
                    ParseError(
                        head.line,
                        head.col,
                        f"'{name}' expects {expected_msg(arity)} operand(s), found {got}",
                    )
                )
            else:
                check_special(name, form, operands, errors, depth, set(heads) | {"truth-table"})
        elif name in user_defns:
            if got != user_defns[name]:
                errors.append(
                    ParseError(
                        head.line,
                        head.col,
                        f"'{name}' expects {user_defns[name]} operand(s) (declared by defn), found {got}",
                    )
                )
            else:
                pt_list = (typed_registry or {}).get(name)
                if pt_list is not None:
                    _check_call_site_literals(name, operands, pt_list, errors)
        elif name == "truth-table":
            check_truth_table(form, operands, errors, heads, user_defns, typed_registry or {}, depth)
        else:
            errors.append(
                ParseError(
                    head.line,
                    head.col,
                    f"unknown head '{name}' (not in the arity table and not a declared defn)",
                )
            )

    # Parameter-list operands of fn/defn are groups of symbols, not call forms
    skip: set[int] = set()
    if isinstance(head, Tok) and head.kind == "SYMBOL" and head.value in ("fn", "defn"):
        skip.add(1 if head.value == "defn" else 0)
    if isinstance(head, Tok) and head.kind == "SYMBOL" and head.value == "truth-table":
        skip.update(range(len(operands)))

    for idx, it in enumerate(operands):
        if isinstance(it, Form) and idx not in skip:
            walk_and_validate(it, heads, user_defns, errors, depth + 1, typed_registry)


def build_node(item: Tok | Form) -> Node | None:
    """Recursively construct a frozen AST Node from a validated Tok or Form."""
    if isinstance(item, Tok):
        match item.kind:
            case "INT":
                return IntLit(int(item.value), line=item.line, col=item.col)
            case "FLOAT":
                return FloatLit(float(item.value), line=item.line, col=item.col)
            case "STRING":
                return StrLit(item.value, line=item.line, col=item.col)
            case "BOOL":
                return BoolLit(item.value == "true", line=item.line, col=item.col)
            case "NIL":
                return NilLit(line=item.line, col=item.col)
            case "SYMBOL":
                return Sym(item.value, line=item.line, col=item.col)
            case _:
                return None

    if not isinstance(item, Form) or not item.items:
        return None

    head = item.items[0]
    if not isinstance(head, Tok) or head.kind != "SYMBOL":
        return None

    name = head.value
    operands = item.items[1:]
    line, col = item.lparen.line, item.lparen.col

    match name:
        case "def":
            if len(operands) != 2 or not isinstance(operands[0], Tok):
                return None
            val = build_node(operands[1])
            if val is None:
                return None
            return Def(
                name=Sym(operands[0].value, line=operands[0].line, col=operands[0].col),
                value=val,
                line=line,
                col=col,
            )

        case "defn":
            if len(operands) != 3 or not isinstance(operands[0], Tok) or not isinstance(operands[1], Form):
                return None
            params: list[Sym] = []
            ptypes: list[ParamType | None] = []
            for p in operands[1].items:
                pname = _annotation_param_name(p)
                if pname is None:
                    return None
                if isinstance(p, Form):
                    pt = _build_param_type(p.items[2])
                    if pt is None:
                        return None
                    ptypes.append(pt)
                    name_tok_p = p.items[0]
                    assert isinstance(name_tok_p, Tok)
                    params.append(Sym(pname, line=name_tok_p.line, col=name_tok_p.col))
                else:
                    assert isinstance(p, Tok)
                    ptypes.append(None)
                    params.append(Sym(pname, line=p.line, col=p.col))
            body = build_node(operands[2])
            if body is None:
                return None
            return Defn(
                name=Sym(operands[0].value, line=operands[0].line, col=operands[0].col),
                params=params,
                body=body,
                param_types=tuple(ptypes) if any(t is not None for t in ptypes) else None,
                line=line,
                col=col,
            )

        case "fn":
            if len(operands) != 2 or not isinstance(operands[0], Form):
                return None
            params = []
            ptypes_fn: list[ParamType | None] = []
            for p in operands[0].items:
                pname = _annotation_param_name(p)
                if pname is None:
                    return None
                if isinstance(p, Form):
                    pt = _build_param_type(p.items[2])
                    if pt is None:
                        return None
                    ptypes_fn.append(pt)
                    name_tok_p = p.items[0]
                    assert isinstance(name_tok_p, Tok)
                    params.append(Sym(pname, line=name_tok_p.line, col=name_tok_p.col))
                else:
                    assert isinstance(p, Tok)
                    ptypes_fn.append(None)
                    params.append(Sym(pname, line=p.line, col=p.col))
            body = build_node(operands[1])
            if body is None:
                return None
            return Fn(
                params=params,
                body=body,
                param_types=tuple(ptypes_fn) if any(t is not None for t in ptypes_fn) else None,
                line=line,
                col=col,
            )

        case "truth-table":
            part = _partition_tt(operands)
            if part is None:
                return None
            tt_name, param_forms, row_forms = part
            if not param_forms or not row_forms:
                return None
            tparams: list[tuple[str, ParamType]] = []
            for pf in param_forms:
                if not (
                    isinstance(pf, Form)
                    and len(pf.items) == 3
                    and isinstance(pf.items[0], Tok)
                    and isinstance(pf.items[1], Tok)
                    and pf.items[1].kind == "COLON"
                ):
                    return None
                pt = _build_param_type(pf.items[2])
                if pt is None:
                    return None
                pn = pf.items[0]
                assert isinstance(pn, Tok)
                tparams.append((pn.value, pt))
            n = len(tparams)
            rows_meta: list[tuple[tuple[bool | int | None, ...], Node]] = []
            for rf in row_forms:
                if not (
                    isinstance(rf, Form)
                    and len(rf.items) == 3
                    and isinstance(rf.items[0], Form)
                    and isinstance(rf.items[1], Tok)
                    and rf.items[1].kind == "ARROW"
                ):
                    return None
                slots_form = rf.items[0]
                assert isinstance(slots_form, Form)
                if len(slots_form.items) != n:
                    return None
                slots: list[bool | int | None] = []
                for st in slots_form.items:
                    if not isinstance(st, Tok):
                        return None
                    if st.kind == "BOOL":
                        slots.append(st.value == "true")
                    elif st.kind == "INT":
                        slots.append(int(st.value))
                    elif st.kind == "SYMBOL" and st.value == "_":
                        slots.append(None)
                    else:
                        return None
                expr_node = build_node(rf.items[2])
                if expr_node is None:
                    return None
                rows_meta.append((tuple(slots), expr_node))
            default_slots, default_expr = rows_meta[-1]
            if any(s is not None for s in default_slots):
                return None
            # Desugar bottom-up: body starts at the default row's expression and
            # each non-default row (reverse order) becomes one if-level. Only
            # existing primitives (if, ==, and) are used, so the reference
            # interpreter and the LLVM backend execute identical structure.
            body = default_expr
            for slots_r, expr_r in reversed(rows_meta[:-1]):
                cond = _slot_condition(tparams, slots_r, line, col)
                body = If(cond=cond, then=expr_r, else_=body, line=line, col=col)
            tt_spec = TruthTableSpec(
                params=tuple(tparams),
                rows=tuple(rows_meta[:-1]),
                default_expr=default_expr,
            )
            return Defn(
                name=Sym(tt_name.value, line=tt_name.line, col=tt_name.col),
                params=[Sym(nm, line=tt_name.line, col=tt_name.col) for nm, _ in tparams],
                body=body,
                param_types=tuple(pt for _, pt in tparams),
                truth_table=tt_spec,
                line=line,
                col=col,
            )

        case "let":
            if len(operands) != 3 or not isinstance(operands[0], Tok):
                return None
            val = build_node(operands[1])
            body = build_node(operands[2])
            if val is None or body is None:
                return None
            return Let(
                name=Sym(operands[0].value, line=operands[0].line, col=operands[0].col),
                value=val,
                body=body,
                line=line,
                col=col,
            )

        case "if":
            if len(operands) != 3:
                return None
            if_cond = build_node(operands[0])
            if_then = build_node(operands[1])
            if_else = build_node(operands[2])
            if if_cond is None or if_then is None or if_else is None:
                return None
            return If(cond=if_cond, then=if_then, else_=if_else, line=line, col=col)

        case "and":
            if len(operands) != 2:
                return None
            l = build_node(operands[0])
            r = build_node(operands[1])
            if l is None or r is None:
                return None
            return And(l=l, r=r, line=line, col=col)

        case "or":
            if len(operands) != 2:
                return None
            l = build_node(operands[0])
            r = build_node(operands[1])
            if l is None or r is None:
                return None
            return Or(l=l, r=r, line=line, col=col)

        case "sorry":
            if len(operands) != 1 or not isinstance(operands[0], Tok):
                return None
            return Sorry(
                reason=StrLit(operands[0].value, line=operands[0].line, col=operands[0].col),
                line=line,
                col=col,
            )

        case "grant":
            caps: list[Sym] = []
            for op in operands:
                if not isinstance(op, Tok) or op.kind != "SYMBOL":
                    return None
                caps.append(Sym(op.value, line=op.line, col=op.col))
            return Grant(caps=caps, line=line, col=col)

        case "list":
            items: list[Node] = []
            for op in operands:
                list_item = build_node(op)
                if list_item is None:
                    return None
                items.append(list_item)
            return ListLit(items=items, line=line, col=col)

        case _:
            # All primitives and user function calls become Call nodes
            args: list[Node] = []
            for op in operands:
                call_arg = build_node(op)
                if call_arg is None:
                    return None
                args.append(call_arg)
            return Call(head=name, args=args, line=line, col=col)


class Parser:
    """Prosecutorial recursive-descent parser for Netelpro source text."""

    def __init__(self, table_path: str | Path | None = None) -> None:
        self.table_path = Path(table_path) if table_path else TABLE_PATH
        self.heads = load_table(self.table_path)
        self.reserved = set(self.heads) | {"truth-table"}

    def parse(self, src_or_toks: str | Sequence[Tok]) -> ParseResult:
        """Parse source text or token sequence into a ParseResult.

        Accumulates all prosecutorial errors across the translation unit.
        Returns a ParseResult containing the AST Program, errors, and defn_registry.
        """
        errors: list[ParseError] = []

        if isinstance(src_or_toks, str):
            try:
                toks = tokenize(src_or_toks)
            except LexError as e:
                errors.append(ParseError(e.line, e.col, e.message))
                return ParseResult(Program([], line=1, col=1), errors=errors, defn_registry={})
        else:
            toks = list(src_or_toks)

        forms, paren_errors = parse_s_expressions(toks)
        errors.extend(paren_errors)

        user_defns, dup_errors, typed_registry = collect_defns(forms, self.reserved)
        errors.extend(dup_errors)

        for f in forms:
            walk_and_validate(f, self.heads, user_defns, errors, depth=1, typed_registry=typed_registry)

        program_nodes: list[Node] = []
        for f in forms:
            node = build_node(f)
            if node is not None:
                program_nodes.append(node)

        first_line = forms[0].lparen.line if forms else 1
        first_col = forms[0].lparen.col if forms else 1
        program = Program(forms=program_nodes, line=first_line, col=first_col)

        return ParseResult(program=program, errors=errors, defn_registry=user_defns)


def parse(src_or_toks: str | Sequence[Tok], table_path: str | Path | None = None) -> ParseResult:
    """Convenience functional interface for parsing Netelpro programs."""
    return Parser(table_path=table_path).parse(src_or_toks)
