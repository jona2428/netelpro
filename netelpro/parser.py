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
import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from netelpro.ast_nodes import (
    And,
    BoolLit,
    Call,
    Def,
    Defn,
    EffectRow,
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
    ParamType,
    Predicate,
    Program,
    Prove,
    RefType,
    Sorry,
    StrLit,
    Sym,
    TruthTableSpec,
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
    effect_warnings: list[ParseError] = field(default_factory=list)

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


EFFECT_VERBS: tuple[str, ...] = ("read", "write", "delete", "network", "io")


def _extract_effect_clauses(
    forms: list[Form],
    errors: list[ParseError],
    warnings: list[ParseError],
) -> dict[str, tuple[EffectRow, ...]]:
    """Fase 3: extract ': (effects ...)' clauses from top-level defn forms (spec §1.1).

    Splices the clause out of each defn form IN PLACE so every downstream pass
    (collect_defns, refinement prosecution, walk_and_validate, build_node) keeps
    seeing the classic 3-operand defn shape — zero changes elsewhere in the parser.
    Validates D6 (duplicates -> warning), D7 (unknown verb), D8 (empty pattern).
    Returns defn name -> declared rows.
    """
    clauses: dict[str, tuple[EffectRow, ...]] = {}

    def _err(anchor: Tok | Form, msg: str) -> None:
        if isinstance(anchor, Form):
            errors.append(ParseError(anchor.lparen.line, anchor.lparen.col, msg))
        else:
            errors.append(ParseError(anchor.line, anchor.col, msg))

    for form in forms:
        if not form.items or not isinstance(form.items[0], Tok):
            continue
        if form.items[0].value != "defn":
            continue
        # Clause placement: [defn, NAME, PARAMS, COLON, (effects ...), BODY].
        colon_idx: int | None = None
        if len(form.items) >= 6 and isinstance(form.items[3], Tok) and form.items[3].kind == "COLON":
            colon_idx = 3
        if colon_idx is None:
            continue
        colon = form.items[colon_idx]
        if not (isinstance(colon, Tok) and colon.kind == "COLON"):
            continue
        name_tok = form.items[1]
        effects_form = form.items[colon_idx + 1]
        if not (isinstance(name_tok, Tok) and name_tok.kind == "SYMBOL"):
            continue
        if not (
            isinstance(effects_form, Form)
            and effects_form.items
            and isinstance(effects_form.items[0], Tok)
            and effects_form.items[0].value == "effects"
        ):
            _err(effects_form if isinstance(effects_form, Form) else colon,
                 "': (effects ...)' clause expected after parameter list")
            continue
        # Splice COLON + effects form out of the defn form.
        del form.items[colon_idx : colon_idx + 2]
        if len(effects_form.items) < 2:
            _err(effects_form, "'(effects ...)' requires at least one effect row")
            continue
        rows: list[EffectRow] = []
        seen: set[tuple[str, str]] = set()
        for row in effects_form.items[1:]:
            if not (
                isinstance(row, Form)
                and len(row.items) == 2
                and isinstance(row.items[0], Tok)
                and row.items[0].kind == "SYMBOL"
                and isinstance(row.items[1], Tok)
                and row.items[1].kind == "STRING"
            ):
                _err(row if isinstance(row, Form) else effects_form,
                     'effect row must be \'(VERBO "patrón")\' with VERBO a symbol and patrón a string literal')
                continue
            verb, pat = row.items[0].value, row.items[1].value
            if verb not in EFFECT_VERBS:
                _err(row, f"unknown effect verb '{verb}' — valid verbs: {', '.join(EFFECT_VERBS)}")
                continue
            if not pat:
                _err(row, "effect pattern cannot be empty string")
                continue
            if (verb, pat) in seen:
                warnings.append(ParseError(
                    row.lparen.line, row.lparen.col,
                    f"duplicate effect row ({verb} \"{pat}\") — deduplicated (D6)",
                ))
                continue
            seen.add((verb, pat))
            rows.append(EffectRow(verb=verb, pattern=pat, line=row.lparen.line, col=row.lparen.col))
        if rows:
            clauses[name_tok.value] = tuple(rows)
    return clauses


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
    if isinstance(ty, Tok) and ty.kind == "SYMBOL":
        if ty.value == "Bool":
            return ParamType(kind="bool", enum=())
        if ty.value == "Int":
            return ParamType(kind="int", enum=())
        if ty.value == "Evidence":
            return ParamType(kind="evidence", enum=())
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


_REF_OPS: dict[str, str] = {
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "!=": "!=",
    "==": "==",
}


def _refine_predicates(ty: Tok | Form) -> tuple[Predicate, ...] | None:
    """Extract predicates from '(Ref Int P1 P2 ...)' if well-formed, else None."""
    if not (isinstance(ty, Form) and len(ty.items) >= 2):
        return None
    head = ty.items[0]
    base = ty.items[1]
    if not (isinstance(head, Tok) and head.kind == "SYMBOL" and head.value == "Ref"):
        return None
    if not (isinstance(base, Tok) and base.kind == "SYMBOL" and base.value == "Int"):
        return None
    preds: list[Predicate] = []
    for p in ty.items[2:]:
        if not (
            isinstance(p, Form)
            and len(p.items) == 2
            and isinstance(p.items[0], Tok)
            and p.items[0].kind == "SYMBOL"
            and p.items[0].value in _REF_OPS
            and isinstance(p.items[1], Tok)
            and p.items[1].kind == "INT"
        ):
            return None
        op_tok = p.items[0]
        assert isinstance(op_tok, Tok)
        const_tok = p.items[1]
        assert isinstance(const_tok, Tok)
        preds.append(Predicate(op=_REF_OPS[op_tok.value], const=int(const_tok.value)))
    if not preds:
        return None
    return tuple(preds)


def _param_type_of(ty: Tok | Form) -> ParamType | RefType | None:
    """Union type of a param annotation: refinement first, then plain/enum."""
    preds = _refine_predicates(ty)
    if preds is not None:
        return RefType(predicates=preds)
    return _build_param_type(ty)


def _check_type_annotation(
    ty: Tok | Form, errors: list[ParseError]
) -> ParamType | RefType | None:
    """Validate a TYPE annotation, reporting a prosecutorial error if malformed."""
    ref_preds = _refine_predicates(ty)
    if ref_preds is not None:
        return RefType(predicates=ref_preds)
    pt = _build_param_type(ty)
    if pt is None:
        if isinstance(ty, Tok):
            errors.append(
                ParseError(
                    ty.line,
                    ty.col,
                    f"unsupported parameter type {ty.value!r}, expected Bool, Int, (Int <int literals>) or (Ref Int (<op> <int>)...)",
                )
            )
        elif isinstance(ty, Form):
            errors.append(
                ParseError(
                    ty.lparen.line,
                    ty.lparen.col,
                    "unsupported parameter type, expected Bool, Int, (Int <int literals>) or (Ref Int (<op> <int>)...)",
                )
            )
    return pt


def _check_annotated_param(
    form: Form, errors: list[ParseError]
) -> tuple[str, ParamType | RefType | None] | None:
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
    ptypes: list[ParamType | RefType | None],
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
        if isinstance(ptype, RefType):
            continue  # refinements are prosecuted by _prosecute_refinements
        if ptype.kind == "int":
            continue  # plain Int: no domain to check at call-site (spec F2 §1.1)
        if ptype.kind == "evidence":
            # Spec F4 D1: Evidence is opaque and non-fabricable. Literals can
            # never stand in for it (only a bound Evidence param symbol may).
            if arg.kind == "SYMBOL":
                continue  # a bound Evidence symbol flows through: origin preserved
            errors.append(
                ParseError(
                    arg.line,
                    arg.col,
                    f"type-strict call: literal passed to Evidence parameter #{idx + 1} of '{fname}'"
                    " — evidence originates at the FFI boundary, not in-language",
                )
            )
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
    typed_registry: dict[str, list[ParamType | RefType | None]],
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
            pname, pty = res
            if isinstance(pty, RefType):
                errors.append(
                    ParseError(
                        pf.lparen.line,
                        pf.lparen.col,
                        "refinement types not allowed in truth-table parameters (use enum)",
                    )
                )
                continue
            params.append((pname, pty))
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
) -> tuple[dict[str, int], list[ParseError], dict[str, list[ParamType | RefType | None]]]:
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
    typed: dict[str, list[ParamType | RefType | None]] = {}
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
                        _param_type_of(t.items[2]) if isinstance(t, Form) and len(t.items) == 3 else None
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
    if name == "fold":
        if not operands or not (isinstance(operands[0], Tok) and operands[0].kind == "SYMBOL"):
            errors.append(
                ParseError(
                    form.lparen.line,
                    form.lparen.col,
                    "'fold' requires an 'and' or 'or' operator symbol as first operand",
                )
            )
        elif operands[0].value not in ("and", "or"):
            errors.append(
                ParseError(
                    operands[0].line,
                    operands[0].col,
                    f"fold: operator must be 'and' or 'or', found '{operands[0].value}'",
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
                        if isinstance(p, Form) and len(p.items) == 3:
                            _check_type_annotation(p.items[2], errors)

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

    elif name == "prove":
        # Spec F4 D4: prove is only valid inside a defn/fn body. The check_
        # special walk tracks depth via the existing arity walk; nesting rules
        # are enforced structurally here and positionally in check_truth_table.
        if depth <= 1:
            errors.append(
                ParseError(
                    form.lparen.line,
                    form.lparen.col,
                    "'prove' is only valid inside a defn/fn body",
                )
            )
        if len(operands) != 2:
            errors.append(
                ParseError(
                    form.lparen.line,
                    form.lparen.col,
                    "'prove' requires exactly 2 operands: (prove CLAIM (evidence NAME : Evidence))",
                )
            )
        else:
            ev_form = operands[1]
            leaves = _flatten_ev_form(ev_form) if isinstance(ev_form, Form) else None
            if leaves is None:
                errors.append(
                    ParseError(
                        form.lparen.line,
                        form.lparen.col,
                        "evidence form must be a binder '(evidence NAME : Evidence)' "
                        "or a binary '(and ev-form ev-form)' of them",
                    )
                )
            else:
                for _name_tok, binder in leaves:
                    if not (
                        len(binder.items) == 4
                        and isinstance(binder.items[2], Tok)
                        and binder.items[2].kind == "COLON"
                        and isinstance(binder.items[3], Tok)
                        and binder.items[3].kind == "SYMBOL"
                        and binder.items[3].value == "Evidence"
                    ):
                        errors.append(
                            ParseError(
                                form.lparen.line,
                                form.lparen.col,
                                "evidence binder must be (evidence NAME : Evidence)",
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


_NEG_OP: dict[str, str] = {
    "<": ">=",
    ">=": "<",
    ">": "<=",
    "<=": ">",
    "==": "!=",
    "!=": "==",
}


def _flatten_ev_form(form: Form) -> list[tuple[Tok, Form]] | None:
    """F4-v2: flatten a prove evidence form into its binders.

    Either a direct binder '(evidence NAME : Evidence)' or a strictly binary
    '(and ev-form ev-form)' tree.  Returns (name_tok, binder_form) per leaf,
    or None if the shape is invalid.  Composition desugars to nested prove
    forms: claim AND NOT(e1 AND e2) == nested hole condition, so runtime and
    codegen stay untouched and each hole reports its own binder coords.
    Known trade-off: the claim expression is evaluated once per nesting
    level (interpreter and native agree).  Claims are propositions — keep
    them pure; effectful claims would observe repeated evaluation."""
    if isinstance(form, Form) and len(form.items) == 3 and isinstance(form.items[0], Tok):
        head = form.items[0]
        if head.kind == "SYMBOL" and head.value == "and":
            leaves: list[tuple[Tok, Form]] = []
            for sub in form.items[1:]:
                if not isinstance(sub, Form):
                    return None
                sub_leaves = _flatten_ev_form(sub)
                if sub_leaves is None:
                    return None
                leaves.extend(sub_leaves)
            return leaves
    if (
        isinstance(form, Form)
        and len(form.items) == 4
        and isinstance(form.items[0], Tok)
        and form.items[0].kind == "SYMBOL"
        and form.items[0].value == "evidence"
        and isinstance(form.items[1], Tok)
        and form.items[1].kind == "SYMBOL"
    ):
        return [(form.items[1], form)]
    return None

_CMP_FN: dict[str, object] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _cmp_fact(f: Tok | Form) -> tuple[str, str, int] | None:
    """(op, subject, const) if f is '(op SUBJECT const)' with symbol/expr subject."""
    if not (isinstance(f, Form) and len(f.items) == 3):
        return None
    h = f.items[0]
    a = f.items[1]
    b = f.items[2]
    if not (isinstance(h, Tok) and h.kind == "SYMBOL" and h.value in _CMP_FN):
        return None
    if not (isinstance(b, Tok) and b.kind == "INT"):
        return None
    if isinstance(a, Tok) and a.kind == "SYMBOL":
        return (h.value, a.value, int(b.value))
    if isinstance(a, Form) and len(a.items) == 3:
        sub = a.items[0]
        x = a.items[1]
        y = a.items[2]
        if (
            isinstance(sub, Tok)
            and sub.kind == "SYMBOL"
            and sub.value == "-"
            and isinstance(x, Tok)
            and x.kind == "SYMBOL"
            and isinstance(y, Tok)
            and y.kind == "INT"
        ):
            return (h.value, f"(- {x.value} {y.value})", int(b.value))
    return None


def _arg_key(a: Tok | Form) -> str | None:
    """Argument identity for guard lookup: variable name or expression key."""
    if isinstance(a, Tok) and a.kind == "SYMBOL":
        return a.value
    if (
        isinstance(a, Form)
        and len(a.items) == 3
        and isinstance(a.items[0], Tok)
        and a.items[0].kind == "SYMBOL"
        and a.items[0].value == "-"
        and isinstance(a.items[1], Tok)
        and a.items[1].kind == "SYMBOL"
        and isinstance(a.items[2], Tok)
        and a.items[2].kind == "INT"
    ):
        return f"(- {a.items[1].value} {a.items[2].value})"
    return None


def _fact_proves(facts: set[tuple[str, int]], op: str, const: int) -> bool:
    for f_op, f_val in facts:
        if f_op == "lit":
            fn = _CMP_FN[op]
            assert callable(fn)
            if fn(f_val, const):
                return True
        elif _cmp_implies(f_op, f_val, op, const):
            return True
    return False


def _derive_shifted_facts(facts: set[tuple[str, int]], shift: int) -> set[tuple[str, int]]:
    """F2-v2: affine translation of facts about a variable to facts about
    (- var k).  Sound over Z for all six operators: (var OP c) implies
    ((- var k) OP (c - k)).  Linear shift only — no Presburger (spec F2 v2)."""
    return {(op, val - shift) for op, val in facts}


def _cmp_implies(f_op: str, f_val: int, op: str, const: int) -> bool:
    """Does (x f_op f_val) imply (x op const) over the integers? Pure operator
    lattice on constants — no symbolic arithmetic (spec F2 D6/D7 keep)."""
    if op == "==":
        return f_op == "==" and f_val == const
    if op == "!=":
        if f_op == "==":
            return f_val != const
        if f_op == "!=":
            return f_val == const
        return f_op in ("<", ">")  # x < c or x > c proves x != c
    if op == "<":
        if f_op == "<":
            return f_val <= const
        if f_op == "<=":
            return f_val < const
        if f_op == "==":
            return f_val < const  # x == c proves x < c'
        return False
    if op == "<=":
        if f_op in ("<", "<="):
            return f_val <= const
        if f_op == "==":
            return f_val <= const  # x == c proves x <= c'
        return False
    if op == ">":
        if f_op == ">":
            return f_val >= const
        if f_op == ">=":
            return f_val > const
        if f_op == "==":
            return f_val > const  # x == c proves x > c'
        return False
    if op == ">=":
        if f_op in (">", ">="):
            return f_val >= const
        if f_op == "==":
            return f_val >= const  # x == c proves x >= c'
        return False
    return False


def _with_fact(
    guards: dict[str, set[tuple[str, int]]], subject: str, fact: tuple[str, int]
) -> dict[str, set[tuple[str, int]]]:
    g = {k: set(v) for k, v in guards.items()}
    g.setdefault(subject, set()).add(fact)
    return g


def _prosecute_refinements(
    node: Tok | Form,
    guards: dict[str, set[tuple[str, int]]],
    refined: dict[str, list[ParamType | RefType | None]],
    errors: list[ParseError],
) -> None:
    """F2 prosecutor: call args to refined params need proven literal or
    dominating guard (D1/D2). Purely static: erasure total."""
    if isinstance(node, Tok) or not node.items:
        return
    head = node.items[0]
    if not (isinstance(head, Tok) and head.kind == "SYMBOL"):
        return
    name = head.value
    ops = node.items[1:]

    if name == "if" and len(ops) == 3:
        _prosecute_refinements(ops[0], guards, refined, errors)
        fact = _cmp_fact(ops[0])
        if fact is not None:
            f_op, subj, f_const = fact
            g_then = _with_fact(guards, subj, (f_op, f_const))
            g_else = _with_fact(guards, subj, (_NEG_OP[f_op], f_const))
            _prosecute_refinements(ops[1], g_then, refined, errors)
            _prosecute_refinements(ops[2], g_else, refined, errors)
        else:
            _prosecute_refinements(ops[1], guards, refined, errors)
            _prosecute_refinements(ops[2], guards, refined, errors)
        return

    if name == "and" and len(ops) == 2:
        _prosecute_refinements(ops[0], guards, refined, errors)
        fact = _cmp_fact(ops[0])
        g_right = _with_fact(guards, fact[1], (fact[0], fact[2])) if fact else guards
        _prosecute_refinements(ops[1], g_right, refined, errors)
        return

    if name == "or" and len(ops) == 2:
        _prosecute_refinements(ops[0], guards, refined, errors)
        _prosecute_refinements(ops[1], guards, refined, errors)
        return

    if name == "let" and len(ops) == 3:
        _prosecute_refinements(ops[1], guards, refined, errors)
        g_body = {k: set(v) for k, v in guards.items()}
        nm = ops[0]
        if isinstance(nm, Tok) and nm.kind == "SYMBOL":
            g_body.pop(nm.value, None)
            val = ops[1]
            if isinstance(val, Tok) and val.kind == "INT":
                g_body[nm.value] = {("lit", int(val.value))}
            elif isinstance(val, Tok) and val.kind == "SYMBOL" and val.value in g_body:
                g_body[nm.value] = set(g_body[val.value])
        _prosecute_refinements(ops[2], g_body, refined, errors)
        return

    if name == "fn" and ops and isinstance(ops[0], Form):
        pnames = [_annotation_param_name(t) for t in ops[0].items]
        g_fn = {k: set(v) for k, v in guards.items() if k not in pnames}
        for t, pn in zip(ops[0].items, pnames):
            if pn is None or not (isinstance(t, Form) and len(t.items) == 3):
                continue
            preds = _refine_predicates(t.items[2])
            if preds is not None:
                g_fn[pn] = {(p.op, p.const) for p in preds}
        for o in ops[1:]:
            _prosecute_refinements(o, g_fn, refined, errors)
        return

    if name in refined:
        rts = refined[name]
        if len(ops) == len(rts):
            for idx, (rt, arg) in enumerate(zip(rts, ops)):
                if not isinstance(rt, RefType):
                    continue
                if isinstance(arg, Tok) and arg.kind == "INT":
                    lit = int(arg.value)
                    for p in rt.predicates:
                        fn = _CMP_FN[p.op]
                        assert callable(fn)
                        if not fn(lit, p.const):
                            errors.append(
                                ParseError(
                                    arg.line,
                                    arg.col,
                                    f"literal {lit} violates refinement '({p.op} {p.const})' "
                                    f"for parameter #{idx} of '{name}'",
                                )
                            )
                else:
                    key = _arg_key(arg)
                    facts = guards.get(key, set()) if key is not None else set()
                    shift_facts: set[tuple[str, int]] = set()
                    if (
                        key is not None
                        and isinstance(arg, Form)
                        and len(arg.items) == 3
                        and isinstance(arg.items[0], Tok)
                        and arg.items[0].kind == "SYMBOL"
                        and arg.items[0].value == "-"
                        and isinstance(arg.items[1], Tok)
                        and arg.items[1].kind == "SYMBOL"
                        and isinstance(arg.items[2], Tok)
                        and arg.items[2].kind == "INT"
                    ):
                        shift_facts = _derive_shifted_facts(
                            guards.get(arg.items[1].value, set()),
                            int(arg.items[2].value),
                        )
                    for p in rt.predicates:
                        if not _fact_proves(facts | shift_facts, p.op, p.const):
                            if key is None:
                                errors.append(
                                    ParseError(
                                        head.line,
                                        head.col,
                                        f"refinement '({p.op} {p.const})' cannot be proven for a "
                                        f"non-literal argument at call to '{name}' "
                                        "(literal or guarded variable required)",
                                    )
                                )
                            else:
                                errors.append(
                                    ParseError(
                                        head.line,
                                        head.col,
                                        f"refinement predicate '({p.op} {p.const})' not proven "
                                        f"for argument '{key}' at call to '{name}'",
                                    )
                                )

    for o in ops:
        if isinstance(o, Form):
            _prosecute_refinements(o, guards, refined, errors)


def walk_and_validate(
    form: Form,
    heads: dict[str, tuple[Arity, str]],
    user_defns: dict[str, int],
    errors: list[ParseError],
    depth: int,
    typed_registry: dict[str, list[ParamType | RefType | None]] | None = None,
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
        elif name == "prove":
            check_special(name, form, operands, errors, depth, set(heads) | {"truth-table"})
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
    if isinstance(head, Tok) and head.kind == "SYMBOL" and head.value == "prove":
        skip.add(1)  # evidence binder: validated in check_special, never walked

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
                    pt = _param_type_of(p.items[2])
                    if pt is None:
                        return None
                    if isinstance(pt, RefType):
                        pt = None  # erasure (D4): refinement never reaches the AST
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
                    pt = _param_type_of(p.items[2])
                    if pt is None:
                        return None
                    if isinstance(pt, RefType):
                        pt = None  # erasure (D4): refinement never reaches the AST
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

        case "fold":
            if len(operands) < 3 or not isinstance(operands[0], Tok):
                return None
            op = operands[0].value
            if op not in ("and", "or"):
                return None
            built: list[Node] = []
            for o in operands[1:]:
                child = build_node(o)
                if child is None:
                    return None
                built.append(child)
            acc: Node = built[-1]
            for child in reversed(built[:-1]):
                if op == "and":
                    acc = And(l=child, r=acc, line=line, col=col)
                else:
                    acc = Or(l=child, r=acc, line=line, col=col)
            return acc

        case "sorry":
            if len(operands) != 1 or not isinstance(operands[0], Tok):
                return None
            return Sorry(
                reason=StrLit(operands[0].value, line=operands[0].line, col=operands[0].col),
                line=line,
                col=col,
            )

        case "prove":
            if len(operands) != 2:
                return None
            claim = build_node(operands[0])
            if claim is None or not isinstance(operands[1], Form):
                return None
            leaves = _flatten_ev_form(operands[1])
            if leaves is None:
                return None
            node: Prove | None = None
            for name_tok, _binder in leaves:
                node = Prove(
                    claim=node if node is not None else claim,
                    ev_name=name_tok.value,
                    line=name_tok.line,
                    col=name_tok.col,
                )
            return node

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
        effect_warnings: list[ParseError] = []

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

        # Fase 3: extract ': (effects ...)' clauses BEFORE any other pass so the
        # classic defn shape is preserved for every downstream prosecutor.
        effect_clauses = _extract_effect_clauses(forms, errors, effect_warnings)

        user_defns, dup_errors, typed_registry = collect_defns(forms, self.reserved)
        errors.extend(dup_errors)

        refined: dict[str, list[ParamType | RefType | None]] = {}
        for f in forms:
            head = f.items[0] if f.items else None
            if not (isinstance(head, Tok) and head.kind == "SYMBOL" and head.value == "defn"):
                continue
            if len(f.items) != 4 or not isinstance(f.items[2], Form):
                continue
            nm = f.items[1]
            if not (isinstance(nm, Tok) and nm.kind == "SYMBOL" and nm.value not in self.reserved):
                continue
            refined[nm.value] = [
                _param_type_of(t.items[2]) if isinstance(t, Form) and len(t.items) == 3 else None
                for t in f.items[2].items
            ]
        if any(any(r is not None for r in lst) for lst in refined.values()):
            for f in forms:
                if isinstance(f, Form):
                    _prosecute_refinements(f, {}, refined, errors)

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

        # Spec F4 D5/D7: every declared Evidence param must be consumed by a
        # prove form of its defn; prove never appears inside a truth-table (D7)
        # — a Prove node in a desugared (truth-table ...) defn is the violation.
        def _collect_proves(n: Node) -> list[Prove]:
            found: list[Prove] = []
            stack: list[Node] = [n]
            while stack:
                cur = stack.pop()
                if isinstance(cur, Prove):
                    found.append(cur)
                for f2 in ("claim", "cond", "then", "else_", "l", "r", "value", "body"):
                    child = getattr(cur, f2, None)
                    if isinstance(child, Node):
                        stack.append(child)
            return found

        for node in program.forms:
            if isinstance(node, Defn):
                consumed_ev: set[str] = set()
                evidence_params: set[str] = {
                    s.name
                    for s, pt in zip(node.params, node.param_types or ())
                    if pt is not None and pt.kind == "evidence"
                }
                in_table = node.truth_table is not None
                for pv in _collect_proves(node.body):
                    # B2: an Evidence param is not Bool — as claim it is a type
                    # error (D1). Static-complete: params are the only source
                    # of Evidence-typed values in v1.
                    if isinstance(pv.claim, Sym) and pv.claim.name in evidence_params:
                        errors.append(
                            ParseError(
                                pv.claim.line,
                                pv.claim.col,
                                f"claim of 'prove' is Evidence parameter '{pv.claim.name}', not Bool",
                            )
                        )
                    if in_table:
                        errors.append(
                            ParseError(
                                pv.line,
                                pv.col,
                                "'prove' is not valid inside a truth-table body",
                            )
                        )
                    consumed_ev.add(pv.ev_name)
                for pname, pt in zip([s.name for s in node.params], node.param_types or ()):
                    if pt is not None and pt.kind == "evidence" and pname not in consumed_ev:
                        errors.append(
                            ParseError(
                                node.name.line if hasattr(node.name, "line") else 1,
                                node.name.col if hasattr(node.name, "col") else 1,
                                f"Evidence parameter '{pname}' of '{node.name.name}' is never used by a prove form",
                            )
                        )

        # Fase 3: attach extracted effect rows to their Defn nodes (frozen —
        # same object.__setattr__ discipline as __post_init__ coercions).
        for node in program.forms:
            if isinstance(node, Defn):
                rows = effect_clauses.get(node.name.name)
                if rows:
                    object.__setattr__(node, "effects", rows)

        return ParseResult(
            program=program,
            errors=errors,
            defn_registry=user_defns,
            effect_warnings=effect_warnings,
        )


def parse(src_or_toks: str | Sequence[Tok], table_path: str | Path | None = None) -> ParseResult:
    """Convenience functional interface for parsing Netelpro programs."""
    return Parser(table_path=table_path).parse(src_or_toks)
