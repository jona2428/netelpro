"""Straylight parser -- Phase 1 recursive-descent parser and AST constructor.

The prosecutor thesis:
Straylight programs are validated mechanically against a single source of truth:
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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from straylight.ast_nodes import (
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
from straylight.lexer import LexError, Lexer, Tok, tokenize

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
    """The outcome of parsing a Straylight translation unit.

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
) -> tuple[dict[str, int], list[ParseError]]:
    """Register top-level user defn arities (name -> parameter count).

    Enables mechanical validation of user function calls regardless of
    definition order (forward references permitted in top-level declarations).
    Reserved heads and malformed definitions are excluded from registration.
    Duplicate top-level defns (same name declared twice) are NOT silently
    overwritten in silence: the duplicate is reported as a prosecutorial error
    and the LAST declaration wins the arity so downstream call checks remain
    mechanically deterministic.
    """
    out: dict[str, int] = {}
    dup_errors: list[ParseError] = []
    for f in forms:
        if (
            len(f.items) == 4
            and isinstance(f.items[0], Tok)
            and f.items[0].value == "defn"
            and isinstance(f.items[1], Tok)
            and f.items[1].kind == "SYMBOL"
            and f.items[1].value not in reserved
            and isinstance(f.items[2], Form)
            and all(isinstance(t, Tok) and t.kind == "SYMBOL" for t in f.items[2].items)
        ):
            param_names = [t.value for t in f.items[2].items if isinstance(t, Tok)]
            # Parameter uniqueness is required for valid defn registration
            if len(param_names) == len(set(param_names)):
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
    return out, dup_errors


def check_special(
    name: str,
    form: Form,
    operands: list[Tok | Form],
    errors: list[ParseError],
    depth: int,
    reserved: set[str],
) -> None:
    """Enforce structural invariants for Straylight special forms."""
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
                    if not (isinstance(p, Tok) and p.kind == "SYMBOL"):
                        kind = p.kind if isinstance(p, Tok) else "nested form"
                        errors.append(
                            ParseError(
                                form.lparen.line,
                                form.lparen.col,
                                f"parameters must be symbols, found {kind}",
                            )
                        )
                    elif p.value in seen:
                        # Uniqueness of parameter names is strictly enforced in v0.1
                        errors.append(
                            ParseError(
                                p.line,
                                p.col,
                                f"duplicate parameter '{p.value}'",
                            )
                        )
                    else:
                        seen.add(p.value)

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
                check_special(name, form, operands, errors, depth, set(heads))
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

    for idx, it in enumerate(operands):
        if isinstance(it, Form) and idx not in skip:
            walk_and_validate(it, heads, user_defns, errors, depth + 1)


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
            for p in operands[1].items:
                if not isinstance(p, Tok) or p.kind != "SYMBOL":
                    return None
                params.append(Sym(p.value, line=p.line, col=p.col))
            body = build_node(operands[2])
            if body is None:
                return None
            return Defn(
                name=Sym(operands[0].value, line=operands[0].line, col=operands[0].col),
                params=params,
                body=body,
                line=line,
                col=col,
            )

        case "fn":
            if len(operands) != 2 or not isinstance(operands[0], Form):
                return None
            params = []
            for p in operands[0].items:
                if not isinstance(p, Tok) or p.kind != "SYMBOL":
                    return None
                params.append(Sym(p.value, line=p.line, col=p.col))
            body = build_node(operands[1])
            if body is None:
                return None
            return Fn(params=params, body=body, line=line, col=col)

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
            cond = build_node(operands[0])
            then = build_node(operands[1])
            else_ = build_node(operands[2])
            if cond is None or then is None or else_ is None:
                return None
            return If(cond=cond, then=then, else_=else_, line=line, col=col)

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
                n = build_node(op)
                if n is None:
                    return None
                items.append(n)
            return ListLit(items=items, line=line, col=col)

        case _:
            # All primitives and user function calls become Call nodes
            args: list[Node] = []
            for op in operands:
                n = build_node(op)
                if n is None:
                    return None
                args.append(n)
            return Call(head=name, args=args, line=line, col=col)


class Parser:
    """Prosecutorial recursive-descent parser for Straylight source text."""

    def __init__(self, table_path: str | Path | None = None) -> None:
        self.table_path = Path(table_path) if table_path else TABLE_PATH
        self.heads = load_table(self.table_path)
        self.reserved = set(self.heads)

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

        user_defns, dup_errors = collect_defns(forms, self.reserved)
        errors.extend(dup_errors)

        for f in forms:
            walk_and_validate(f, self.heads, user_defns, errors, depth=1)

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
    """Convenience functional interface for parsing Straylight programs."""
    return Parser(table_path=table_path).parse(src_or_toks)
