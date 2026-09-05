#!/usr/bin/env python3
"""Netelpro fiscal -- Phase 0 mechanical verifier.

Checks a program's *structure* only: paren balance with exact positions,
operand counts per form against the fixed-arity table, and declared arities
of user `defn`s. No name resolution, no types, no evaluation -- those are
later phases. This is the embryo of the compiler-as-prosecutor: it proves
that "count parentheses and operands against a table" fully determines
whether a program is well-formed, which is the core design claim of
Netelpro's grammar (see docs/SPEC.md).

Usage:
    python3 tools/check_arity.py FILE.sl [FILE2.sl ...] [--table PATH]

Exit code 0 = all files structurally valid; 1 = violations found (each
reported with line/column).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

TABLE_PATH = Path(__file__).resolve().parent.parent / "spec" / "arity_table.json"

DELIMS = " \t\r\n();"
ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
INT_RE = re.compile(r"-?[0-9]+\Z")
FLOAT_RE = re.compile(r"-?[0-9]+\.[0-9]+\Z")
OP_RE = re.compile(r"(\+|-|\*|/|==|!=|<=|>=|<|>)\Z")
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_?>-]*\Z")


class FiscalError(Exception):
    """Fatal scanner error: parsing cannot continue."""


def pos_lc(line: int, col: int) -> str:
    return f"line {line}, col {col}"


@dataclass
class Tok:
    kind: str  # LPAREN RPAREN SYMBOL INT FLOAT STRING BOOL NIL
    value: str
    line: int
    col: int


@dataclass
class Form:
    lparen: Tok
    items: list = field(default_factory=list)  # Tok | Form
    rparen: Tok | None = None


def pos(t: Tok) -> str:
    return pos_lc(t.line, t.col)


def classify(text: str, line: int, col: int) -> str:
    if INT_RE.match(text):
        return "INT"
    if FLOAT_RE.match(text):
        return "FLOAT"
    if text in ("true", "false"):
        return "BOOL"
    if text == "nil":
        return "NIL"
    if OP_RE.match(text) or SYMBOL_RE.match(text):
        return "SYMBOL"
    raise FiscalError(f"{pos_lc(line, col)}: invalid token {text!r}")


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(src)
    line, col = 1, 1

    def advance(ch: str) -> None:
        nonlocal line, col
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1

    while i < n:
        c = src[i]
        if c in " \t\r\n":
            advance(c)
            i += 1
            continue
        if c == ";":
            while i < n and src[i] != "\n":
                advance(src[i])
                i += 1
            continue
        if c in "()":
            toks.append(Tok("LPAREN" if c == "(" else "RPAREN", c, line, col))
            advance(c)
            i += 1
            continue
        if c == '"':
            sl, sc = line, col
            buf: list[str] = []
            i += 1
            advance(c)
            while True:
                if i >= n:
                    raise FiscalError(f"{pos_lc(sl, sc)}: unterminated string literal")
                ch = src[i]
                if ch == "\\":
                    if i + 1 >= n:
                        raise FiscalError(f"{pos_lc(sl, sc)}: unterminated string literal (dangling escape)")
                    esc = src[i + 1]
                    if esc not in ESCAPES:
                        raise FiscalError(f"{pos(line, col) if False else pos_lc(line, col)}: invalid escape \\{esc}")
                    buf.append(ESCAPES[esc])
                    advance("\\")
                    advance(esc)
                    i += 2
                    continue
                if ch == '"':
                    advance(ch)
                    i += 1
                    break
                buf.append(ch)
                advance(ch)
                i += 1
            toks.append(Tok("STRING", "".join(buf), sl, sc))
            continue
        # symbol or number
        sl, sc = line, col
        j = i
        buf2: list[str] = []
        while j < n and src[j] not in DELIMS:
            buf2.append(src[j])
            j += 1
        text = "".join(buf2)
        for ch in text:
            advance(ch)
        i = j
        toks.append(Tok(classify(text, sl, sc), text, sl, sc))
    return toks


def parse_program(toks: list[Tok]) -> tuple[list[Form], list[str]]:
    errors: list[str] = []
    forms: list[Form] = []
    stack: list[Form] = []
    for t in toks:
        if t.kind == "LPAREN":
            stack.append(Form(lparen=t))
        elif t.kind == "RPAREN":
            if not stack:
                errors.append(f"{pos(t)}: stray ')' with no matching '('")
                continue
            f = stack.pop()
            f.rparen = t
            if stack:
                stack[-1].items.append(f)
            else:
                forms.append(f)
        else:
            if not stack:
                errors.append(f"{pos(t)}: token {t.value!r} outside any form (top level must be forms)")
                continue
            stack[-1].items.append(t)
    for f in stack:
        errors.append(f"{pos(f.lparen)}: '(' opened here is never closed (missing ')')")
    return forms, errors


@dataclass
class Arity:
    lo: int
    hi: int | None  # None = open arity, closed by ')'


def load_table(path: str | Path = TABLE_PATH) -> dict[str, tuple[Arity, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    heads: dict[str, tuple[Arity, str]] = {}
    for group in ("special_forms", "primitives"):
        for name, item in data[group].items():
            lo, hi = item["arity"]
            heads[name] = (Arity(lo, hi), group)
    return heads


def collect_defns(forms: list[Form]) -> dict[str, int]:
    """Register top-level defn arities (name -> parameter count) so calls are
    checkable regardless of definition order."""
    out: dict[str, int] = {}
    for f in forms:
        if (
            len(f.items) == 4
            and isinstance(f.items[0], Tok)
            and f.items[0].value == "defn"
            and isinstance(f.items[1], Tok)
            and f.items[1].kind == "SYMBOL"
            and isinstance(f.items[2], Form)
            and all(isinstance(t, Tok) and t.kind == "SYMBOL" for t in f.items[2].items)
        ):
            out[f.items[1].value] = len(f.items[2].items)
    return out


def expected_msg(a: Arity) -> str:
    if a.hi is None:
        return f"at least {a.lo}"
    if a.hi == a.lo:
        return str(a.lo)
    return f"{a.lo} to {a.hi}"


def check_special(
    name: str,
    form: Form,
    operands: list,
    errors: list[str],
    depth: int,
    reserved: set[str],
) -> None:
    if name in ("def", "defn", "let"):
        t = operands[0]
        if not (isinstance(t, Tok) and t.kind == "SYMBOL"):
            kind = t.kind if isinstance(t, Tok) else "nested form"
            errors.append(f"{pos(form.lparen)}: '{name}' requires a symbol name, found {kind}")
        elif name in ("def", "defn") and t.value in reserved:
            errors.append(f"{pos(t)}: '{t.value}' is a reserved head and cannot be redefined with '{name}'")
    if name in ("defn", "fn"):
        params = operands[1] if name == "defn" else operands[0]
        if not isinstance(params, Form):
            errors.append(f"{pos(form.lparen)}: '{name}' parameter list must be a parenthesized group")
        else:
            seen: set[str] = set()
            for p in params.items:
                if not (isinstance(p, Tok) and p.kind == "SYMBOL"):
                    kind = p.kind if isinstance(p, Tok) else "nested form"
                    errors.append(f"{pos(form.lparen)}: parameters must be symbols, found {kind}")
                elif p.value in seen:
                    errors.append(f"{pos(p)}: duplicate parameter '{p.value}'")
                else:
                    seen.add(p.value)
    elif name == "sorry":
        t = operands[0]
        if not (isinstance(t, Tok) and t.kind == "STRING"):
            errors.append(f"{pos(form.lparen)}: 'sorry' requires a string literal reason")
    elif name == "grant":
        if depth != 1:
            errors.append(f"{pos(form.lparen)}: 'grant' is only valid at top level, not nested")


def walk(
    form: Form,
    heads: dict[str, tuple[Arity, str]],
    user: dict[str, int],
    errors: list[str],
    depth: int,
) -> None:
    if not form.items:
        errors.append(f"{pos(form.lparen)}: empty form '()' is not valid")
        return
    head = form.items[0]
    operands = form.items[1:]
    if isinstance(head, Form):
        errors.append(f"{pos(form.lparen)}: form head must be a symbol, not a nested form (no first-class calls in v0.1)")
    elif head.kind != "SYMBOL":
        errors.append(f"{pos(head)}: form head must be a symbol, found {head.kind} {head.value!r}")
    else:
        name = head.value
        got = len(operands)
        if name in heads:
            arity, _group = heads[name]
            if got < arity.lo or (arity.hi is not None and got > arity.hi):
                errors.append(f"{pos(head)}: '{name}' expects {expected_msg(arity)} operand(s), found {got}")
            else:
                check_special(name, form, operands, errors, depth, set(heads))
        elif name in user:
            if got != user[name]:
                errors.append(
                    f"{pos(head)}: '{name}' expects {user[name]} operand(s) (declared by defn), found {got}"
                )
        else:
            errors.append(f"{pos(head)}: unknown head '{name}' (not in the arity table and not a declared defn)")
    # Parameter-list operands of fn/defn are groups of symbols, not call
    # forms -- the fiscal must not audit them as expressions.
    skip: set[int] = set()
    if isinstance(head, Tok) and head.kind == "SYMBOL" and head.value in ("fn", "defn"):
        skip.add(1 if head.value == "defn" else 0)
    for idx, it in enumerate(operands):
        if isinstance(it, Form) and idx not in skip:
            walk(it, heads, user, errors, depth + 1)


def check_source(src: str, table_path: str | Path = TABLE_PATH) -> list[str]:
    try:
        toks = tokenize(src)
    except FiscalError as e:
        return [str(e)]
    forms, errors = parse_program(toks)
    if errors:
        return errors
    heads = load_table(table_path)
    user = collect_defns(forms)
    for f in forms:
        walk(f, heads, user, errors, depth=1)
    return errors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Netelpro fiscal: mechanical arity checker (Phase 0)")
    ap.add_argument("files", nargs="+", help=".sl files to check")
    ap.add_argument("--table", default=str(TABLE_PATH), help="path to arity_table.json")
    args = ap.parse_args(argv)
    total = 0
    for fp in args.files:
        src = Path(fp).read_text(encoding="utf-8")
        errs = check_source(src, args.table)
        if errs:
            total += len(errs)
            print(f"== {fp}: {len(errs)} error(s)")
            for e in errs:
                print(f"  {e}")
        else:
            print(f"== {fp}: OK")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()