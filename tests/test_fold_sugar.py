"""Contract tests for fold sugar (v0.6): flat n-ary and/or -> right-assoc binary.

Response to external review (Gemini, 2026-09-08): binary-only and/or forces deep
nesting that small models lose paren balance on. fold keeps the arity table
mechanical (fold counts: op + >=2 operands) and gives LLMs flat syntax.
"""
from netelpro.parser import parse


def _body(src: str):
    r = parse(src)
    assert r.ok, [str(e) for e in r.errors]
    return r.program.forms[0].body


def test_fold_and_desugars_right_assoc():
    n = _body("(defn f () (fold and true false true))")
    from netelpro.ast_nodes import And, BoolLit
    assert isinstance(n, And)
    assert isinstance(n.l, BoolLit) and n.l.value is True
    assert isinstance(n.r, And)          # right-assoc: (and a (and b c))


def test_fold_or_desugars():
    from netelpro.ast_nodes import Or
    n = _body("(defn f () (fold or true true))")
    assert isinstance(n, Or)


def test_fold_rejects_non_logical_op_with_coords():
    r = parse("(defn f () (fold + 1 2))")
    assert not r.ok
    assert "'and' or 'or'" in r.errors[0].message
    assert r.errors[0].col == 18        # exact provenance of the bad operator


def test_fold_arity_violation_from_table():
    r = parse("(defn f () (fold and true))")
    assert not r.ok
    assert "expects at least 3" in r.errors[0].message


def test_fold_flat_disjunction_parses():
    n = _body("(defn f () (fold or (== x 1) (== x 2) (== x 3)))")
    from netelpro.ast_nodes import Or
    assert isinstance(n, Or)
