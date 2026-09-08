"""Phase 2 contract tests: Int refinement types (spec 2026-09-08 F2, §7).

Boundaries B1-B12 from spec §5. The refinement prosecutor is static-only
(erasure total): nothing reaches evaluator or codegen.
"""
from netelpro.parser import parse


def _errs(src: str) -> list[str]:
    return [str(e) for e in parse(src).errors]


# ---------- §5 B1: literal satisfies ----------


def test_ref_literal_satisfies():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(f 5)')
    assert r == []


# ---------- §5 B2: literal violates ----------


def test_ref_literal_violates():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(f 0)')
    assert any('violates refinement' in e for e in r)


# ---------- §5 B3: unguarded variable ----------


def test_ref_unguarded_variable():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(defn g ((y : Int)) (f y))')
    assert any('not proven' in e for e in r)


# ---------- §5 B4: guard dominates in then-branch ----------


def test_ref_guarded_variable_if_then():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(defn g ((x : Int)) (if (> x 0) (f x) 0))')
    assert r == []


# ---------- §5 B5: guard on wrong variable ----------


def test_ref_guard_wrong_variable():
    r = _errs(
        '(defn f ((x : (Ref Int (> 0)))) x)\n'
        '(defn g ((x : Int) (y : Int)) (if (> y 0) (f x) 0))'
    )
    assert any('not proven' in e for e in r)


# ---------- §5 B6: call in else (guard proves <= 0 there) ----------


def test_ref_call_in_else_unguarded():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(defn g ((x : Int)) (if (> x 0) 0 (f x)))')
    assert any('not proven' in e for e in r)


# ---------- §5 B7: And short-circuit dominates ----------


def test_ref_guarded_variable_and():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(defn g ((x : Int)) (and (> x 0) (f x)))')
    assert r == []


# ---------- §5 B8: nested guards conjunct + integer op lattice ----------


def test_ref_nested_guards():
    r = _errs(
        '(defn f ((x : (Ref Int (> 0) (<= 10)))) x)\n'
        '(defn g ((x : Int)) (if (> x 0) (if (< x 10) (f x) 0) 0))'
    )
    assert r == []


def test_ref_lattice_weak_guard_does_not_prove():
    r = _errs('(defn f ((x : (Ref Int (< 10)))) x)\n(defn g ((x : Int)) (if (<= x 10) (f x) 0))')
    assert any('not proven' in e for e in r)


def test_ref_lattice_strong_guard_implies_weak():
    r = _errs('(defn f ((x : (Ref Int (!= 0)))) x)\n(defn g ((x : Int)) (if (> x 0) (f x) 0))')
    assert r == []


# ---------- §5 B10: variable operand in predicate ----------


def test_ref_predicate_variable_operand():
    r = _errs('(defn h ((y : Int)) y)\n(defn f ((x : (Ref Int (> y)))) x)')
    assert any('unsupported parameter type' in e for e in r)# ---------- §5 B11: refinements forbidden in truth-table params (D5) ----------


def test_ref_in_truth_table():
    r = _errs('(truth-table tt (p : (Ref Int (> 0))) ((true) -> 1) ((_) -> 0))')
    assert any('truth-table' in e for e in r)


# ---------- §5 B12/D7: recursive self-call is a normal call-site ----------


def test_ref_recursive_call_unguarded():
    r = _errs(
        '(defn fact ((n : (Ref Int (>= 0)))) (if (== n 0) 1 (* n (fact (- n 1)))))'
    )
    assert any('not proven' in e for e in r)


def test_ref_recursive_call_guarded():
    # D7 faithful reading: guard on the expression itself, not arithmetic implication
    r = _errs(
        '(defn fact ((n : (Ref Int (>= 0))))'
        ' (if (== n 0) 1'
        ' (if (>= (- n 1) 0) (* n (fact (- n 1))) (sorry "unreachable"))))'
    )
    assert r == []


# ---------- plain Int becomes a legal type (spec F2 §1.1) ----------


def test_plain_int_type_legal():
    r = _errs('(defn f ((x : Int)) x)\n(f 5)')
    assert r == []


# ---------- Phase 1 bool/int strictness pin intact ----------


def test_fase1_regression_typed_params():
    r = _errs('(defn f ((x : Bool)) x)\n(f 1)')
    assert any('type-strict' in e for e in r)


def test_fase1_regression_enum():
    r = _errs(
        '(truth-table t1 (p : (Int 0 1))'
        ' ((0) -> 10) ((1) -> 20) ((_) -> 0))'
        '\n(t1 0)'
    )
    assert r == []


# ---------- let propagation semantics ----------


def test_let_literal_factoring():
    r = _errs('(defn f ((x : (Ref Int (> 0)))) x)\n(defn g () (let y 7 (f y)))')
    assert r == []


def test_let_inherits_variable_facts():
    r = _errs(
        '(defn f ((x : (Ref Int (> 0)))) x)\n'
        '(defn g ((z : Int)) (if (> z 0) (let w z (f w)) 0))'
    )
    assert r == []


def test_let_does_not_propagate_arithmetic():
    r = _errs(
        '(defn f ((x : (Ref Int (> 0)))) x)\n'
        '(defn g () (let q 7 (let r (+ q 1) (f r))))'
    )
    assert any('not proven' in e for e in r)


# ---------- erasure: refinements never reach the AST ----------


def test_ref_erasure_ast_clean():
    # Erasure total (D4): the refinement type never reaches the AST. The defn
    # must build cleanly (no silent drop) with no refinement traces anywhere.
    r = parse('(defn f ((x : (Ref Int (> 0)))) x)')
    assert r.ok
    assert len(r.program.forms) == 1
    defn = r.program.forms[0]
    assert defn.__class__.__name__ == 'Defn'
    rendered = repr(defn)
    assert 'Ref' not in rendered and 'Predicate' not in rendered