"""Fase 3 contract tests: per-function effect rows (spec §8, tests 1-12)."""

from netelpro.effects import check_effect_rows
from netelpro.evaluator import evaluate
from netelpro.parser import parse


def _rows(src: str):
    pr = parse(src)
    return pr, check_effect_rows(pr.program)


def test_effect_declared_allows_io():
    pr, errs = _rows('(defn f () : (effects (io "*"))\n  (print "hola"))')
    assert pr.ok
    assert errs == []


def test_effect_missing_rejects_io():
    pr, errs = _rows('(defn f ()\n  (print "hola"))')
    assert pr.ok
    assert pr.ok
    assert len(errs) == 1
    assert "requires effect (io \"*\")" in errs[0].message
    assert errs[0].line == 2 and errs[0].col == 3


def test_composition_caller_subset_rejected():
    src = (
        '(defn inner () : (effects (io "*") (write "*") (delete "*"))\n'
        '  (print "x"))\n'
        '(defn main () : (effects (io "*") (write "*"))\n'
        '  (inner))'
    )
    pr, errs = _rows(src)
    assert pr.ok  # syntactically fine — rejected by effect composition, not parse
    assert len(errs) == 1
    assert 'brings effects (delete "*")' in errs[0].message
    assert "declares only" in errs[0].message


def test_composition_exact_match_ok():
    src = (
        '(defn inner () : (effects (io "*") (write "*") (delete "*"))\n'
        '  (print "x"))\n'
        '(defn main () : (effects (io "*") (write "*") (delete "*"))\n'
        '  (inner))'
    )
    pr, errs = _rows(src)
    assert pr.ok and errs == []


def test_effectful_from_pure_rejected():
    src = (
        '(defn inner () : (effects (read "./data/*")) 1)\n'
        '(defn main ()\n'
        '  (inner))'
    )
    pr, errs = _rows(src)
    assert pr.ok  # parse fine — composition (D2) rejects
    assert len(errs) == 1
    assert "declares none" in errs[0].message
def test_unknown_verb_rejected():
    pr = parse('(defn f () : (effects (execute "*")) 1)')
    assert not pr.ok
    assert any("unknown effect verb 'execute'" in e.message for e in pr.errors)


def test_valid_verbs_accepted():
    pr = parse('(defn f () : (effects (read "a") (write "b") (delete "c") (network "d")) 1)')
    assert pr.ok


def test_empty_pattern_rejected():
    pr = parse('(defn f () : (effects (read "")) 1)')
    assert not pr.ok
    assert any("pattern cannot be empty" in e.message for e in pr.errors)


def test_duplicate_effect_warning():
    pr = parse('(defn f () : (effects (read "a") (read "a")) 1)')
    assert pr.ok
    assert len(pr.effect_warnings) == 1
    assert "deduplicated" in pr.effect_warnings[0].message


def test_effect_in_truth_table_forbidden():
    # truth-table is top-level-only and desugars to an if-chain inside the
    # enclosing defn — so the defn's effects clause governs its rows.
    src = (
        '(defn f ((b : Bool))\n'
        '  : (effects (io "*"))\n'
        '  (if b (print "x") false))\n'
        '(f true)'
    )
    pr = parse(src)
    assert pr.ok  # declared io covers the print the table body reaches
    top_only = [e for e in check_effect_rows(pr.program) if "top-level" not in e.message]
    assert top_only == []
    # ...but without the declaration the same body is rejected (B2 composition)
    src2 = (
        '(defn f ((b : Bool))\n'
        '  (if b (print "x") false))\n'
        '(f true)'
    )
    pr2 = parse(src2)
    assert any("requires effect" in e.message for e in check_effect_rows(pr2.program))


def test_lambda_no_effects_syntax():
    src = '(defn f () ((fn () : (effects (io "*")) 1)))'
    pr = parse(src)
    # fn with 4 operands violates its arity [2,2] — clause is not fn syntax.
    assert not pr.ok


def test_top_level_effectful_call_rejected():
    src = (
        '(defn f () : (effects (io "*"))\n'
        '  (print "x"))\n'
        '(f)'
    )
    _, errs = _rows(src)
    assert len(errs) == 1
    assert "top-level call" in errs[0].message


def test_evaluation_unchanged_effects_metadata_only():
    # Metadata-only (D1): evaluation of the WITH-clause program must be
    # structurally identical to the same program WITHOUT the clause.
    with_clause = (
        '(defn f () : (effects (io "*"))\n'
        '  (+ 1 2))\n'
        '(f)'
    )
    without = (
        '(defn f ()\n'
        '  (+ 1 2))\n'
        '(f)'
    )
    assert evaluate(parse(with_clause).program) == evaluate(parse(without).program) == 3


def test_recursive_effects_declared_ok():
    src = (
        '(defn even? ((n : Int)) : (effects (io "*"))\n'
        '  (print "x"))\n'
        '(defn odd? ((n : Int)) : (effects (io "*"))\n'
        '  (even? n))'
    )
    pr, errs = _rows(src)
    assert pr.ok and errs == []
def test_regression_phase1_unchanged():
    src = (
        '(truth-table f (n : (Int 0 1))\n'
        '  ((0) -> 10)\n'
        '  ((1) -> 20)\n'
        '  ((_) -> 0))\n'
        '(f 1)'
    )
    pr = parse(src)
    assert pr.ok and evaluate(pr.program) == 20
    assert check_effect_rows(pr.program) == []


def test_def_effects_no_clause_semantics():
    # def constant: no effects clause exists (D11) — binds value as before.
    pr = parse('(def x 42)\n(defn f () x)\n(f)')
    assert pr.ok
    assert evaluate(pr.program) == 42


def test_grant_coexistence_d10():
    # Legacy top-level grant still owns file-wide io (caps pass unchanged).
    from netelpro.caps import check_capabilities, collect_grants
    src = (
        '(grant io)\n'
        '(defn f ()\n'
        '  (print "x"))\n'
        '(f)'
    )
    pr = parse(src)
    cap_errs = check_capabilities(pr.program, collect_grants(pr.program))
    assert cap_errs == []  # caps pass: granted
    row_errs = check_effect_rows(pr.program)
    assert len(row_errs) == 1  # effect rows: f must declare its own io