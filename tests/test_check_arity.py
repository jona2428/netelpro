"""Phase 0 test suite: the fiscal must prove the grammar thesis.

Every test asserts that structural validity is decided purely by counting
parens/operands against the arity table -- no semantic analysis involved.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_arity as ca  # noqa: E402


def errors(src: str) -> list[str]:
    return ca.check_source(src)


VALID = """
; straylight v0.1 -- structurally valid program
(def limit 10)
(defn clamp (x) (if (< x 0) 0 (if (> x limit) limit x)))
(defn sum-to (n acc) (if (== n 0) acc (sum-to (- n 1) (+ acc n))))
(def total (sum-to limit 0))
(def clamped (clamp -3))
(def label (str-cat "sum: " (int->str total)))
(def xs (cons 1 (cons 2 nil)))
(def packed (list 1 2 3))
(def nothing (list))
(def flag (and (< clamped 5) (not false)))
(let tmp (+ total 1) (- tmp 1))
(sorry "unverified placeholder")
(grant io)
"""


def test_valid_program_has_zero_errors():
    assert errors(VALID) == []


def test_all_special_forms_and_primitives_used():
    # sanity: the valid program actually exercises the table heads
    heads = ca.load_table()
    used = set()
    toks = ca.tokenize(VALID)
    for t in toks:
        if t.kind == "SYMBOL" and t.value in heads:
            used.add(t.value)
    assert {"def", "defn", "if", "let", "and", "not", "sorry", "grant",
            "+", "-", "==", "<", ">", "cons", "list", "str-cat",
            "int->str"} <= used


def test_arithmetic_arity_violation():
    errs = errors("(+ 1 2 3)")
    assert len(errs) == 1
    assert "'+' expects 2 operand(s), found 3" in errs[0]
    assert errs[0].startswith("line 1")


def test_if_requires_three_operands():
    errs = errors("(if 1 2)")
    assert any("'if' expects 3 operand(s), found 2" in e for e in errs)


def test_unknown_head_rejected():
    errs = errors("(foo 1 2)")
    assert any("unknown head 'foo'" in e for e in errs)


def test_list_is_the_only_open_arity():
    assert errors("(list)") == []
    assert errors("(list 1 2 3 4 5 6 7 8 9)") == []


def test_empty_form_rejected():
    errs = errors("(def x ())")
    assert any("empty form '()' is not valid" in e for e in errs)


def test_unclosed_paren_reports_position():
    errs = errors("(def x 1")
    assert any("never closed" in e for e in errs)
    assert any("line 1, col 1" in e for e in errs)


def test_stray_paren_reports_position():
    errs = errors(")\n(def x 1)")
    assert any("stray ')'" in e for e in errs)
    assert any("line 1, col 1" in e for e in errs)


def test_token_outside_form_rejected():
    errs = errors("42")
    assert any("outside any form" in e for e in errs)


def test_sorry_requires_string_literal():
    errs = errors("(sorry 42)")
    assert any("'sorry' requires a string literal reason" in e for e in errs)
    assert errors('(sorry "legit reason")') == []


def test_grant_only_top_level():
    assert errors("(grant io net)") == []
    errs = errors("(defn f (x) (grant io))")
    assert any("'grant' is only valid at top level" in e for e in errs)


def test_defn_duplicate_params():
    errs = errors("(defn f (x x) x)")
    assert any("duplicate parameter 'x'" in e for e in errs)


def test_defn_params_must_be_symbols():
    errs = errors("(defn f (x 1) x)")
    assert any("parameters must be symbols" in e for e in errs)


def test_user_defn_call_arity():
    errs = errors("(defn f (x) x) (f 1 2)")
    assert any("'f' expects 1 operand(s) (declared by defn), found 2" in e for e in errs)
    assert errors("(defn f (x) x) (f 7)") == []


def test_forward_reference_allowed():
    assert errors("(def a (g 1)) (defn g (x) x)") == []


def test_reserved_heads_cannot_be_redefined():
    errs = errors("(defn if (a b c) a)")
    assert any("reserved head" in e for e in errs)
    errs2 = errors("(def + 5)")
    assert any("reserved head" in e for e in errs2)


def test_def_name_must_be_symbol():
    errs = errors("(def 5 5)")
    assert any("requires a symbol name" in e for e in errs)


def test_fn_param_checks():
    assert errors("(fn (x) (+ x 1))") == []
    errs = errors("(fn (x x) x)")
    assert any("duplicate parameter 'x'" in e for e in errs)
    errs2 = errors("(fn x x)")
    assert any("parameter list must be a parenthesized group" in e for e in errs2)


def test_form_head_must_be_symbol():
    errs = errors("(1 2)")
    assert any("form head must be a symbol" in e for e in errs)
    errs2 = errors("((fn (x) x) 1)")
    assert any("not a nested form" in e for e in errs2)


def test_invalid_token():
    errs = errors("(1ab 2)")
    assert any("invalid token" in e for e in errs)


def test_unterminated_string():
    errs = errors('(def s "abc')
    assert any("unterminated string" in e for e in errs)


def test_invalid_escape():
    errs = errors(r'(def s "a\qb")')
    assert any("invalid escape" in e for e in errs)


def test_comments_ignored_and_negative_numbers():
    src = "; header comment\n(def neg (- 0 5)) ; trailing comment\n(def f -2.5)\n"
    assert errors(src) == []


def test_errors_are_positioned():
    src = "(def x 1)\n(def y 2)\n(+ 1 2 3)\n"
    errs = errors(src)
    assert errs and errs[0].startswith("line 3")