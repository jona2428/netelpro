"""Phase 2 test suite for the Straylight tree-walking evaluator.

Validates semantic contracts including value model, strict booleans,
arithmetic rules, comparisons, lexical closures, tail call optimization (TCO),
list primitives, string conversions, prosecutorial diagnostics, and top-level evaluation.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import straylight

from straylight import (
    run_source,
    StrayError,
    StrayRuntimeError,
    StrayHoleError,
    StrayList,
    is_nil,
    format_value,
)

rs = run_source


class TestArithmetic:
    def test_add_happy_path(self) -> None:
        val = rs("(+ 1 2)")
        assert val == 3
        assert type(val) is int

    def test_sub_happy_path(self) -> None:
        val = rs("(- 10 4)")
        assert val == 6
        assert type(val) is int

    def test_mul_happy_path(self) -> None:
        val = rs("(* 3 4)")
        assert val == 12
        assert type(val) is int

    def test_promotion(self) -> None:
        val = rs("(+ 1 2.5)")
        assert val == 3.5
        assert isinstance(val, float)

    def test_div_always_float(self) -> None:
        val1 = rs("(/ 7 2)")
        assert val1 == 3.5
        assert isinstance(val1, float)

        val2 = rs("(/ 6 3)")
        assert val2 == 2.0
        assert isinstance(val2, float)

    def test_div_by_zero_int(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(/ 1 0)")

    def test_div_by_zero_float(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(/ 1 0.0)")

    def test_quot_truncation(self) -> None:
        assert rs("(quot 7 2)") == 3
        assert rs("(quot -7 2)") == -3

    def test_rem(self) -> None:
        assert rs("(rem -7 2)") == -1
        assert rs("(rem 7 2)") == 1

    def test_quot_int_only(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(quot 7.0 2)")

    def test_quot_zero_div(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(quot 7 0)")

    def test_arithmetic_bool_trap(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(+ true 1)")


class TestComparisons:
    def test_less_than(self) -> None:
        assert rs("(< 1 2)") is True
        assert rs("(< 1.5 2)") is True

    def test_less_than_or_equal(self) -> None:
        assert rs("(<= 2 2)") is True

    def test_greater_than(self) -> None:
        assert rs("(> 3 2)") is True

    def test_greater_than_or_equal(self) -> None:
        assert rs("(>= 2 2)") is True

    def test_equality_string(self) -> None:
        assert rs('(== "x" "x")') is True

    def test_equality_cross_type(self) -> None:
        assert rs('(== 1 "1")') is False
        assert rs('(!= 1 "1")') is True

    def test_equality_bool_int_trap(self) -> None:
        assert rs("(== true 1)") is False

    def test_equality_nil(self) -> None:
        assert rs("(== nil nil)") is True

    def test_ordering_non_numeric(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs('(< "a" "b")')
        with pytest.raises(StrayRuntimeError):
            rs("(< true false)")


class TestStrictBools:
    def test_if_non_bool(self) -> None:
        with pytest.raises(StrayRuntimeError, match="condition is not Bool"):
            rs("(if 1 2 3)")

    def test_and_non_bool_left(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(and 1 true)")

    def test_or_non_bool_right(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(or false 2)")

    def test_not_non_bool(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(not 1)")


class TestShortCircuit:
    def test_or_short_circuit_success(self) -> None:
        assert rs("(or true (/ 1 0))") is True

    def test_and_short_circuit_success(self) -> None:
        assert rs("(and false (/ 1 0))") is False

    def test_or_evaluates_right_on_false(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(or false (/ 1 0))")

    def test_and_evaluates_right_on_true(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(and true (/ 1 0))")


class TestClosures:
    def test_make_adder_closure(self) -> None:
        # No first-class calls in v0.1 (spec §9.1): closures are exercised via
        # defn bodies capturing the global environment.
        src = """
        (def x 10)
        (defn add-x (y) (+ x y))
        (add-x 5)
        """
        assert rs(src) == 15

    def test_let_shadowing(self) -> None:
        src = """
        (def x 1)
        (let x 5 x)
        """
        assert rs(src) == 5

    def test_let_global_unpolluted(self) -> None:
        src = """
        (def x 1)
        (let x 5 nil)
        (== x 1)
        """
        assert rs(src) is True

    def test_nested_let_composition(self) -> None:
        src = """
        (let a 10
          (let b 20
            (let c (+ a b)
              (* c 2))))
        """
        assert rs(src) == 60

    def test_recursive_fibonacci(self) -> None:
        src = """
        (defn fib (n)
          (if (< n 2)
              n
              (+ (fib (- n 1)) (fib (- n 2)))))
        (fib 15)
        """
        assert rs(src) == 610


class TestTCO:
    def test_tco_deep_sum(self) -> None:
        src = """
        (defn sum-to (n acc)
          (if (== n 0)
              acc
              (sum-to (- n 1) (+ acc n))))
        (sum-to 100000 0)
        """
        assert rs(src) == 5000050000

    def test_tco_and_right(self) -> None:
        src = """
        (defn down (n)
          (if (== n 0)
              false
              (and true (down (- n 1)))))
        (down 100000)
        """
        assert rs(src) is False


class TestLists:
    def test_list_construct(self) -> None:
        res = rs("(list 1 2 3)")
        assert isinstance(res, StrayList)
        assert len(res) == 3

    def test_is_nil(self) -> None:
        assert rs("(is-nil nil)") is True
        assert rs("(is-nil (list 1))") is False

    def test_head(self) -> None:
        assert rs("(head (list 10 20))") == 10

    def test_tail(self) -> None:
        res = rs("(tail (list 10 20))")
        assert len(res) == 1
        assert rs("(head (tail (list 10 20)))") == 20

    def test_cons(self) -> None:
        res = rs("(cons 0 (list 1 2))")
        assert len(res) == 3
        assert rs("(head (cons 0 (list 1 2)))") == 0

    def test_head_nil_raises(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(head nil)")

    def test_tail_nil_raises(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(tail nil)")

    def test_nth(self) -> None:
        assert rs("(nth (list 1 2 3) 1)") == 2

    def test_nth_out_of_range(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(nth (list 1 2 3) 5)")
        with pytest.raises(StrayRuntimeError):
            rs("(nth (list 1 2 3) -1)")

    def test_len(self) -> None:
        assert rs("(len (list 1 2))") == 2

    def test_cons_invalid_second_arg(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs("(cons 1 2)")


class TestStrings:
    def test_str_cat(self) -> None:
        assert rs('(str-cat "foo" "bar")') == "foobar"

    def test_str_len(self) -> None:
        assert rs('(str-len "hola")') == 4

    def test_int_to_str(self) -> None:
        assert rs("(int->str 42)") == "42"

    def test_str_to_int(self) -> None:
        assert rs('(str->int "-17")') == -17

    def test_str_to_int_invalid(self) -> None:
        with pytest.raises(StrayRuntimeError):
            rs('(str->int "12a")')
        with pytest.raises(StrayRuntimeError):
            rs("(str->int 12)")

    def test_str_to_int_strict_whitespace(self) -> None:
        # Regression: str->int must not silently strip whitespace (prosecutor
        # strictness); '+5' is not an integer literal form in v0.1 either.
        with pytest.raises(StrayRuntimeError):
            rs('(str->int " 5")')
        with pytest.raises(StrayRuntimeError):
            rs('(str->int "+5")')

    def test_int_to_float(self) -> None:
        res = rs("(int->float 3)")
        assert res == 3.0
        assert isinstance(res, float)


class TestProsecutor:
    def test_sorry_raises_hole_error(self) -> None:
        with pytest.raises(StrayHoleError) as exc_info:
            rs('(sorry "not implemented")')
        assert "not implemented" in str(exc_info.value)
        assert exc_info.value.reason == "not implemented"

    def test_grant_capability_no_crash(self) -> None:
        assert rs("(grant io net) (+ 40 2)") == 42

    def test_runtime_error_position(self) -> None:
        with pytest.raises(StrayRuntimeError) as exc_info:
            rs("(/ 1 0)")
        assert exc_info.value.line >= 1
        assert exc_info.value.col >= 1

    def test_unbound_symbol(self) -> None:
        # Bare symbols are rejected at top level (top level must be forms);
        # unbound symbols surface at runtime inside a form.
        with pytest.raises(StrayRuntimeError) as exc_info:
            rs("(+ 1 y)")
        assert "unbound" in str(exc_info.value)

    def test_undefined_function_is_parse_error(self) -> None:
        with pytest.raises(StrayError) as exc_info:
            rs("(f 1)")
        assert "unknown head 'f'" in str(exc_info.value)

    def test_call_non_function(self) -> None:
        # A def-bound name is not callable in v0.1: the parser rejects the
        # head statically (no first-class calls).
        with pytest.raises(StrayError) as exc_info:
            rs("(def x 5) (x 1)")
        assert "unknown head 'x'" in str(exc_info.value)

    def test_arity_mismatch(self) -> None:
        # Arity of user defns is enforced statically by the parser fiscal.
        with pytest.raises(StrayError) as exc_info:
            rs("(defn f (x) x) (f 1 2)")
        assert "expects 1" in str(exc_info.value)

    def test_deep_non_tail_recursion_is_prosecutorial(self) -> None:
        # Regression: raw Python RecursionError must never leak to the user;
        # it is translated into a StrayRuntimeError with position.
        src = (
            "(defn f (n) (if (< n 2) n (+ (f (- n 1)) (f (- n 2))))) (f 5000)"
        )
        with pytest.raises(StrayRuntimeError, match="stack depth"):
            rs(src)


class TestTopLevel:
    def test_parse_errors_aggregate(self) -> None:
        with pytest.raises(StrayError) as exc_info:
            rs("(def x)")
        assert len(exc_info.value.errors) >= 1

    def test_last_value_semantics(self) -> None:
        # Top level must be forms (parser rule since Phase 0); the value of
        # the LAST evaluated expression form is returned.
        assert rs("(+ 1 2) (+ 3 4) (+ 5 6)") == 11

    def test_def_only_returns_nil(self) -> None:
        assert is_nil(rs("(def x 1)"))

    def test_print_output_and_nil_return(self, capsys: pytest.CaptureFixture[str]) -> None:
        res = rs('(print "hola")')
        captured = capsys.readouterr()
        assert "hola" in captured.out
        assert is_nil(res)


class TestValues:
    def test_format_value_numbers(self) -> None:
        assert format_value(3) == "3"
        assert "3" in format_value(3.0)

    def test_format_value_bool_and_list(self) -> None:
        b_str = format_value(True)
        assert isinstance(b_str, str)
        l_str = format_value(StrayList((1, 2)))
        assert isinstance(l_str, str)
        assert "1" in l_str

    def test_is_nil_helper(self) -> None:
        assert is_nil(StrayList()) is True
        assert is_nil(StrayList((1,))) is False
