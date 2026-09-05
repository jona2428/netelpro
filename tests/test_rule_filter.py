"""Tests for the rule_filter bridge (Phase 6): Neuromancer gate rules as
compiled pure Netelpro functions.

Boundary law (v0.1): all filter-rule params are Int (i64); Booleans cross
the machine boundary as 0/1 flags. Differential verification vs the
interpreter is mandatory for every compiled rule.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netelpro.rule_filter import (  # noqa: E402
    RuleFilter,
    RuleFilterError,
    compile_filter,
)

RULE_SRC = """\
(defn filter-rule (priority confidence approved)
  (or (and (>= priority 3) (< confidence 90))
      (== approved 1)))
"""


class TestCompileAndManifest:
    def test_compiles_clean_rule(self):
        f = compile_filter(RULE_SRC)
        assert f.manifest() == []

    def test_sorry_manifest_is_listed_not_rejected(self):
        src = RULE_SRC + '\n(defn unused () (sorry "pending rule refinement"))\n'
        f = compile_filter(src)
        mf = f.manifest()
        assert len(mf) == 1
        assert "pending rule refinement" in mf[0]
        assert "line" in mf[0]

    def test_missing_filter_rule_rejected(self):
        with pytest.raises(RuleFilterError) as ei:
            compile_filter("(defn other (x) x)")
        assert "'filter-rule'" in str(ei.value)
        assert ei.value.line == 0 and ei.value.col == 0

    def test_bool_param_compiles_and_decides(self):
        # v0.2: a param whose use statically demands Bool is LEGAL — it
        # compiles to an i1 boundary param. (if b 1 0) -> Bool param, Int return.
        f = compile_filter("(defn filter-rule (b) (if b 1 0))")
        assert f.decide(True) is True
        assert f.decide(False) is False
        # Differential agreement on both engines (bools serialized as true/false).
        assert f.verify([((True,), True), ((False,), False)]) == []

    def test_mixed_use_param_conflict_rejected(self):
        # The prosecutor still kills ambiguity: the SAME param demanded Bool
        # (and) and Int (+) is a compile error with exact coordinates.
        with pytest.raises(RuleFilterError) as ei:
            compile_filter("(defn filter-rule (flag) (and flag (+ flag 1)))")
        assert "type mismatch" in str(ei.value)
        assert ei.value.line >= 1 and ei.value.col >= 1

    def test_parse_error_carries_position(self):
        with pytest.raises(RuleFilterError) as ei:
            compile_filter("(defn filter-rule (x) x")  # unterminated
        assert ei.value.line >= 1

    def test_ungranted_print_rejected_at_compile(self):
        with pytest.raises(RuleFilterError):
            compile_filter(RULE_SRC + '(print "sneaky")')

    def test_deny_filter_rule_shadowing(self):
        with pytest.raises(RuleFilterError):
            compile_filter(
                "(defn filter-rule (x) x)\n(defn filter-rule (y) y)"
            )


class TestDecide:
    def _f(self):
        return compile_filter(RULE_SRC)

    def test_truth_table_native(self):
        f = self._f()
        assert f.decide(3, 80, 0) is True
        assert f.decide(2, 80, 0) is False
        assert f.decide(3, 95, 0) is False
        assert f.decide(2, 95, 1) is True
        assert f.decide(4, 10, 0) is True

    def test_decide_accepts_python_bool_flag(self):
        # Boundary: approved as Python bool True -> ctypes c_int64 accepts it.
        f = self._f()
        assert f.decide(2, 95, True) is True

    def test_decide_arity_mismatch_raises(self):
        f = self._f()
        with pytest.raises((TypeError, RuleFilterError)):
            f.decide(3, 80)

    def test_recursion_rule_native(self):
        # Parity via mutual fallback: proves native TCO (1001 deep, no stack
        # growth). n=0 -> 0, n=1 -> 1, else recurse by 2.
        src = (
            "(defn filter-rule (n)\n"
            "  (if (== n 0)\n"
            "      0\n"
            "      (if (== n 1)\n"
            "          1\n"
            "          (filter-rule (- n 2)))))\n"
        )
        f = compile_filter(src)
        assert f.decide(0) is False
        assert f.decide(1) is True
        assert f.decide(50) is False
        assert f.decide(51) is True
        assert f.decide(1001) is True  # TCO: 1001 frames, no host stack

    def test_def_constant_in_rule(self):
        src = ("(def threshold 3)\n"
               "(defn filter-rule (p) (>= p threshold))\n")
        f = compile_filter(src)
        assert f.decide(3) is True
        assert f.decide(2) is False


class TestVerifyDifferential:
    def test_agreement_on_truth_table(self):
        f = compile_filter(RULE_SRC)
        cases = [((3, 80, 0), True), ((2, 80, 0), False), ((3, 95, 0), False),
                 ((2, 95, 1), True), ((4, 10, 0), True), ((1, 99, 0), False)]
        assert f.verify(cases) == []

    def test_mismatch_detection(self):
        # Force a mismatch by comparing the native rule against a WRONG
        # host-side expectation — verify() must report it.
        f = compile_filter(RULE_SRC)
        cases = [((3, 80, 0), False)]
        mism = f.verify(cases)
        assert mism and mism[0][0] == (3, 80, 0)

    def test_verify_runs_both_engines(self):
        src = (
            "(defn filter-rule (n)\n"
            "  (if (== n 0)\n"
            "      0\n"
            "      (if (== n 1)\n"
            "          1\n"
            "          (filter-rule (- n 2)))))\n"
        )
        f = compile_filter(src)
        cases = [((0,), False), ((1,), True), ((50,), False),
                 ((51,), True), ((1001,), True)]
        assert f.verify(cases) == []


class TestFactory:
    def test_compile_filter_is_rulefilter(self):
        assert isinstance(compile_filter(RULE_SRC), RuleFilter)

    def test_error_str_includes_position(self):
        with pytest.raises(RuleFilterError) as ei:
            compile_filter("(defn filter-rule (flag) (and flag (+ flag 1)))")
        s = str(ei.value)
        assert "line" in s and "col" in s


class TestGateRuleExample:
    def test_example_file_compiles_and_verifies(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "gate_rule.sl"
        f = compile_filter(example.read_text(encoding="utf-8"))
        assert f.manifest() == []
        cases = [((3, 80, 0), True), ((2, 80, 0), False), ((3, 95, 0), False),
                 ((2, 95, 1), True), ((4, 10, 0), True), ((1, 99, 0), False)]
        assert f.verify(cases) == []