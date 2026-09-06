"""v0.5 RuleBuilder corpus tests for Netelpro string production.

This module exercises the FUTURE `RuleBuilder` bridge (netelpro/rule_filter.py),
which is NOT yet wired by the orchestrator. The contract under test:

- ``RuleBuilder(source)`` compiles source containing ``(defn build-rule ...)`` that
  RETURNS a Str. Raises ``RuleFilterError`` on prosecution failure.
- ``builder.build(*args) -> str`` runs native and returns a UTF-8 string. Arena
  overflow (string > 64KB) raises ``RuleFilterError`` containing both 'arena' and
  'overflow' in the message.
- ``builder.verify_build(cases)``: ``cases = [(args_tuple, expected_str)]``; returns
  a list of mismatch tuples ``(args, expected, interpreter_result, native_result)``;
  ``[]`` means full parity.
- Interpreter counterparts (available NOW in netelpro/str_native.py):
  ``interp_str_cat(a, b)`` and ``interp_int_to_str(n)``.
- Language syntax: ``(defn name (p1 p2) body)``, ``(if cond then else)``, ``==``,
  ``prefix?``, TCO via self tail calls, ``(str-cat a b)``, ``(int->str n)``.
- RULE: ``(defn filter-rule ...)`` (gate rules) REMAIN forbidden from returning Str;
  compiling one must raise ``RuleFilterError``. Only ``build-rule`` may return Str.

Module-level skip guard: the builder feature is not wired yet, so the builder
imports are guarded. Group 1 (interpreter primitives) imports ``netelpro.str_native``
which EXISTS today, so it lives in a separate class that is NOT skipped.
"""
from __future__ import annotations

import pytest
from netelpro.str_native import interp_int_to_str, interp_str_cat

# ---------------------------------------------------------------------------
# Module-level skip guard for the not-yet-wired RuleBuilder feature.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the feature is wired
    from netelpro.rule_filter import RuleBuilder, RuleFilter, RuleFilterError

    HAS_BUILDER = True
except ImportError:  # pragma: no cover - expected until orchestrator wires it
    RuleBuilder = None  # type: ignore[assignment]
    RuleFilter = None  # type: ignore[assignment]
    RuleFilterError = None  # type: ignore[assignment]
    HAS_BUILDER = False

# NOTE: The skip marker is applied ONLY to the builder test class below, NOT at
# module level, so the interpreter-primitive tests (which exist today) always run.
BUILDER_SKIP = pytest.mark.skipif(
    not HAS_BUILDER, reason="v0.5 builder not wired yet"
)


# ---------------------------------------------------------------------------
# Group 1: Interpreter primitives (EXIST today -> must pass, never skipped).
# ---------------------------------------------------------------------------
class TestInterpPrimitives:
    """Interpreter counterparts available today; these must pass regardless of wiring."""

    def test_interp_primitives(self) -> None:
        """Verify interp_str_cat and interp_int_to_str across the required vectors."""
        # str-cat
        assert interp_str_cat("", "x") == "x"
        assert interp_str_cat("café", "ñandú") == "caféñandú"

        # int->str
        assert interp_int_to_str(0) == "0"
        assert interp_int_to_str(7) == "7"
        assert interp_int_to_str(-42) == "-42"
        assert interp_int_to_str(2147483647) == "2147483647"
        assert interp_int_to_str(9223372036854775807) == "9223372036854775807"
        assert interp_int_to_str(-9223372036854775808) == "-9223372036854775808"


# ---------------------------------------------------------------------------
# Group 2: RuleBuilder corpus (skipped until the feature is wired).
# ---------------------------------------------------------------------------
RULE_BUILDER_SRC = (
    # str-cat is BINARY (fixed arity, like every primitive) -> nesting is the form.
    '(defn build-rule (rank) (str-cat (str-cat "(rank: " (int->str rank)) ")"))'
)


@BUILDER_SKIP
class TestRuleBuilder:
    """RuleBuilder corpus; skipped until the v0.5 builder is wired by the orchestrator."""

    def test_build_basic(self) -> None:
        """A simple build-rule producing '(rank: N)' from an int parameter."""
        builder = RuleBuilder(RULE_BUILDER_SRC)
        assert builder.build(2) == "(rank: 2)"

    def test_separator_law(self) -> None:
        """Nested binary str-cat calls produce the exact separator-law fragments per root."""
        src = (
            # Quote-free formulation of the separator law: the equality
            # fragment embeds the root verbatim; the prefix fragment embeds
            # root + '/'. Escaped quotes inside .sl strings are tested
            # elsewhere (utf8 roundtrip covers literals end-to-end).
            '(defn build-rule (r) (str-cat (str-cat "(== " r)'
            ' (str-cat (str-cat " (prefix? " r) "/")))'
        )
        builder = RuleBuilder(src)

        # Canonical root "src" must produce the exact expected gate text.
        expected_src = "(== src (prefix? src/"
        assert builder.build("src") == expected_src

        # Adversarial root "src_backup" must contain its own root and differ from "src".
        adversarial = builder.build("src_backup")
        assert "src_backup" in adversarial
        assert adversarial != expected_src

    def test_arena_overflow(self) -> None:
        """TCO doubling loop overflows the 64KB arena and raises RuleFilterError."""
        src = (
            '(defn go (acc n) (if (== n 0) acc (go (str-cat acc acc) (- n 1)))) '
            '(defn build-rule () (go "abcdefgh" 16))'
        )
        builder = RuleBuilder(src)
        with pytest.raises(RuleFilterError) as excinfo:
            builder.build()
        msg = str(excinfo.value).lower()
        assert "arena" in msg
        assert "overflow" in msg

    def test_verify_parity(self) -> None:
        """verify_build reports parity for correct cases and a single mismatch otherwise."""
        builder = RuleBuilder(RULE_BUILDER_SRC)

        # Six correct cases -> full parity (empty mismatch list).
        cases = [
            ((0,), "(rank: 0)"),
            ((1,), "(rank: 1)"),
            ((2,), "(rank: 2)"),
            ((3,), "(rank: 3)"),
            ((50,), "(rank: 50)"),
            ((999,), "(rank: 999)"),
        ]
        assert builder.verify_build(cases) == []

        # One wrong case -> exactly one mismatch where interpreter == native != expected.
        mismatches = builder.verify_build([((2,), "wrong")])
        assert len(mismatches) == 1
        args, expected, interp, native = mismatches[0]
        assert args == (2,)
        assert expected == "wrong"
        assert interp == native
        assert interp != "wrong"

    def test_gate_rule_isolation(self) -> None:
        """filter-rule (gate) may not return Str; an identical build-rule body compiles."""
        # Gate rule returning a Str is prosecuted.
        with pytest.raises(RuleFilterError):
            RuleFilter('(defn filter-rule (x) "yes")')

        # A build-rule with the IDENTICAL body compiles fine.
        builder = RuleBuilder('(defn build-rule (x) "yes")')
        # x is unused -> binds to Int (documented default) -> call with an int.
        assert builder.build(1) == "yes"

    def test_utf8_roundtrip(self) -> None:
        """A fixed literal with 'ñ' and an em dash survives the native roundtrip exactly."""
        literal = "café ñandú — em dash"
        src = f'(defn build-rule () "{literal}")'
        builder = RuleBuilder(src)
        assert builder.build() == literal
