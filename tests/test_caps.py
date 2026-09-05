"""Phase 3 test suite for Straylight capabilities-as-types.

Validates static capability enforcement (prosecutor philosophy), grant collection,
prosecutorial diagnostics with exact source coordinates, CLI enforcement,
and defense-in-depth runtime guards.
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
    is_nil,
)
from straylight.caps import (
    check_capabilities,
    collect_grants,
    CapError,
    KNOWN_CAPABILITIES,
)
from straylight.parser import parse
from straylight.__main__ import main


class TestStaticCapabilityCheck:
    """Static capability enforcement: effects audited before evaluation."""

    def test_print_without_grant_reports_cap_error_exact_coords(self) -> None:
        """print without grant -> CapError at exact line/col of the call node."""
        src = "  (print 123)"
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert len(errors) == 1
        err = errors[0]
        assert isinstance(err, CapError)
        assert err.line == 1
        assert err.col == 3
        assert "capability 'io' required by 'print' but not granted" in err.message
        assert "add (grant io) at top level" in err.message
        assert str(err) == "line 1, col 3: capability 'io' required by 'print' but not granted — add (grant io) at top level"

    def test_grant_io_allows_print(self) -> None:
        """(grant io) + print -> zero errors from check_capabilities."""
        src = '(grant io)\n(print "hello")'
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert errors == []

    def test_uncalled_defn_body_is_statically_checked(self) -> None:
        """print inside a defn body that is never called -> STILL detected (static, not runtime)."""
        src = """(defn dead-logger (msg)
  (print msg))
(def res (+ 1 2))
"""
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 2
        assert err.col == 3
        assert "capability 'io' required by 'print' but not granted" in err.message

    def test_nested_print_in_fn_let_if_branches_detected(self) -> None:
        """print nested in (fn ...) inside (let ...) inside (if ...) branches -> detected."""
        src = """(if true
  (let f (fn (x) (print x))
    f)
  (let g (fn (y) (print y))
    g))
"""
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert len(errors) == 2
        assert errors[0].line == 2
        assert errors[0].col == 18
        assert errors[1].line == 4
        assert errors[1].col == 18
        for err in errors:
            assert "capability 'io' required by 'print'" in err.message

    def test_two_un_granted_prints_produce_aggregated_errors(self) -> None:
        """two print calls both un-granted -> 2 aggregated CapErrors."""
        src = '(print "first")\n(print "second")\n'
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert len(errors) == 2
        assert errors[0].line == 1
        assert errors[0].col == 1
        assert errors[1].line == 2
        assert errors[1].col == 1
        assert all("capability 'io'" in err.message for err in errors)

    def test_unknown_capability_mentions_known_capabilities(self) -> None:
        """(grant net) -> unknown-capability error message mentions known capabilities: io."""
        src = "(grant net)"
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert len(errors) == 1
        err = errors[0]
        assert "unknown capability 'net'" in err.message
        assert "known capabilities: io" in err.message

    def test_pure_program_has_zero_capability_errors(self) -> None:
        """pure program (no print) -> no capability errors."""
        src = """(def x 10)
(defn double (n) (* n 2))
(def y (double x))
(if (> y 15) y 0)
"""
        pr = parse(src)
        assert pr.ok
        errors = check_capabilities(pr.program)
        assert errors == []


class TestGrantGrammarRegressions:
    """Grammar regression tests for the grant special form."""

    def test_grant_not_top_level_rejected(self) -> None:
        """grant not top-level (nested) -> parse().ok is False (regression)."""
        src = "(defn f () (grant io))"
        pr = parse(src)
        assert not pr.ok
        assert any("'grant' is only valid at top level, not nested" in err.message for err in pr.errors)

    def test_grant_zero_operands_rejected(self) -> None:
        """(grant) with zero operands -> parse().ok is False (arity min 1, regression)."""
        src = "(grant)"
        pr = parse(src)
        assert not pr.ok
        assert any("'grant' expects at least 1 operand(s), found 0" in err.message for err in pr.errors)


class TestCliEndToEnd:
    """CLI end-to-end integration tests via straylight.__main__.main(argv)."""

    def test_cli_end_to_end_enforcement(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """temp .sl file with print and no grant -> main returns 1 and capsys stderr mentions 'capability';
        same file with (grant io) -> returns 0.
        """
        # Negative case: un-granted print fails compilation before evaluation
        bad_file = tmp_path / "bad.sl"
        bad_file.write_text('(print "fail")', encoding="utf-8")
        ret_bad = main([str(bad_file)])
        captured_bad = capsys.readouterr()
        assert ret_bad == 1
        assert "capability" in captured_bad.err.lower()

        # Positive case: granted print compiles and evaluates successfully
        good_file = tmp_path / "good.sl"
        good_file.write_text('(grant io)\n(print "ok")', encoding="utf-8")
        ret_good = main([str(good_file)])
        captured_good = capsys.readouterr()
        assert ret_good == 0
        assert "ok" in captured_good.out


class TestDefenseInDepthAndIntegration:
    """Defense-in-depth runtime checks and direct evaluator integration."""

    def test_defense_in_depth_runtime_guard(self) -> None:
        """run_source('(print "x")') (no grant) raises StrayRuntimeError matching "capability 'io'"."""
        with pytest.raises(StrayRuntimeError, match="capability 'io'"):
            run_source('(print "x")')

    def test_integration_run_source_with_grant(self, capsys: pytest.CaptureFixture[str]) -> None:
        """run_source('(grant io) (print "ok")') returns nil and capsys captured out contains 'ok'."""
        result = run_source('(grant io) (print "ok")')
        assert is_nil(result)
        captured = capsys.readouterr()
        assert "ok" in captured.out


class TestCollectGrantsAndCapError:
    """Unit tests for collect_grants and CapError records."""

    def test_collect_grants_varieties(self) -> None:
        pr_empty = parse("(+ 1 2)")
        assert collect_grants(pr_empty.program) == set()

        pr_single = parse("(grant io)")
        assert collect_grants(pr_single.program) == {"io"}

        pr_multiple_forms = parse("(grant io)\n(grant net)")
        assert collect_grants(pr_multiple_forms.program) == {"io", "net"}

        pr_multiple_args = parse("(grant io net fs)")
        assert collect_grants(pr_multiple_args.program) == {"io", "net", "fs"}

    def test_cap_error_frozen(self) -> None:
        err = CapError("unauthorized", line=4, col=2)
        assert err.message == "unauthorized"
        assert err.line == 4
        assert err.col == 2
        assert str(err) == "line 4, col 2: unauthorized"
        with pytest.raises(Exception):
            err.line = 5  # type: ignore[misc]
