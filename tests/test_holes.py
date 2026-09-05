"""Phase 4 test suite: static hole prosecution & the parser's silent-hole law.

The parser fiscal (Phase 1) already enforces most of Phase 4 at parse time:
- unknown call heads (not in arity table, not a declared defn, not a special form)
  are PARSE errors: "unknown head 'X' (not in the arity table and not a declared defn)"
- duplicate top-level defn: PARSE error "duplicate defn 'f' (already defined at top level)"
- def/defn/grant are top-level ONLY (nested declarations are parse errors)
- defn/def bodies are single-expression (fixed arity 3)

Phase 4 adds the static holes pass (holes.py): the sorry MANIFEST — every
(sorry "reason") compiles clean, is collected with exact coordinates, and
raises StrayHoleError if ever executed. Forward refs & mutual recursion are
legal (the parser's registry-based check handles them: registry counts
top-level defns BEFORE walking bodies).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netelpro.parser import parse
from netelpro.holes import check_holes, collect_declarations, HoleError
from netelpro.evaluator import (
    StrayHoleError,
    StrayRuntimeError,
    run_source,
)


class TestParserSilentHoleLaw:
    """Phase 4 verified in the parser: silent holes die at PARSE, with exact coords."""

    def test_unknown_head_is_parse_error(self) -> None:
        pr = parse("(defn main () (missing-fn 1))")
        assert not pr.ok
        msgs = [e.message for e in pr.errors]
        assert any("unknown head 'missing-fn'" in m for m in msgs)
        assert any("not in the arity table and not a declared defn" in m for m in msgs)

    def test_unknown_head_exact_coords(self) -> None:
        pr = parse("(defn main () (missing-fn 1))")
        err = pr.errors[0]
        assert err.line == 1
        assert err.col == 16

    def test_silent_hole_inside_list_arg_is_caught(self) -> None:
        pr = parse("(defn f (xs) (cons 1 xs))\n(defn main () (f (list (ghost 1))))")
        assert not pr.ok
        assert any("unknown head 'ghost'" in e.message for e in pr.errors)

    def test_def_value_unknown_head_caught(self) -> None:
        pr = parse("(def x (missing 1))")
        assert not pr.ok
        assert any("unknown head 'missing'" in e.message for e in pr.errors)

    def test_aggregation_multiple_unknown_heads(self) -> None:
        src = (
            "(defn main ()\n"
            "  (if true (alpha 1) (beta 2)))\n"
            "(defn other () (gamma 3))"
        )
        pr = parse(src)
        assert not pr.ok
        msgs = [e.message for e in pr.errors]
        assert any("'alpha'" in m for m in msgs)
        assert any("'beta'" in m for m in msgs)
        assert any("'gamma'" in m for m in msgs)

    def test_duplicate_top_level_defn_is_parse_error(self) -> None:
        pr = parse("(defn f (x) x)\n(defn f (x) (* x 2))")
        assert not pr.ok
        assert any("duplicate defn 'f' (already defined at top level)" in e.message
                   for e in pr.errors)

    def test_no_first_class_calls_v0_1(self) -> None:
        pr = parse("((fn (f) (f 1)) (fn (x) x))")
        assert not pr.ok
        assert any("no first-class calls in v0.1" in e.message for e in pr.errors)


class TestSorryManifest:
    """(sorry "reason") compiles clean and is collected — never hidden."""

    def hs(self, src: str):
        pr = parse(src)
        assert pr.ok, f"parse failed: {pr.errors}"
        return check_holes(pr.program)

    def test_sorry_compiles_clean_with_manifest(self) -> None:
        src = '(defn not-yet (x) (sorry "TODO: implement"))\n(defn main () (not-yet 1))'
        errors, manifest = self.hs(src)
        assert len(errors) == 0
        assert len(manifest) == 1
        assert manifest[0]["reason"] == "TODO: implement"
        assert manifest[0]["line"] == 1
        assert manifest[0]["col"] == 19

    def test_two_sorries_two_manifest_entries(self) -> None:
        src = '(defn a () (sorry "one"))\n(defn b () (sorry "two"))'
        errors, manifest = self.hs(src)
        assert len(errors) == 0
        assert len(manifest) == 2
        assert [m["reason"] for m in manifest] == ["one", "two"]

    def test_manifest_positions_exact(self) -> None:
        _, manifest = self.hs('(defn a ()\n  (sorry "here"))')
        assert manifest[0]["line"] == 2
        assert manifest[0]["col"] == 3

    def test_sorry_in_conditional_never_fires_at_compile(self) -> None:
        src = '(defn a (x) (if (< x 0) (sorry "negatives") (+ x 1)))'
        errors, manifest = self.hs(src)
        assert len(errors) == 0
        assert len(manifest) == 1

    def test_sorry_still_raises_at_runtime(self) -> None:
        with pytest.raises(StrayHoleError):
            run_source('(sorry "boom")')

    def test_sorry_inside_let_value_collected(self) -> None:
        src = '(defn a () (let y (sorry "deferred") y))'
        errors, manifest = self.hs(src)
        assert len(errors) == 0
        assert len(manifest) == 1


class TestHolesPassAPI:
    def test_collect_declarations_top_level(self) -> None:
        pr = parse("(defn a (x) x)\n(def b 2)")
        decls = collect_declarations(pr.program)
        assert decls == {"a": 1, "b": 2}

    def test_collect_declarations_skips_non_defs(self) -> None:
        pr = parse("(print 1)")
        decls = collect_declarations(pr.program)
        assert decls == {}

    def test_hole_error_str_format(self) -> None:
        err = HoleError(message="boom", line=3, col=7)
        assert str(err) == "line 3, col 7: boom"

    def test_clean_program_empty_both(self) -> None:
        pr = parse("(defn add (x y) (+ x y))\n(defn main () (add 1 2))")
        errors, manifest = check_holes(pr.program)
        assert errors == []
        assert manifest == []

    def test_grant_form_ignored(self) -> None:
        pr = parse("(grant io)\n(defn main () (print \"hi\"))")
        errors, manifest = check_holes(pr.program)
        assert errors == []
        assert manifest == []


class TestResolutionLegal:
    """What the parser law permits: forward refs and mutual recursion compile."""

    def test_forward_reference_top_level(self) -> None:
        pr = parse("(defn a (x) (b x))\n(defn b (x) (* x 2))\n(defn main () (a 21))")
        assert pr.ok
        errors, _ = check_holes(pr.program)
        assert errors == []

    def test_mutual_recursion(self) -> None:
        src = (
            "(defn even? (n) (if (== n 0) true (odd? (- n 1))))\n"
            "(defn odd? (n) (if (== n 0) false (even? (- n 1))))\n"
            "(defn main () (even? 10))"
        )
        pr = parse(src)
        assert pr.ok, [str(e) for e in pr.errors]
        errors, _ = check_holes(pr.program)
        assert errors == []

    def test_forward_ref_runtime_works(self) -> None:
        # run_source evaluates the LAST top-level form as the program value;
        # defn is just a declaration, so the call must be the last form.
        assert run_source("(defn a (x) (b x))\n(defn b (x) (* x 2))\n(a 21)") == 42


class TestRuntimeBehaviorVerified:
    def test_let_syntax_is_three_operand(self) -> None:
        """Verified: (let name value body) — fixed arity 3."""
        assert run_source("(defn main () (let x 5 (+ x 1)))\n(main)") == 6

    def test_def_value_called_as_fn_is_parse_error_not_runtime(self) -> None:
        """Even the v0.1 limitation is moot: the parser rejects it at PARSE time."""
        pr = parse("(def x 5)\n(x 1)")
        assert not pr.ok
        assert any("unknown head 'x'" in e.message for e in pr.errors)

    def test_sorry_reason_must_be_string(self) -> None:
        pr = parse("(sorry 42)")
        assert not pr.ok
        assert any("string literal" in e.message for e in pr.errors)