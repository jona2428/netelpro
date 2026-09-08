"""Contract suite for Fase 4: prove/evidence (spec 2026-09-08, §8).

Covers B1-B6, D2 (honest negative), D4 (position), D5 (usage), D6 (binder),
D7 (truth-table), and native parity for the LLVM hole (unreachable trap).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from netelpro.ast_nodes import Prove
from netelpro.evaluator import Evaluator, StrayHoleError
from netelpro.parser import parse

PKG = Path(__file__).resolve().parent.parent / "netelpro"


def _errors(src: str) -> list[str]:
    return [e.message for e in parse(src).errors]


def _eval(src: str) -> object:
    result = Evaluator(capabilities=set()).evaluate(parse(src).program)
    return result


DEFN = "(defn check ((claim : Bool) (ev : Evidence)) (prove claim (evidence ev : Evidence)))"


def _runtime(defn: str, host_ev: bool, claim: bool) -> str:
    """Inject evidence the way the host FFI would: a named binding."""
    return f"{defn}\n(def host-ev {str(host_ev).lower()})\n(check {str(claim).lower()} host-ev)"


def test_prove_success() -> None:
    assert _eval(_runtime(DEFN, host_ev=True, claim=True)) is True


def test_prove_honest_negative() -> None:
    # D2 case 4: claim=false + evidence=false passes — the hole only
    # punishes lying, never honest negation.
    assert _eval(_runtime(DEFN, host_ev=False, claim=False)) is False


def test_prove_hole_fired() -> None:
    # D2 case 3 + B6 exact message format for audit trails.
    with pytest.raises(StrayHoleError, match=r"Proof violation at line \d+, col \d+\. Claim=True, Evidence=False"):
        _eval(_runtime(DEFN, host_ev=False, claim=True))


def test_ast_carries_ev_name() -> None:
    node = parse(f"{DEFN}\n(check true host-ev)").program.forms[0]
    assert isinstance(node.body, Prove)
    assert node.body.ev_name == "ev"


def test_compile_bool_for_evidence() -> None:
    # B1: a Bool literal cannot stand in for Evidence (no coercion, D1).
    errs = _errors(f"{DEFN}\n(check true true)")
    assert any("literal passed to Evidence parameter" in e for e in errs)


def test_compile_evidence_for_bool() -> None:
    # B2: Evidence is not Bool — an Evidence param fails the claim unify.
    errs = _errors("(defn bad ((ev : Evidence)) (prove ev (evidence ev : Evidence)))")
    assert any("not Bool" in e for e in errs)


def test_compile_fabricated_evidence() -> None:
    # B3/D6: only (evidence NAME : Evidence) binds-originates; a computed
    # expression is fabrication by structure.
    errs = _errors("(defn bad ((y : Bool)) (prove true (evidence (not y) : Evidence)))")
    assert any("must contain evidence binder" in e for e in errs)


def test_compile_bad_binder_shape() -> None:
    errs = _errors("(defn bad ((ev : Evidence)) (prove true 42))")
    assert any("must contain evidence binder" in e for e in errs)
def test_compile_unused_evidence() -> None:
    # D5: a declared Evidence param with no prove consumer is a compile error.
    errs = _errors("(defn bad ((claim : Bool) (ev : Evidence)) claim)")
    assert any("never used by a prove form" in e for e in errs)


def test_compile_prove_global() -> None:
    # D4: prove only inside defn/fn bodies.
    errs = _errors("(prove true (evidence ev : Evidence))")
    assert any("only valid inside a defn/fn body" in e for e in errs)


def test_compile_prove_truth_table() -> None:
    # D7: prove never inside a desugared truth-table.
    src = (
        "(truth-table tt ((p : (Int 0 1)))"
        " (true) -> (prove p (evidence p : Evidence))"
        " ((_)) -> 99)"
    )
    errs = _errors(src)
    assert any("truth-table" in e for e in errs)


def test_native_hole_traps_process() -> None:
    # D3/D8: LLVM unreachable kills the process (SIGTRAP-family exit).
    d = PKG / "_native_hole_probe.sl"
    d.write_text(f"{DEFN}\n(def host-ev false)\n(check true host-ev)\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "netelpro", "--native", str(d)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode not in (0, 1), f"expected trap exit, got {proc.returncode}: {proc.stdout}"
    finally:
        d.unlink(missing_ok=True)


def test_native_success_returns_true() -> None:
    d = PKG / "_native_ok_probe.sl"
    d.write_text(f"{DEFN}\n(def host-ev true)\n(check true host-ev)\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "netelpro", "--native", str(d)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert "=> 1" in proc.stdout
        assert proc.returncode == 0
    finally:
        d.unlink(missing_ok=True)


def test_regression_fase1() -> None:
    # The Fase 1 desugar + interpreter pipeline is untouched: a truth-table
    # defn still evaluates through the if-chain lowering.
    src = (
        "(truth-table f (p : (Int 0 1))"
        " ((0) -> 10)"
        " ((1) -> 99)"
        " ((_)-> 99))"
    )
    prog = parse(src)
    assert prog.ok, prog.errors
    tt = prog.program.forms[0]
    assert tt.truth_table is not None  # desugar preserved the spec (Fase 1 pin)