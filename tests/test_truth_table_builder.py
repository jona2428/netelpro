"""Contract tests for the truth-table builder (spec §8 items 8-9; step 5).

The builder's generated fallback must reproduce the interpreter oracle over
the full declared domain plus out-of-domain probes, with structural
type-strictness (``is True``) closing the 2026-09-07 aliasing bug class.

D1 note: .sl sources here satisfy the exhaustiveness prosecutor (non-default
rows must cover the full declared product; the mandatory all-wildcard default
row never contributes coverage). Renderer-rejection cases construct the
Defn/TruthTableSpec manually to isolate the renderer contract from the
parser's own diagnostics.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

from builders.truth_table_builder import (
    MAX_PRODUCT,
    BuilderError,
    build_truth_table_artifacts,
    oracle_cases,
    render_fallback,
    source_sha8,
)
from netelpro import parse
from netelpro.ast_nodes import (
    BoolLit,
    Call,
    Defn,
    IntLit,
    Let,
    ParamType,
    Sym,
    TruthTableSpec,
)


def _tt(rows: str, params: str = "(a : Bool)") -> str:
    return f"(truth-table f {params}\n{rows})\n"


def _defn_of(src: str):
    prog = parse(src)
    assert not prog.errors, prog.errors
    return prog.program.forms[0]


_BOOL_PARAM = (("a", ParamType("bool")),)


def _fb_ns(text: str) -> dict:
    """Materialize generated text as a real module (importlib, no exec)."""
    with tempfile.TemporaryDirectory() as td:
        return _load_module(text, "gen_fallback", Path(td))


def _load_module(text: str, name: str, directory: Path) -> dict:
    """Materialize generated text as a real module and return its namespace."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gen_artifact.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod).copy()


def _manual_defn(default_expr, params=_BOOL_PARAM):
    """A Defn with a hand-built TruthTableSpec, bypassing the parser."""
    return Defn(
        name=Sym("f"),
        params=[Sym(n) for n, _ in params],
        body=BoolLit(True),
        truth_table=TruthTableSpec(
            params=tuple(params),
            rows=(((True,), IntLit(0)),),
            default_expr=default_expr,
        ),
    )


# ---------------------------------------------------------------------------
# Header / provenance / purity
# ---------------------------------------------------------------------------


def test_fallback_header_pins_provenance() -> None:
    src = _tt("((true) -> 10)\n((false) -> 0)\n((_) -> 0)")
    defn = _defn_of(src)
    text = render_fallback(defn, src, "data/netelpro_gate.sl")
    sha = source_sha8(src)
    assert text.startswith(
        f"# GENERATED FROM data/netelpro_gate.sl AT {sha} BY "
        "netelpro.truth_table_builder -- DO NOT EDIT."
    ), text.splitlines()[0]


def test_source_sha8_is_deterministic_and_sensitive() -> None:
    a = _tt("((true) -> 1)\n((false) -> 0)\n((_) -> 0)")
    b = _tt("((true) -> 2)\n((false) -> 0)\n((_) -> 0)")
    assert source_sha8(a) == source_sha8(a)
    assert source_sha8(a) != source_sha8(b)


def test_fallback_has_zero_netelpro_imports() -> None:
    src = _tt("((true) -> 1)\n((false) -> 0)\n((_) -> 0)")
    text = render_fallback(_defn_of(src), src, "x.sl")
    assert "import" not in text, text  # pure, dependency-free artifact
    assert "netelpro" not in text.replace("netelpro.truth_table_builder", ""), text


def test_builder_cap_matches_prosecutor_discipline() -> None:
    assert MAX_PRODUCT == 256


# ---------------------------------------------------------------------------
# Generated fallback semantics (hand-computed pins)
# ---------------------------------------------------------------------------


def test_fallback_reproduces_hand_computed_semantics() -> None:
    src = _tt(
        "((true true) -> 10)\n((false _) -> 20)\n((true false) -> 15)\n((_ _) -> 30)",
        params="(a : Bool) (b : Bool)",
    )
    ns = _fb_ns(render_fallback(_defn_of(src), src, "x.sl"))
    f = ns["f"]
    assert f(True, True) == 10
    assert f(False, True) == 20  # wildcard in slot 2
    assert f(False, False) == 20
    assert f(True, False) == 15  # dedicated row, not the default


def test_fallback_is_type_strict_on_bools_anti_aliasing_pin() -> None:
    """The 2026-09-07 gate bug class: int 1 conflating with True.

    In the generated artifact a bool slot compiles to ``is True``, so an int
    argument can never satisfy a bool row -- it falls through to the default.
    """
    src = _tt("((true) -> 1)\n((false) -> 0)\n((_) -> 0)")
    ns = _fb_ns(render_fallback(_defn_of(src), src, "x.sl"))
    assert ns["f"](1) == 0  # int 1 does NOT alias True: default, not 1
    assert ns["f"](True) == 1
    assert ns["f"](False) == 0


def test_fallback_first_match_wins_in_declared_order() -> None:
    src = _tt(
        "((true _) -> 1)\n((true true) -> 2)\n((false _) -> 0)\n((_ _) -> 3)",
        params="(a : Bool) (b : Bool)",
    )
    ns = _fb_ns(render_fallback(_defn_of(src), src, "x.sl"))
    assert ns["f"](True, True) == 1  # first declared row wins over row 2
    assert ns["f"](True, False) == 1
    assert ns["f"](False, True) == 0


def test_fallback_ood_int_probe_hits_default() -> None:
    src = _tt("((0) -> 5)\n((1) -> 6)\n((_) -> 99)", params="(n : (Int 0 1))")
    ns = _fb_ns(render_fallback(_defn_of(src), src, "x.sl"))
    assert ns["f"](0) == 5
    assert ns["f"](1) == 6
    assert ns["f"](7) == 99  # out of declared enum domain -> default
    assert ns["f"](True) == 99  # bool never satisfies an Int slot


def test_fallback_int_slot_type_strict() -> None:
    """An Int-enum slot condition rejects bools (Python: True == 1)."""
    src = _tt("((1) -> 11)\n((_) -> 0)", params="(n : (Int 1))")
    ns = _fb_ns(render_fallback(_defn_of(src), src, "x.sl"))
    assert ns["f"](True) == 0  # type(x) is int fails for bool True
    assert ns["f"](1) == 11


def test_fallback_body_expressions_render() -> None:
    src = _tt(
        "((true true) -> 10)\n"
        "((true false) -> (if b 15 14))\n"
        "((false _) -> (if (or a (not b)) 7 8))\n"
        "((_ _) -> 9)",
        params="(a : Bool) (b : Bool)",
    )
    ns = _fb_ns(render_fallback(_defn_of(src), src, "x.sl"))
    f = ns["f"]
    assert f(True, True) == 10
    assert f(True, False) == 14  # (if false 15 14)
    assert f(False, True) == 8  # (or false (not true)) -> else
    assert f(False, False) == 7  # (or false (not false)) -> then


# ---------------------------------------------------------------------------
# Oracle + differential matrix
# ---------------------------------------------------------------------------


def test_oracle_covers_full_domain_and_ood_probe() -> None:
    src = _tt(
        "((true 0) -> 1)\n((true 1) -> 4)\n((false _) -> 2)\n((_ _) -> 3)",
        params="(a : Bool) (n : (Int 0 1))",
    )
    cases = oracle_cases(_defn_of(src), src)
    # 2 bools x 2 ints = 4 declared combos + 1 OOD probe (n=2, bool at min)
    assert len(cases) == 5, sorted(cases)
    assert cases[(True, 0)] == 1
    assert cases[(True, 1)] == 4
    assert cases[(False, 0)] == 2 and cases[(False, 1)] == 2
    assert cases[(True, 2)] == 3  # OOD int -> default (never contributes coverage)


def test_differential_matrix_runs_green() -> None:
    src = _tt(
        "((true true) -> 10)\n((false _) -> 20)\n((true false) -> 15)\n((_ _) -> 30)",
        params="(a : Bool) (b : Bool)",
    )
    defn = _defn_of(src)
    fallback_name = "tt_f_fallback.py"
    fallback, matrix = build_truth_table_artifacts(defn, src, "x.sl", fallback_name)
    assert "4 cases" in matrix, matrix  # 2^2 domain, no Int-enum probes
    tmp = Path(__file__).parent / "_tt_generated_tmp"
    tmp.mkdir(exist_ok=True)
    fb_path = tmp / fallback_name
    mx_path = tmp / "test_tt_matrix_generated.py"
    try:
        fb_path.write_text(fallback, encoding="utf-8")
        mx_path.write_text(matrix, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("tt_matrix_gen", mx_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.test_generated_fallback_matches_build_time_oracle()  # raises on drift
    finally:
        fb_path.unlink(missing_ok=True)
        mx_path.unlink(missing_ok=True)


def test_oracle_rejects_unparseable_source() -> None:
    valid = _tt("((true) -> 1)\n((false) -> 0)\n((_) -> 0)")
    broken = valid + "(defn broken"
    with pytest.raises(BuilderError, match="does not parse"):
        oracle_cases(_defn_of(valid), broken)


# ---------------------------------------------------------------------------
# Closed rendering subset -- nothing renders silently (manual AST construction)
# ---------------------------------------------------------------------------


def test_unsupported_node_type_raises() -> None:
    defn = _manual_defn(Let(name=Sym("x"), value=IntLit(1), body=Sym("x")))
    with pytest.raises(BuilderError, match="Let"):
        render_fallback(defn, "(truth-table f (a : Bool) ((true) -> 1)\n((_) -> 0))", "x.sl")


def test_free_symbol_raises() -> None:
    defn = _manual_defn(Sym("c"))
    with pytest.raises(BuilderError, match="not a table parameter"):
        render_fallback(defn, "(truth-table f (a : Bool) ((true) -> 1)\n((_) -> 0))", "x.sl")


def test_unsupported_call_head_raises() -> None:
    defn = _manual_defn(Call("foo", [IntLit(1), IntLit(2)]))
    with pytest.raises(BuilderError, match="unsupported call 'foo'"):
        render_fallback(defn, "(truth-table f (a : Bool) ((true) -> 1)\n((_) -> 0))", "x.sl")


def test_wrong_arity_supported_head_raises() -> None:
    defn = _manual_defn(Call("+", [IntLit(1), IntLit(2), IntLit(3)]))
    with pytest.raises(BuilderError, match="unsupported call '\\+'"):
        render_fallback(defn, "(truth-table f (a : Bool) ((true) -> 1)\n((_) -> 0))", "x.sl")


def test_non_truth_table_defn_raises() -> None:
    src = "(defn g (x) (+ x 1))\n"
    with pytest.raises(BuilderError, match="no truth_table spec"):
        render_fallback(_defn_of(src), src, "x.sl")