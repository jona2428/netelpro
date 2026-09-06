"""Unit tests for Netelpro static effect inference and gate rule purity enforcement.

Validates:
(a) Direct effects (primitive calls requiring capabilities).
(b) Transitive effect propagation (f -> g -> print).
(c) Direct recursion (factorial with print converges and includes io).
(d) Mutual recursion (is-even / is-odd with print converges and propagates).
(e) Pure filter-rule produces empty diagnostics.
(f) Impure filter-rule reports EffectError with exact line and column coordinates.
(g) Indirectly impure filter-rule includes the complete call chain in the diagnostic.
(h) Pure defns produce frozenset().
(i) Anonymous nested functions (Fn) attribute effects to the enclosing Defn.
(j) Non-convergence cap raises EffectError("effect inference did not converge").
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netelpro.ast_nodes import Call, Defn, IntLit, Program, Sym  # noqa: E402 - sys.path shim above
from netelpro.effects import (  # noqa: E402 - sys.path shim above
    EffectError,
    check_gate_purity,
    infer_effects,
)
from netelpro.parser import parse  # noqa: E402 - sys.path shim above


class TestInferEffects:
    """Effect inference across direct, transitive, recursive, and nested forms."""

    def test_direct_effect_print_in_defn(self) -> None:
        """(a) efecto directo: print inside defn -> {'io'}."""
        src = """(defn log-msg (x)
  (print x))
"""
        pr = parse(src)
        assert pr.ok
        effects = infer_effects(pr.program)
        assert effects == {"log-msg": frozenset({"io"})}

    def test_transitive_propagation(self) -> None:
        """(b) propagación transitiva: f -> g -> print."""
        src = """(defn sink (x)
  (print x))

(defn helper (x)
  (sink x))

(defn caller (x)
  (helper x))
"""
        pr = parse(src)
        assert pr.ok
        effects = infer_effects(pr.program)
        assert effects == {
            "sink": frozenset({"io"}),
            "helper": frozenset({"io"}),
            "caller": frozenset({"io"}),
        }

    def test_direct_recursion_converges_with_effects(self) -> None:
        """(c) recursión directa: factorial with print converges and includes 'io'."""
        src = """(defn fact (n)
  (if (<= n 1)
      (print 1)
      (* n (fact (- n 1)))))
"""
        pr = parse(src)
        assert pr.ok
        effects = infer_effects(pr.program)
        assert effects == {"fact": frozenset({"io"})}

    def test_mutual_recursion_converges_with_effects(self) -> None:
        """(d) recursión mutua: is-even / is-odd with print in one branch."""
        src = """(defn is-even (n)
  (if (== n 0)
      true
      (is-odd (- n 1))))

(defn is-odd (n)
  (if (== n 0)
      (if (print 0) false false)
      (is-even (- n 1))))
"""
        pr = parse(src)
        assert pr.ok
        effects = infer_effects(pr.program)
        assert effects == {
            "is-even": frozenset({"io"}),
            "is-odd": frozenset({"io"}),
        }

    def test_defn_without_effects_is_empty_frozenset(self) -> None:
        """(h) defn sin efectos -> frozenset() vacío."""
        src = """(defn add (a b)
  (+ a b))

(defn square (x)
  (* x x))

(defn pure-caller (x)
  (add (square x) 1))
"""
        pr = parse(src)
        assert pr.ok
        effects = infer_effects(pr.program)
        assert effects == {
            "add": frozenset(),
            "square": frozenset(),
            "pure-caller": frozenset(),
        }

    def test_nested_fn_attributes_effects_to_enclosing_defn(self) -> None:
        """(i) Fn anónimo anidado: hereda el effect set del defn contenedor."""
        src = """(defn make-logger (prefix)
  (fn (msg)
    (print (str-cat prefix msg))))

(defn make-adder (x)
  (fn (y)
    (+ x y)))
"""
        pr = parse(src)
        assert pr.ok
        effects = infer_effects(pr.program)
        assert effects == {
            "make-logger": frozenset({"io"}),
            "make-adder": frozenset(),
        }

    def test_unknown_heads_and_pure_primitives_ignored(self) -> None:
        """Unknown call heads and pure primitives do not inject phantom effects."""
        # Unknown heads are caught by the fiscal/parser, but infer_effects must ignore them.
        program = Program(
            forms=[
                Defn(
                    name=Sym("worker"),
                    params=[Sym("x")],
                    body=Call("unknown-callee", [Call("+", [Sym("x"), IntLit(10)])]),
                )
            ]
        )
        effects = infer_effects(program)
        assert effects == {"worker": frozenset()}

    def test_non_convergence_cap_raises_effect_error(self) -> None:
        """(j) Non-convergence triggers EffectError('effect inference did not converge')."""
        src = """(defn helper (x)
  (print x))
(defn caller (x)
  (helper x))
"""
        pr = parse(src)
        assert pr.ok
        with pytest.raises(EffectError, match="effect inference did not converge"):
            infer_effects(pr.program, _max_iterations=0)


class TestGatePurity:
    """Gate rule purity enforcement for 'filter-rule'."""

    def test_pure_filter_rule_returns_empty_diagnostics(self) -> None:
        """(e) filter-rule puro -> []."""
        src = """(defn helper (x)
  (> x 0))

(defn filter-rule (item)
  (helper item))
"""
        pr = parse(src)
        assert pr.ok
        diagnostics = check_gate_purity(pr.program)
        assert diagnostics == []

    def test_direct_impure_filter_rule_reports_exact_coordinates(self) -> None:
        """(f) filter-rule impuro directo -> EffectError con coordenadas correctas."""
        src = """(defn filter-rule (x)
  (print x))
"""
        pr = parse(src)
        assert pr.ok
        diagnostics = check_gate_purity(pr.program)
        assert len(diagnostics) == 1
        err = diagnostics[0]
        assert isinstance(err, EffectError)
        assert err.line == 1
        assert err.col == 1
        assert err.message == "gate rule 'filter-rule' must be pure (effects: io via <chain: filter-rule -> print>)"
        assert str(err) == "line 1, col 1: gate rule 'filter-rule' must be pure (effects: io via <chain: filter-rule -> print>)"

    def test_indirect_impure_filter_rule_reports_complete_chain(self) -> None:
        """(g) filter-rule impuro indirecto -> mensaje incluye la CHAIN."""
        src = """(defn sink (x)
  (print x))

(defn helper (x)
  (sink x))

(defn filter-rule (x)
  (helper x))
"""
        pr = parse(src)
        assert pr.ok
        diagnostics = check_gate_purity(pr.program)
        assert len(diagnostics) == 1
        err = diagnostics[0]
        assert err.line == 7
        assert err.col == 1
        expected_msg = "gate rule 'filter-rule' must be pure (effects: io via <chain: filter-rule -> helper -> sink -> print>)"
        assert err.message == expected_msg
        assert str(err) == f"line 7, col 1: {expected_msg}"

    def test_no_filter_rule_present_returns_empty_list(self) -> None:
        """Program without any filter-rule definition returns []."""
        src = """(defn other-rule (x)
  (print x))
"""
        pr = parse(src)
        assert pr.ok
        assert check_gate_purity(pr.program) == []

    def test_effect_error_dataclass_properties(self) -> None:
        """Test EffectError properties and inheritance."""
        err = EffectError(message="rule violated", line=3, col=12)
        assert err.message == "rule violated"
        assert err.line == 3
        assert err.col == 12
        assert str(err) == "line 3, col 12: rule violated"
        assert isinstance(err, Exception)
