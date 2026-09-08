"""Tests de netelpro/gate.py: el Gate generico fail-closed extraido del
patron de guards de Neuromancer (propuestas 1+5 de IDEAS_APLICACIONES).

Contract verificado aca:
- check() nunca levanta: cualquier fallo -> (False, razon explicita);
  reason None <=> la regla compilo y decidio (True o False).
- Differential: Gate.decide == RuleFilter.decide sobre la misma regla .sl.
- Fail-closed: regla ausente, error de compilacion, aridad invalida,
  sorry hole.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from netelpro.gate import Gate, GateError, check_file, open_gate  # noqa: E402
from netelpro.rule_filter import RuleFilter  # noqa: E402

_LE_INT = "(defn filter-rule (amount limit) (<= amount limit))"
_ADMIN_AGE = "(defn filter-rule (admin age) (if admin true (>= age 18)))"
_SORRY = '(defn filter-rule (x) (sorry "pending: table not signed off"))'


def _write_rule(tmp_path: Path, source: str, name: str = "rule.sl") -> Path:
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Decisiciones reales: (True/False, None)
# ---------------------------------------------------------------------------


def test_check_allows_within_limit(tmp_path: Path):
    gate = Gate(_write_rule(tmp_path, _LE_INT))
    assert gate.check(5, 10) == (True, None)
    assert gate.check(10, 10) == (True, None)


def test_check_denies_beyond_limit_is_real_decision(tmp_path: Path):
    """Denegado POR LA REGLA: (False, None), distinto de un fallo."""
    gate = Gate(_write_rule(tmp_path, _LE_INT))
    assert gate.check(11, 10) == (False, None)


def test_check_bool_rule(tmp_path: Path):
    gate = Gate(_write_rule(tmp_path, _ADMIN_AGE))
    assert gate.check(True, 17) == (True, None)
    assert gate.check(False, 17) == (False, None)
    assert gate.check(False, 20) == (True, None)  # else: (>= 20 18) -> true
    assert gate.check(False, 18) == (True, None)  # borde exacto


# ---------------------------------------------------------------------------
# Differential: Gate == RuleFilter sobre la misma fuente .sl
# ---------------------------------------------------------------------------


def test_differential_decide_matches_rulefilter_int_rule(tmp_path: Path):
    rule = _write_rule(tmp_path, _LE_INT)
    gate = Gate(rule)
    rf = RuleFilter(_LE_INT, "filter-rule")
    cases = [(0, 0), (1, 5), (5, 10), (10, 10), (11, 10), (-3, 2), (999, 1000)]
    for args in cases:
        assert gate.check(*args) == (rf.decide(*args), None)


def test_differential_decide_matches_rulefilter_bool_rule(tmp_path: Path):
    gate = Gate(_write_rule(tmp_path, _ADMIN_AGE))
    rf = RuleFilter(_ADMIN_AGE, "filter-rule")
    cases = [(True, 17), (True, 18), (False, 20), (False, 17), (False, 18)]
    for args in cases:
        assert gate.check(*args) == (rf.decide(*args), None)


def test_manifest_passthrough_clean_rule(tmp_path: Path):
    gate = Gate(_write_rule(tmp_path, _LE_INT))
    assert gate.manifest() == RuleFilter(_LE_INT, "filter-rule").manifest()


def test_open_gate_returns_gate(tmp_path: Path):
    gate = open_gate(_write_rule(tmp_path, _LE_INT))
    assert isinstance(gate, Gate)


# ---------------------------------------------------------------------------
# Fail-closed: regla ausente
# ---------------------------------------------------------------------------


def test_gate_missing_file_raises_gate_error(tmp_path: Path):
    with pytest.raises(GateError, match="not readable"):
        Gate(tmp_path / "nope.sl")


def test_check_file_missing_rule_denies_with_reason(tmp_path: Path):
    allow, reason = check_file(str(tmp_path / "nope.sl"), 1)
    assert allow is False
    assert reason is not None and "gate unavailable" in reason


# ---------------------------------------------------------------------------
# Fail-closed: error de compilacion
# ---------------------------------------------------------------------------


def test_gate_compile_error_raises_gate_error(tmp_path: Path):
    _write_rule(tmp_path, "(defn filter-rule (x) (+ x")
    with pytest.raises(GateError, match="compile failed"):
        Gate(tmp_path / "rule.sl")


def test_check_file_compile_error_denies_with_reason(tmp_path: Path):
    _write_rule(tmp_path, "(defn filter-rule (x) (+ x")
    allow, reason = check_file(str(tmp_path / "rule.sl"), 1)
    assert allow is False
    assert reason is not None and "compile failed" in reason


# ---------------------------------------------------------------------------
# Fail-closed: mal uso del gate (aridad / sorry hole)
# ---------------------------------------------------------------------------


def test_check_arity_misuse_denies_with_reason(tmp_path: Path):
    """Regla de 2 params invocada con 1 arg: denial explicito, no crash."""
    gate = Gate(_write_rule(tmp_path, _LE_INT))
    allow, reason = gate.check(5)
    assert allow is False
    assert reason is not None and "gate" in reason


def test_check_sorry_hole_denies_with_reason(tmp_path: Path):
    """Un sorry declarado compila pero decide falla: fail-closed con razon."""
    gate = Gate(_write_rule(tmp_path, _SORRY))
    assert len(gate.manifest()) >= 1
    allow, reason = gate.check(1)
    assert allow is False
    assert reason is not None


def test_check_file_full_cycle_fail_closed(tmp_path: Path):
    """Un host externo: una sola llamada, nunca levanta, siempre decide."""
    rule = _write_rule(tmp_path, _LE_INT)
    assert check_file(str(rule), 7, 10) == (True, None)
    assert check_file(str(rule), 70, 10) == (False, None)
    assert check_file(str(tmp_path / "gone.sl"), 7, 10)[0] is False