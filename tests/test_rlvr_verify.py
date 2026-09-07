"""Tests del verificador RLVR (rlvr/verify.py): pipeline estático
(parse+capabilities+holes) + ejecución por intérprete, binario -- pasa TODOS
los casos o se descarta. Ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §4."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlvr.tasks import double_value, reverse_list  # noqa: E402
from rlvr.verify import verify_program  # noqa: E402

_GOOD_DOUBLE = "(defn double (x) (+ x x))"

_WRONG_DOUBLE = "(defn double (x) (+ x 1))"  # compila, pero da la respuesta incorrecta

_SYNTAX_ERROR = "(defn double (x) (+ x x)"  # falta un paréntesis de cierre

_GOOD_REVERSE = """
(defn reverse-acc (xs acc)
  (if (is-nil xs)
      acc
      (reverse-acc (tail xs) (cons (head xs) acc))))

(defn reverse-list (xs)
  (reverse-acc xs (list)))
"""


def test_correct_arithmetic_program_passes_all_cases():
    result = verify_program(_GOOD_DOUBLE, double_value, num_cases=10, seed=0)
    assert result.compiled is True
    assert result.passed is True
    assert result.cases_passed == result.cases_total == 10
    assert result.error is None


def test_wrong_program_compiles_but_fails_some_cases():
    result = verify_program(_WRONG_DOUBLE, double_value, num_cases=10, seed=0)
    assert result.compiled is True
    assert result.passed is False
    assert result.cases_passed < result.cases_total
    assert result.error is not None


def test_syntax_error_program_never_compiles():
    result = verify_program(_SYNTAX_ERROR, double_value, num_cases=10, seed=0)
    assert result.compiled is False
    assert result.passed is False
    assert result.cases_total == 0
    assert result.error is not None


def test_correct_list_program_passes_all_cases():
    result = verify_program(_GOOD_REVERSE, reverse_list, num_cases=10, seed=0)
    assert result.compiled is True
    assert result.passed is True
    assert result.cases_passed == result.cases_total == 10


def test_passing_requires_every_case_not_just_some():
    # NOTA: double_value.gen_inputs(20, seed=0) no genera ningun x == 0 --
    # verificado directamente (rng.randint(-1000, 1000) x20 con esa semilla no
    # toca cero) -- asi que un candidato "correcto solo en cero" nunca pasaria
    # ni un caso con esta semilla (cases_passed quedaria en 0, no en un
    # partial pass). Se usa en cambio un predicado con corte en signo
    # (correcto solo para x > 0), que garantiza mezcla de pass/fail contra los
    # 20 casos reales de esta semilla (hay valores positivos y negativos).
    only_positive = "(defn double (x) (if (> x 0) (+ x x) 999999))"
    result = verify_program(only_positive, double_value, num_cases=20, seed=0)
    assert result.compiled is True
    assert result.passed is False
    assert 0 < result.cases_passed < result.cases_total
