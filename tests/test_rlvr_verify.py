"""Tests del verificador RLVR (rlvr/verify.py): pipeline estático
(parse+capabilities+holes) + ejecución por intérprete, binario -- pasa TODOS
los casos o se descarta. Ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §4."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

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

# ---------------------------------------------------------------------------
# Fix C1 (review final): budget de pasos -- un runaway tail-recursive debe
# abortar acotado, no colgar la sesion de Colab ni el loop de verificacion.
# ---------------------------------------------------------------------------


def test_runaway_tail_recursion_is_bounded_not_hanging():
    """(defn double (x) (double x)) sin caso base: TCO loop infinito.

    Sin budget (pre-fix) esto cuelga indefinidamente. Con StepBudget el
    candidato se descarta con error acotado. Timeout corto de pytest: si el
    fix regresa, el test muere en 10s en vez de colgar la suite entera.
    """
    runaway = "(defn double (x) (double x))"
    result = verify_program(runaway, double_value, num_cases=5, seed=0, max_steps=50_000)
    assert result.compiled is True
    assert result.passed is False
    assert result.cases_passed == 0
    assert result.error is not None
    assert "steps" in result.error


def test_runaway_must_not_hang_the_suite():
    """Guard del guard: el mismo programa runaway pero con budget reducido.

    Si alguien quita el wiring del budget, este test cuelga. El decorador
    de timeout no se usa porque pytest-timeout no esta garantizado; el
    assertion de wall-clock (<10s) hace el trabajo de guard de regresion.
    """
    import time

    from netelpro.evaluator import StepBudget, run_source

    start = time.perf_counter()
    with pytest.raises(Exception):
        run_source("(defn spin (n) (spin n))\n(spin 0)", budget=StepBudget(20_000))
    assert time.perf_counter() - start < 10.0


def test_deep_legitimate_recursion_passes_with_default_budget():
    """count-down(5000) es recursion legitima por TCO: debe completar.

    El default del verificador (1M pasos) debe dejar pasar recursiones
    legtimas profundas sin tocarlas. Verificado manualmente: 5000 iteraciones
    TCO consumen ~30k pasos.
    """
    deep = """
(defn count-down (n)
  (if (== n 0)
      0
      (count-down (- n 1))))
"""

    from netelpro.evaluator import StepBudget, run_source

    result = run_source(deep + "\n(count-down 5000)", budget=StepBudget(1_000_000))
    assert result == 0


# ---------------------------------------------------------------------------
# Fix M1 (review final): comparacion type-strict bool/int.
# ---------------------------------------------------------------------------


def test_bool_returning_candidate_must_return_bool_not_int():
    """Un candidato que devuelve Int donde la referencia devuelve Bool no pasa.

    Python: True == 1. Sin _values_equal, (defn is-positive ...) devolviendo
    1 en vez de true pasaria todos los casos con semantica incorrecta.
    Caso real del corpus: count_positive (referencia devuelve int) vs
    strings_equal / is_prefix (referencia devuelve bool).
    """
    # strings_equal devuelve bool. Un candidato "equivale" que devuelve
    # 1/0 en vez de true/false debe fallar la verificacion.
    from rlvr.tasks import strings_equal

    int_typed = '(defn strings-equal (a b) (if (== (str-len a) (str-len b)) 1 0))'
    result = verify_program(int_typed, strings_equal, num_cases=20, seed=0)
    # OJO: este candidato es incorrecto de todas formas (compara longitudes,
    # no contenido). El punto del test es que NO pase -- y si alguien rompe
    # _values_equal, hay que asegurar que siga sin pasar. Para el caso puro
    # de conflation (== correcto pero tipo int), ver test de abajo.
    assert result.passed is False


def test_true_int_conflation_is_not_equal():
    """Caso puro de conflation: la referencia devuelve True y el candidato 1.

    Python puro: (1 == True) es True. _values_equal debe decir False.
    """
    from rlvr.verify import _values_equal

    assert _values_equal(1, True) is False
    assert _values_equal(0, False) is False
    assert _values_equal(True, True) is True
    assert _values_equal(1, 1) is True
    assert _values_equal((1, 2), (1, 2)) is True
    assert _values_equal((True,), (1,)) is False


# ---------------------------------------------------------------------------
# Fix M2 (review final): branches estaticos de capability/hole sin test.
# ---------------------------------------------------------------------------


def test_capability_violation_rejects_compilation():
    """(print x) sin (grant io): capability violation -> compiled=False.

    Branch nunca testeado del verificador (Task 5 lo dejo cubierto solo
    manualmente). La capacidad 'io' solo la requiere 'print' en v0.1.
    """
    printer = '(defn show (x) (print x))\n(defn double (x) (+ x x))'
    result = verify_program(printer, double_value, num_cases=5, seed=0)
    assert result.compiled is False
    assert result.passed is False
    assert result.cases_total == 0
    assert result.error is not None
    assert "capability" in result.error
    assert "io" in result.error


def test_hole_violation_rejects_compilation():
    """(sorry "reason") como body: compila limpio, falla en runtime.

    Contrato real de holes (netelpro/holes.py docstring #2): sorry es el
    UNICO camino legal de "no implementado" y compila LIMPIO. El verificador
    lo ejecuta, StrayHoleError cae en el except, candidato descartado como
    passed=False con cases_total > 0. Flujo RAFT correcto.
    """
    holed = '(defn double (x) (sorry "not implemented"))'
    result = verify_program(holed, double_value, num_cases=5, seed=0)
    assert result.compiled is True
    assert result.passed is False
    assert result.cases_total == 5
    assert result.cases_passed == 0
    assert result.error is not None


def test_hole_in_dead_branch_runtime_unreachable_still_discards():
    """Sorry en rama muerta: compila, ramas vivas no afectadas.

    La semantica es manifest estatico + fallo runtime. La rama muerta no
    se ejecuta, las ramas vivas evaluan limpio: el candidato PUEDE pasar.
    Este test fija ese contrato (sorry no envenena la compilacion).
    """
    holed_dead = '(defn double (x) (if true (+ x x) (sorry "never reached")))'
    result = verify_program(holed_dead, double_value, num_cases=5, seed=0)
    assert result.compiled is True
    assert result.passed is True
    assert result.cases_passed == 5