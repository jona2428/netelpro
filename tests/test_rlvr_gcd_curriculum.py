"""Lever run #3: curriculum para gcd_pair (2026-09-08).

gcd_pair (OOD) quedó 0/8 en runs #1/#2 (ver README, hipótesis refutada del
pool acumulado): la habilidad faltante es recursión de dos argumentos con
reorden de parámetros, que ninguna tarea train ejercita. Este módulo fija el
contrato del curriculum: 3 tareas TRAIN (no OOD) solubles, gradúadas por el
verificador RLVR real, con el corpus en 58 (38 train + 20 OOD intacto).
"""
from __future__ import annotations

from rlvr.tasks import (
    OOD_TASK_IDS,
    TASK_MODULE_NAMES,
    load_task,
    split_train_ood,
)
from rlvr.verify import verify_program

CURRICULUM = ["halving_steps", "euclid_steps", "sub_gcd_steps"]

SOLUTIONS = {
    "halving_steps": (
        "(defn halving-steps (n)\n"
        "  (if (< n 2)\n"
        "      0\n"
        "      (+ 1 (halving-steps (quot n 2)))))"
    ),
    "euclid_steps": (
        "(defn euclid-steps (a b)\n"
        "  (if (== b 0)\n"
        "      0\n"
        "      (+ 1 (euclid-steps b (rem a b)))))"
    ),
    "sub_gcd_steps": (
        "(defn sub-gcd-steps (a b)\n"
        "  (if (== a b)\n"
        "      0\n"
        "      (if (> a b)\n"
        "          (+ 1 (sub-gcd-steps (- a b) b))\n"
        "          (+ 1 (sub-gcd-steps a (- b a))))))"
    ),
}


def test_curriculum_registered_in_train_not_ood():
    train, _ood = split_train_ood(list(TASK_MODULE_NAMES))
    assert len(TASK_MODULE_NAMES) == 58
    assert len(OOD_TASK_IDS) == 20
    assert len(train) == 38
    assert "gcd_pair" in OOD_TASK_IDS  # el OOD no se toca
    for tid in CURRICULUM:
        assert tid in TASK_MODULE_NAMES
        assert tid in train
        assert tid not in OOD_TASK_IDS


def test_curriculum_tasks_follow_contract():
    for tid in CURRICULUM:
        module = load_task(tid)  # valida REQUIRED_ATTRS
        cases = module.gen_inputs(5, 0)
        assert len(cases) == 5
        assert all(
            len(args) == module.reference.__code__.co_argcount for args in cases
        )


def test_curriculum_solvable_and_verifier_graded():
    """Soluciones Netelpro escritas a mano pasan por el verificador real:
    prueba que las tareas son solubles y que RAFT puede graduarlas."""
    for tid in CURRICULUM:
        result = verify_program(SOLUTIONS[tid], load_task(tid), num_cases=20, seed=0)
        assert result.compiled, f"{tid} no compiló: {result.error}"
        assert result.passed, f"{tid} no pasó: {result.error}"
        assert result.cases_total == result.cases_passed == 20