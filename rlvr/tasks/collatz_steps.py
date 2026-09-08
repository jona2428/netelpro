"""Tarea RLVR: cantidad de pasos de la conjetura de Collatz hasta llegar a 1."""
from __future__ import annotations

import random

TASK_ID = "collatz_steps"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero positivo n y devuelva la "
    "cantidad de pasos de la conjetura de Collatz hasta llegar a 1: en cada "
    "paso, si el número es par se divide por 2, si es impar se multiplica "
    "por 3 y se suma 1. n=1 devuelve 0 pasos."
)
SIGNATURE = "(defn collatz-steps (n) ...)"
FN_NAME = "collatz-steps"


def reference(n: int) -> int:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 30),) for _ in range(n)]


assert reference(1) == 0
assert reference(2) == 1
assert reference(6) == 8
