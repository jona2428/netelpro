"""Tarea RLVR (curriculum gcd): pasos del algoritmo de Euclides con reorden."""
from __future__ import annotations

import random

TASK_ID = "euclid_steps"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros positivos a y b y devuelva "
    "cuántos pasos hace el algoritmo de Euclides: reemplazar (a, b) por "
    "(b, resto de dividir a entre b) mientras b sea distinto de 0, y contar "
    "cuántos reemplazos hizo hasta que b llegó a 0."
)
SIGNATURE = "(defn euclid-steps (a b) ...)"
FN_NAME = "euclid-steps"


def reference(a: int, b: int) -> int:
    steps = 0
    while b != 0:
        a, b = b, a % b
        steps += 1
    return steps


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 500), rng.randint(1, 500)) for _ in range(n)]


assert reference(12, 18) == 3  # (12,18)->(18,12)->(12,6)->(6,0)
assert reference(7, 13) == 4  # (7,13)->(13,7)->(7,6)->(6,1)->(1,0)
assert reference(100, 25) == 1  # (100,25)->(25,0)