"""Tarea RLVR (curriculum gcd): pasos del gcd por restas sucesivas."""
from __future__ import annotations

import random

TASK_ID = "sub_gcd_steps"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros positivos a y b y devuelva "
    "cuántas restas hace el método de Euclides por restas: si a es mayor que "
    "b, reemplazá a por a-b; si no, reemplazá b por b-a; repetí hasta que a "
    "y b sean iguales y contá cuántas restas hiciste."
)
SIGNATURE = "(defn sub-gcd-steps (a b) ...)"
FN_NAME = "sub-gcd-steps"


def reference(a: int, b: int) -> int:
    steps = 0
    while a != b:
        if a > b:
            a -= b
        else:
            b -= a
        steps += 1
    return steps


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 200), rng.randint(1, 200)) for _ in range(n)]


assert reference(12, 18) == 2  # (12,18)->(12,6)->(6,6)
assert reference(7, 13) == 7  # (7,13)->(7,6)->(1,6)->(1,5)->(1,4)->(1,3)->(1,2)->(1,1)
assert reference(25, 25) == 0