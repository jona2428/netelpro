"""Tarea RLVR: cantidad de divisores positivos de un entero positivo."""
from __future__ import annotations

import random

TASK_ID = "count_divisors"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero positivo n y devuelva la "
    "cantidad de divisores positivos de n (incluyendo 1 y n)."
)
SIGNATURE = "(defn count-divisors (n) ...)"
FN_NAME = "count-divisors"


def reference(n: int) -> int:
    return sum(1 for d in range(1, n + 1) if n % d == 0)


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 200),) for _ in range(n)]


assert reference(1) == 1
assert reference(12) == 6
assert reference(13) == 2
