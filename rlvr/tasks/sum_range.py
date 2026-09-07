"""Tarea RLVR: suma de 1 a n (inclusive)."""
from __future__ import annotations

import random

TASK_ID = "sum_range"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero no negativo n y devuelva la "
    "suma de todos los enteros de 1 a n (inclusive). sum-range(0) es 0."
)
SIGNATURE = "(defn sum-range (n) ...)"
FN_NAME = "sum-range"


def reference(n: int) -> int:
    return n * (n + 1) // 2


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 200),) for _ in range(n)]


assert reference(0) == 0
assert reference(1) == 1
assert reference(10) == 55
