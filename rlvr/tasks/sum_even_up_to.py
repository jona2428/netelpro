"""Tarea RLVR: suma de los números pares entre 0 y n, inclusive."""
from __future__ import annotations

import random

TASK_ID = "sum_even_up_to"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero n >= 0 y devuelva la suma de "
    "todos los números pares entre 0 y n, inclusive."
)
SIGNATURE = "(defn sum-even-up-to (n) ...)"
FN_NAME = "sum-even-up-to"


def reference(n: int) -> int:
    return sum(i for i in range(0, n + 1) if i % 2 == 0)


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 100),) for _ in range(n)]


assert reference(0) == 0
assert reference(4) == 6
assert reference(5) == 6
