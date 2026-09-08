"""Tarea RLVR: agregar un entero al final de una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "append_at_end"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y un entero x, "
    "y devuelva una nueva lista igual a xs con x agregado al final."
)
SIGNATURE = "(defn append-at-end (xs x) ...)"
FN_NAME = "append-at-end"


def reference(xs: tuple[int, ...], x: int) -> tuple[int, ...]:
    return xs + (x,)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    rng = random.Random(seed)
    cases: list[tuple[tuple[int, ...], int]] = []
    for _ in range(n):
        xs = tuple(random_int_list(rng, 0, 5, -10, 10))
        x = rng.randint(-10, 10)
        cases.append((xs, x))
    return cases


assert reference((1, 2), 3) == (1, 2, 3)
assert reference((), 5) == (5,)
