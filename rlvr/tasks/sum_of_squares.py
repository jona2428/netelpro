"""Tarea RLVR: suma de los cuadrados de los elementos de una lista."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "sum_of_squares"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros y devuelva la "
    "suma de los cuadrados de sus elementos."
)
SIGNATURE = "(defn sum-of-squares (xs) ...)"
FN_NAME = "sum-of-squares"


def reference(xs: tuple[int, ...]) -> int:
    return sum(x * x for x in xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 5, -10, 10)),) for _ in range(n)]


assert reference((1, 2, 3)) == 14
assert reference((-2,)) == 4
