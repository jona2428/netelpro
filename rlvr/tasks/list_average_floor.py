"""Tarea RLVR: promedio entero (redondeo hacia abajo) de una lista no negativa."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_average_floor"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista no vacía de enteros no "
    "negativos y devuelva el promedio de sus elementos, redondeado hacia "
    "abajo (división entera)."
)
SIGNATURE = "(defn list-average-floor (xs) ...)"
FN_NAME = "list-average-floor"


def reference(xs: tuple[int, ...]) -> int:
    return sum(xs) // len(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 6, 0, 20)),) for _ in range(n)]


assert reference((1, 2, 3)) == 2
assert reference((10,)) == 10
assert reference((0, 0, 3)) == 1
