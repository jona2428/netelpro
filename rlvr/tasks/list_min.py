"""Tarea RLVR: mínimo de una lista de enteros no vacía."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_min"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista no vacía de enteros y "
    "devuelva el menor de sus elementos."
)
SIGNATURE = "(defn list-min (xs) ...)"
FN_NAME = "list-min"


def reference(xs: tuple[int, ...]) -> int:
    return min(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 6, -20, 20)),) for _ in range(n)]


assert reference((3, 1, 2)) == 1
assert reference((-5, 5)) == -5
assert reference((7,)) == 7
