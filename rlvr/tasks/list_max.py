"""Tarea RLVR: máximo elemento de una lista de enteros no vacía."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_max"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros NO vacía y "
    "devuelva su elemento máximo."
)
SIGNATURE = "(defn list-max (xs) ...)"
FN_NAME = "list-max"


def reference(xs: tuple[int, ...]) -> int:
    return max(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 6, -20, 20)),) for _ in range(n)]


assert reference((1, 5, 3)) == 5
assert reference((-1, -5, -3)) == -1
