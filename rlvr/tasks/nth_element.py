"""Tarea RLVR: elemento en una posición (0-indexada) de una lista."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "nth_element"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y un índice "
    "idx (0-indexado, siempre válido para xs) y devuelva el elemento en "
    "esa posición."
)
SIGNATURE = "(defn nth-element (xs idx) ...)"
FN_NAME = "nth-element"


def reference(xs: tuple[int, ...], idx: int) -> int:
    return xs[idx]


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    rng = random.Random(seed)
    cases: list[tuple[tuple[int, ...], int]] = []
    for _ in range(n):
        xs = tuple(random_int_list(rng, 1, 8, -20, 20))
        idx = rng.randint(0, len(xs) - 1)
        cases.append((xs, idx))
    return cases


assert reference((10, 20, 30), 0) == 10
assert reference((10, 20, 30), 2) == 30
