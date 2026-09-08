"""Tarea RLVR: primer índice de un valor en una lista de enteros, o -1."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_index_of"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y un entero "
    "target, y devuelva el índice (desde 0) de la primera aparición de "
    "target en xs, o -1 si no está."
)
SIGNATURE = "(defn list-index-of (xs target) ...)"
FN_NAME = "list-index-of"


def reference(xs: tuple[int, ...], target: int) -> int:
    return xs.index(target) if target in xs else -1


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    rng = random.Random(seed)
    cases: list[tuple[tuple[int, ...], int]] = []
    for _ in range(n):
        xs = tuple(random_int_list(rng, 1, 6, 0, 10))
        target = rng.choice(xs) if (rng.random() < 0.5 and xs) else rng.randint(-5, 15)
        cases.append((xs, target))
    return cases


assert reference((5, 6, 7), 6) == 1
assert reference((5, 6, 7), 9) == -1
assert reference((1, 1, 2), 1) == 0
