"""Tarea RLVR: contar ocurrencias de un valor en una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "count_occurrences"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y un entero "
    "target, y devuelva cuántas veces aparece target en xs."
)
SIGNATURE = "(defn count-occurrences (xs target) ...)"
FN_NAME = "count-occurrences"


def reference(xs: tuple[int, ...], target: int) -> int:
    return xs.count(target)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    rng = random.Random(seed)
    cases: list[tuple[tuple[int, ...], int]] = []
    for _ in range(n):
        xs = tuple(random_int_list(rng, 1, 8, 0, 5))
        target = rng.randint(0, 5)
        cases.append((xs, target))
    return cases


assert reference((1, 2, 1, 3, 1), 1) == 3
assert reference((1, 2, 3), 9) == 0
assert reference((), 1) == 0
