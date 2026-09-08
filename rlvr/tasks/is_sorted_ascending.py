"""Tarea RLVR: determinar si una lista de enteros está ordenada ascendente."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "is_sorted_ascending"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y devuelva true "
    "si está ordenada de forma ascendente (cada elemento <= al siguiente), "
    "false si no. Una lista vacía o de un solo elemento se considera "
    "ordenada."
)
SIGNATURE = "(defn is-sorted-ascending (xs) ...)"
FN_NAME = "is-sorted-ascending"


def reference(xs: tuple[int, ...]) -> bool:
    return list(xs) == sorted(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    cases: list[tuple[tuple[int, ...]]] = []
    for _ in range(n):
        xs = random_int_list(rng, 0, 6, -15, 15)
        if rng.random() < 0.5:
            xs = sorted(xs)
        cases.append((tuple(xs),))
    return cases


assert reference((1, 2, 3)) is True
assert reference((3, 1, 2)) is False
assert reference(()) is True
