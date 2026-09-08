"""Tarea RLVR: determinar si todos los elementos de una lista son positivos."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_all_positive"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista no vacía de enteros y "
    "devuelva true si todos sus elementos son estrictamente positivos "
    "(> 0), false si no."
)
SIGNATURE = "(defn list-all-positive (xs) ...)"
FN_NAME = "list-all-positive"


def reference(xs: tuple[int, ...]) -> bool:
    return all(x > 0 for x in xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 6, -5, 10)),) for _ in range(n)]


assert reference((1, 2, 3)) is True
assert reference((1, -2, 3)) is False
assert reference((5,)) is True
