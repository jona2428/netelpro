"""Tarea RLVR: suma de los elementos de una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_sum"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros y devuelva la "
    "suma de sus elementos."
)
SIGNATURE = "(defn list-sum (xs) ...)"
FN_NAME = "list-sum"


def reference(xs: tuple[int, ...]) -> int:
    return sum(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 6, -20, 20)),) for _ in range(n)]


assert reference((1, 2, 3)) == 6
assert reference((-5, 5)) == 0
