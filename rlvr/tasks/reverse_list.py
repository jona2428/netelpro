"""Tarea RLVR: invertir una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "reverse_list"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros y devuelva la "
    "misma lista invertida."
)
SIGNATURE = "(defn reverse-list (xs) ...)"
FN_NAME = "reverse-list"


def reference(xs: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(xs))


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 6, -20, 20)),) for _ in range(n)]


assert reference((1, 2, 3)) == (3, 2, 1)
assert reference((5,)) == (5,)
