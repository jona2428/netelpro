"""Tarea RLVR: largo de una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_length"
DESCRIPTION_ES = "Escribí una función que reciba una lista de enteros y devuelva su largo."
SIGNATURE = "(defn list-length (xs) ...)"
FN_NAME = "list-length"


def reference(xs: tuple[int, ...]) -> int:
    return len(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 8, -20, 20)),) for _ in range(n)]


assert reference((1, 2, 3)) == 3
assert reference((7,)) == 1
