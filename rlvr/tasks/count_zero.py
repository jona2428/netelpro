"""Tarea RLVR: contar ceros en una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "count_zero"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y devuelva "
    "cuántos de sus elementos son iguales a 0."
)
SIGNATURE = "(defn count-zero (xs) ...)"
FN_NAME = "count-zero"


def reference(xs: tuple[int, ...]) -> int:
    return sum(1 for x in xs if x == 0)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 0, 8, -3, 3)),) for _ in range(n)]


assert reference((0, 1, 0, 2)) == 2
assert reference((1, 2, 3)) == 0
assert reference(()) == 0
