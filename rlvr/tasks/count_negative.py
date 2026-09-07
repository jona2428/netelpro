"""Tarea RLVR: cantidad de elementos negativos (< 0) en una lista."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "count_negative"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros y devuelva "
    "cuántos elementos son estrictamente menores que 0."
)
SIGNATURE = "(defn count-negative (xs) ...)"
FN_NAME = "count-negative"


def reference(xs: tuple[int, ...]) -> int:
    return sum(1 for x in xs if x < 0)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 8, -10, 10)),) for _ in range(n)]


assert reference((1, -2, 3)) == 1
assert reference((1, 2)) == 0
