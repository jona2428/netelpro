"""Tarea RLVR: contar elementos pares en una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "count_even"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y devuelva "
    "cuántos de sus elementos son pares."
)
SIGNATURE = "(defn count-even (xs) ...)"
FN_NAME = "count-even"


def reference(xs: tuple[int, ...]) -> int:
    return sum(1 for x in xs if x % 2 == 0)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 0, 8, -20, 20)),) for _ in range(n)]


assert reference((1, 2, 3, 4)) == 2
assert reference((1, 3, 5)) == 0
assert reference(()) == 0
