"""Tarea RLVR: producto de los elementos de una lista de enteros."""
from __future__ import annotations

import math
import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_product"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista no vacía de enteros y "
    "devuelva el producto de todos sus elementos."
)
SIGNATURE = "(defn list-product (xs) ...)"
FN_NAME = "list-product"


def reference(xs: tuple[int, ...]) -> int:
    return math.prod(xs)


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...]]]:
    rng = random.Random(seed)
    return [(tuple(random_int_list(rng, 1, 5, -4, 4)),) for _ in range(n)]


assert reference((1, 2, 3)) == 6
assert reference((-2, 3)) == -6
assert reference((5,)) == 5
