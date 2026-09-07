"""Tarea RLVR: verificar si un valor está en una lista de enteros."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_int_list

TASK_ID = "list_contains"
DESCRIPTION_ES = (
    "Escribí una función que reciba una lista de enteros xs y un entero "
    "target, y devuelva true si target está en xs, false si no."
)
SIGNATURE = "(defn list-contains (xs target) ...)"
FN_NAME = "list-contains"


def reference(xs: tuple[int, ...], target: int) -> bool:
    return target in xs


def gen_inputs(n: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    rng = random.Random(seed)
    cases: list[tuple[tuple[int, ...], int]] = []
    for _ in range(n):
        xs = tuple(random_int_list(rng, 1, 6, 0, 10))
        target = rng.choice(xs) if (rng.random() < 0.5 and xs) else rng.randint(-5, 15)
        cases.append((xs, target))
    return cases


assert reference((1, 2, 3), 2) is True
assert reference((1, 2, 3), 9) is False
