"""Tarea RLVR: máximo de tres enteros."""
from __future__ import annotations

import random

TASK_ID = "max_of_three"
DESCRIPTION_ES = (
    "Escribí una función que reciba tres enteros a, b, c y devuelva el "
    "mayor de los tres."
)
SIGNATURE = "(defn max-of-three (a b c) ...)"
FN_NAME = "max-of-three"


def reference(a: int, b: int, c: int) -> int:
    return max(a, b, c)


def gen_inputs(n: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    return [
        (rng.randint(-50, 50), rng.randint(-50, 50), rng.randint(-50, 50))
        for _ in range(n)
    ]


assert reference(1, 2, 3) == 3
assert reference(-1, -2, -3) == -1
assert reference(5, 5, 5) == 5
