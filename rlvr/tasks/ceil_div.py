"""Tarea RLVR: división entera hacia arriba (ceiling division)."""
from __future__ import annotations

import random

TASK_ID = "ceil_div"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros no negativos a y b (b > 0) "
    "y devuelva el resultado de a dividido b, redondeado hacia arriba "
    "(ceiling division). Ejemplo: a=7, b=2 -> 4."
)
SIGNATURE = "(defn ceil-div (a b) ...)"
FN_NAME = "ceil-div"


def reference(a: int, b: int) -> int:
    return -(-a // b)


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 200), rng.randint(1, 20)) for _ in range(n)]


assert reference(7, 2) == 4
assert reference(8, 2) == 4
assert reference(0, 5) == 0
