"""Tarea RLVR: producto de los enteros entre a y b, inclusive."""
from __future__ import annotations

import random

TASK_ID = "product_range"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros positivos a y b (a <= b) y "
    "devuelva el producto de todos los enteros entre a y b, inclusive."
)
SIGNATURE = "(defn product-range (a b) ...)"
FN_NAME = "product-range"


def reference(a: int, b: int) -> int:
    result = 1
    for i in range(a, b + 1):
        result *= i
    return result


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    cases: list[tuple[int, int]] = []
    for _ in range(n):
        a = rng.randint(1, 6)
        b = a + rng.randint(0, 4)
        cases.append((a, b))
    return cases


assert reference(2, 4) == 24
assert reference(5, 5) == 5
assert reference(1, 3) == 6
