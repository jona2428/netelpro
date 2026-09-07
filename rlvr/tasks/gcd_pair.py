"""Tarea RLVR: máximo común divisor de dos enteros positivos."""
from __future__ import annotations

import math
import random

TASK_ID = "gcd_pair"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros positivos a y b y devuelva "
    "su máximo común divisor (MCD)."
)
SIGNATURE = "(defn gcd-two (a b) ...)"
FN_NAME = "gcd-two"


def reference(a: int, b: int) -> int:
    return math.gcd(a, b)


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 500), rng.randint(1, 500)) for _ in range(n)]


assert reference(12, 18) == 6
assert reference(7, 13) == 1
assert reference(100, 25) == 25
