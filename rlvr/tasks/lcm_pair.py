"""Tarea RLVR: mínimo común múltiplo de dos enteros positivos."""
from __future__ import annotations

import math
import random

TASK_ID = "lcm_pair"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros positivos a y b y devuelva "
    "su mínimo común múltiplo (MCM)."
)
SIGNATURE = "(defn lcm-two (a b) ...)"
FN_NAME = "lcm-two"


def reference(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 40), rng.randint(1, 40)) for _ in range(n)]


assert reference(4, 6) == 12
assert reference(7, 13) == 91
assert reference(5, 5) == 5
