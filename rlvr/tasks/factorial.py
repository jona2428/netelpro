"""Tarea RLVR: factorial de un entero no negativo."""
from __future__ import annotations

import math
import random

TASK_ID = "factorial"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero no negativo n y devuelva n! "
    "(factorial). Asumí que la entrada siempre es >= 0."
)
SIGNATURE = "(defn factorial (n) ...)"
FN_NAME = "factorial"


def reference(n: int) -> int:
    return math.factorial(n)


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 12),) for _ in range(n)]


assert reference(0) == 1
assert reference(5) == 120
assert reference(10) == 3628800
