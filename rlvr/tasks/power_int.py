"""Tarea RLVR: potencia entera (base elevado a exponente no negativo)."""
from __future__ import annotations

import random

TASK_ID = "power_int"
DESCRIPTION_ES = (
    "Escribí una función que reciba una base entera y un exponente entero "
    "no negativo, y devuelva base elevado a exponente. power(x, 0) es 1."
)
SIGNATURE = "(defn power (base exp) ...)"
FN_NAME = "power"


def reference(base: int, exp: int) -> int:
    return base**exp


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(-5, 5), rng.randint(0, 6)) for _ in range(n)]


assert reference(2, 10) == 1024
assert reference(3, 0) == 1
assert reference(-2, 3) == -8
