"""Tarea RLVR: cantidad de dígitos de un entero no negativo."""
from __future__ import annotations

import random

TASK_ID = "digit_count"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero n >= 0 y devuelva la cantidad "
    "de dígitos que tiene. Ejemplo: 120 -> 3, 0 -> 1."
)
SIGNATURE = "(defn digit-count (n) ...)"
FN_NAME = "digit-count"


def reference(n: int) -> int:
    return len(str(n))


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 999999),) for _ in range(n)]


assert reference(0) == 1
assert reference(120) == 3
assert reference(999999) == 6
