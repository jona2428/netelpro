"""Tarea RLVR: invertir los dígitos de un entero no negativo."""
from __future__ import annotations

import random

TASK_ID = "reverse_digits"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero n >= 0 y devuelva el entero "
    "formado por sus dígitos en orden inverso. Ejemplo: 120 -> 21."
)
SIGNATURE = "(defn reverse-digits (n) ...)"
FN_NAME = "reverse-digits"


def reference(n: int) -> int:
    return int(str(n)[::-1])


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 99999),) for _ in range(n)]


assert reference(120) == 21
assert reference(7) == 7
assert reference(0) == 0
