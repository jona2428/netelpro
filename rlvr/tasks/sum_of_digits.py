"""Tarea RLVR: suma de los dígitos de un entero no negativo."""
from __future__ import annotations

import random

TASK_ID = "sum_of_digits"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero no negativo n y devuelva la "
    "suma de sus dígitos en base 10."
)
SIGNATURE = "(defn digit-sum (n) ...)"
FN_NAME = "digit-sum"


def reference(n: int) -> int:
    return sum(int(ch) for ch in str(n))


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, 999999),) for _ in range(n)]


assert reference(0) == 0
assert reference(123) == 6
assert reference(999) == 27
