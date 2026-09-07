"""Tarea RLVR: duplicar un entero. La más simple del corpus -- valida el
mecanismo de registro end-to-end antes de sumar el resto de las tareas."""
from __future__ import annotations

import random

TASK_ID = "double_value"
DESCRIPTION_ES = "Escribí una función que reciba un entero y devuelva el doble."
SIGNATURE = "(defn double (x) ...)"
FN_NAME = "double"


def reference(x: int) -> int:
    return x * 2


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(-1000, 1000),) for _ in range(n)]


assert reference(0) == 0
assert reference(5) == 10
assert reference(-3) == -6
