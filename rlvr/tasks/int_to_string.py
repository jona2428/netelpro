"""Tarea RLVR: convertir un entero a su representación en string."""
from __future__ import annotations

import random

TASK_ID = "int_to_string"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero y devuelva su representación "
    "como string decimal."
)
SIGNATURE = "(defn int-to-string (n) ...)"
FN_NAME = "int-to-string"


def reference(n: int) -> str:
    return str(n)


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(-1000, 1000),) for _ in range(n)]


assert reference(0) == "0"
assert reference(-42) == "-42"
