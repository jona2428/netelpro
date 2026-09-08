"""Tarea RLVR: componer un string a partir de dos enteros con separador fijo."""
from __future__ import annotations

import random

TASK_ID = "int_pair_to_string"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos enteros a y b, y devuelva un string "
    "con la representación decimal de a, seguida de un guión '-', seguida "
    "de la representación decimal de b. Ejemplo: a=3, b=-5 -> '3--5'."
)
SIGNATURE = "(defn int-pair-to-string (a b) ...)"
FN_NAME = "int-pair-to-string"


def reference(a: int, b: int) -> str:
    return f"{a}-{b}"


def gen_inputs(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(-50, 50), rng.randint(-50, 50)) for _ in range(n)]


assert reference(3, 7) == "3-7"
assert reference(0, 0) == "0-0"
assert reference(-2, 5) == "-2-5"
