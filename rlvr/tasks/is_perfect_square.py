"""Tarea RLVR: determinar si un entero no negativo es un cuadrado perfecto."""
from __future__ import annotations

import math
import random

TASK_ID = "is_perfect_square"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero n >= 0 y devuelva true si n "
    "es un cuadrado perfecto (existe un entero k tal que k*k = n), false si no."
)
SIGNATURE = "(defn is-perfect-square (n) ...)"
FN_NAME = "is-perfect-square"


def reference(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    cases: list[tuple[int]] = []
    for _ in range(n):
        if rng.random() < 0.5:
            k = rng.randint(0, 20)
            cases.append((k * k,))
        else:
            cases.append((rng.randint(0, 400),))
    return cases


assert reference(0) is True
assert reference(16) is True
assert reference(15) is False
