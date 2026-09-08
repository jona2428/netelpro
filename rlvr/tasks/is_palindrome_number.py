"""Tarea RLVR: determinar si un entero no negativo es palíndromo en decimal."""
from __future__ import annotations

import random

TASK_ID = "is_palindrome_number"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero n >= 0 y devuelva true si sus "
    "dígitos son un palíndromo (se leen igual al derecho y al revés), false "
    "si no. Ejemplo: 121 -> true, 123 -> false."
)
SIGNATURE = "(defn is-palindrome-number (n) ...)"
FN_NAME = "is-palindrome-number"


def reference(n: int) -> bool:
    s = str(n)
    return s == s[::-1]


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    cases: list[tuple[int]] = []
    for _ in range(n):
        if rng.random() < 0.5:
            half = str(rng.randint(0, 999))
            cases.append((int(half + half[-2::-1] if len(half) > 1 else half),))
        else:
            cases.append((rng.randint(0, 99999),))
    return cases


assert reference(121) is True
assert reference(123) is False
assert reference(7) is True
