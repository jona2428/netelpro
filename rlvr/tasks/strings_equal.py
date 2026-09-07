"""Tarea RLVR: verificar si dos strings son iguales."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "strings_equal"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos strings y devuelva true si son "
    "exactamente iguales, false si no."
)
SIGNATURE = "(defn strings-equal (a b) ...)"
FN_NAME = "strings-equal"


def reference(a: str, b: str) -> bool:
    return a == b


def gen_inputs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    cases: list[tuple[str, str]] = []
    for _ in range(n):
        a = random_word(rng, 1, 6)
        b = a if rng.random() < 0.5 else random_word(rng, 1, 6)
        cases.append((a, b))
    return cases


assert reference("ab", "ab") is True
assert reference("ab", "ac") is False
