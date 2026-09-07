"""Tarea RLVR: componer un string con su propio largo (str-cat + int->str + str-len)."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "label_with_length"
DESCRIPTION_ES = (
    "Escribí una función que reciba un string s y devuelva un nuevo string "
    "formado por s seguido de la representación decimal de su largo. "
    "Ejemplo: 'ab' -> 'ab2'."
)
SIGNATURE = "(defn label-with-length (s) ...)"
FN_NAME = "label-with-length"


def reference(s: str) -> str:
    return s + str(len(s))


def gen_inputs(n: int, seed: int) -> list[tuple[str]]:
    rng = random.Random(seed)
    return [(random_word(rng, 1, 8),) for _ in range(n)]


assert reference("ab") == "ab2"
assert reference("") == "0"
