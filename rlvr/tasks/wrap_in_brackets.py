"""Tarea RLVR: envolver un string entre corchetes."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "wrap_in_brackets"
DESCRIPTION_ES = (
    "Escribí una función que reciba un string s y devuelva un nuevo string "
    "igual a s pero con '[' antes y ']' después. Ejemplo: 'ab' -> '[ab]'."
)
SIGNATURE = "(defn wrap-in-brackets (s) ...)"
FN_NAME = "wrap-in-brackets"


def reference(s: str) -> str:
    return "[" + s + "]"


def gen_inputs(n: int, seed: int) -> list[tuple[str]]:
    rng = random.Random(seed)
    return [(random_word(rng, 0, 6),) for _ in range(n)]


assert reference("ab") == "[ab]"
assert reference("") == "[]"
