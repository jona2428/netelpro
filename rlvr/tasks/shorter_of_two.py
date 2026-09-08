"""Tarea RLVR: el string más corto entre dos, con empate a favor del primero."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "shorter_of_two"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos strings a y b, y devuelva el que "
    "tenga menos caracteres. Si tienen el mismo largo, devolvé a."
)
SIGNATURE = "(defn shorter-of-two (a b) ...)"
FN_NAME = "shorter-of-two"


def reference(a: str, b: str) -> str:
    return a if len(a) <= len(b) else b


def gen_inputs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(random_word(rng, 1, 8), random_word(rng, 1, 8)) for _ in range(n)]


assert reference("hola", "hi") == "hi"
assert reference("hi", "hola") == "hi"
assert reference("ab", "cd") == "ab"
