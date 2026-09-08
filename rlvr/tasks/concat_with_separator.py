"""Tarea RLVR: concatenar dos strings con un separador entre medio."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "concat_with_separator"
DESCRIPTION_ES = (
    "Escribí una función que reciba tres strings a, sep y b, y devuelva "
    "a seguido de sep seguido de b."
)
SIGNATURE = "(defn concat-with-separator (a sep b) ...)"
FN_NAME = "concat-with-separator"


def reference(a: str, sep: str, b: str) -> str:
    return a + sep + b


def gen_inputs(n: int, seed: int) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    return [
        (random_word(rng, 1, 5), random_word(rng, 1, 2), random_word(rng, 1, 5))
        for _ in range(n)
    ]


assert reference("a", "-", "b") == "a-b"
assert reference("", ",", "x") == ",x"
