"""Tarea RLVR: concatenar tres strings en orden."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "concat_three"
DESCRIPTION_ES = (
    "Escribí una función que reciba tres strings a, b, c y devuelva su "
    "concatenación en ese orden."
)
SIGNATURE = "(defn concat-three (a b c) ...)"
FN_NAME = "concat-three"


def reference(a: str, b: str, c: str) -> str:
    return a + b + c


def gen_inputs(n: int, seed: int) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    return [
        (random_word(rng), random_word(rng), random_word(rng)) for _ in range(n)
    ]


assert reference("a", "b", "c") == "abc"
assert reference("", "", "x") == "x"
