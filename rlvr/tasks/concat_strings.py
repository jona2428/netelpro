"""Tarea RLVR: concatenar dos strings."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "concat_strings"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos strings y devuelva su "
    "concatenación (a seguido de b)."
)
SIGNATURE = "(defn concat-two (a b) ...)"
FN_NAME = "concat-two"


def reference(a: str, b: str) -> str:
    return a + b


def gen_inputs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(random_word(rng), random_word(rng)) for _ in range(n)]


assert reference("ab", "cd") == "abcd"
assert reference("", "x") == "x"
