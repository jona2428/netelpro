"""Tarea RLVR: largo de un string."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "string_length"
DESCRIPTION_ES = "Escribí una función que reciba un string y devuelva su largo (cantidad de caracteres)."
SIGNATURE = "(defn string-length (s) ...)"
FN_NAME = "string-length"


def reference(s: str) -> int:
    return len(s)


def gen_inputs(n: int, seed: int) -> list[tuple[str]]:
    rng = random.Random(seed)
    return [(random_word(rng),) for _ in range(n)]


assert reference("hola") == 4
assert reference("") == 0
