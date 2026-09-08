"""Tarea RLVR: repetir un string n veces."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "string_repeat_n"
DESCRIPTION_ES = (
    "Escribí una función que reciba un string s y un entero n >= 0, y "
    "devuelva s repetido n veces concatenado. Ejemplo: s='ab', n=3 -> "
    "'ababab'. n=0 devuelve el string vacío."
)
SIGNATURE = "(defn string-repeat-n (s n) ...)"
FN_NAME = "string-repeat-n"


def reference(s: str, n: int) -> str:
    return s * n


def gen_inputs(n: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    return [(random_word(rng, 1, 4), rng.randint(0, 4)) for _ in range(n)]


assert reference("ab", 3) == "ababab"
assert reference("x", 0) == ""
assert reference("", 5) == ""
