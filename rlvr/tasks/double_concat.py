"""Tarea RLVR: concatenar un string consigo mismo."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "double_concat"
DESCRIPTION_ES = (
    "Escribí una función que reciba un string s y devuelva s concatenado "
    "con sí mismo. Ejemplo: 'ab' -> 'abab'."
)
SIGNATURE = "(defn double-concat (s) ...)"
FN_NAME = "double-concat"


def reference(s: str) -> str:
    return s + s


def gen_inputs(n: int, seed: int) -> list[tuple[str]]:
    rng = random.Random(seed)
    return [(random_word(rng, 0, 6),) for _ in range(n)]


assert reference("ab") == "abab"
assert reference("") == ""
