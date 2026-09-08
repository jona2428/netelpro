"""Tarea RLVR: determinar si un string está vacío."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "is_empty_string"
DESCRIPTION_ES = (
    "Escribí una función que reciba un string s y devuelva true si está "
    "vacío (largo 0), false si no."
)
SIGNATURE = "(defn is-empty-string (s) ...)"
FN_NAME = "is-empty-string"


def reference(s: str) -> bool:
    return len(s) == 0


def gen_inputs(n: int, seed: int) -> list[tuple[str]]:
    rng = random.Random(seed)
    return [("",) if rng.random() < 0.5 else (random_word(rng, 1, 6),) for _ in range(n)]


assert reference("") is True
assert reference("a") is False
