"""Tarea RLVR: convertir un string de dígitos decimales a entero."""
from __future__ import annotations

import random

TASK_ID = "string_to_int"
DESCRIPTION_ES = (
    "Escribí una función que reciba un string que representa un entero "
    "decimal (sin espacios, sin signo '+', puede tener '-' al inicio) y "
    "devuelva el entero correspondiente."
)
SIGNATURE = "(defn string-to-int (s) ...)"
FN_NAME = "string-to-int"


def reference(s: str) -> int:
    return int(s)


def gen_inputs(n: int, seed: int) -> list[tuple[str]]:
    rng = random.Random(seed)
    return [(str(rng.randint(-1000, 1000)),) for _ in range(n)]


assert reference("0") == 0
assert reference("-42") == -42
assert reference("123") == 123
