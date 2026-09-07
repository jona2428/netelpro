"""Tarea RLVR: verificar si un string es prefijo de otro."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "is_prefix"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos strings text y prefix, y devuelva "
    "true si prefix es un prefijo de text, false si no."
)
SIGNATURE = "(defn is-prefix (text prefix) ...)"
FN_NAME = "is-prefix"


def reference(text: str, prefix: str) -> bool:
    return text.startswith(prefix)


def gen_inputs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    cases: list[tuple[str, str]] = []
    for _ in range(n):
        text = random_word(rng, 3, 8)
        if rng.random() < 0.5:
            prefix = text[: rng.randint(0, len(text))]
        else:
            prefix = random_word(rng, 1, 4)
        cases.append((text, prefix))
    return cases


assert reference("hola", "ho") is True
assert reference("hola", "az") is False
assert reference("hola", "") is True
