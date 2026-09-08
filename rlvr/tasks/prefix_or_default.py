"""Tarea RLVR: devolver text si tiene un prefijo dado, si no un valor por defecto."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "prefix_or_default"
DESCRIPTION_ES = (
    "Escribí una función que reciba tres strings text, prefix y default. "
    "Si prefix es un prefijo de text, devolvé text. Si no, devolvé default."
)
SIGNATURE = "(defn prefix-or-default (text prefix default) ...)"
FN_NAME = "prefix-or-default"


def reference(text: str, prefix: str, default: str) -> str:
    return text if text.startswith(prefix) else default


def gen_inputs(n: int, seed: int) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    cases: list[tuple[str, str, str]] = []
    for _ in range(n):
        text = random_word(rng, 3, 8)
        if rng.random() < 0.5:
            prefix = text[: rng.randint(0, len(text))]
        else:
            prefix = random_word(rng, 1, 4)
        default = random_word(rng, 1, 5)
        cases.append((text, prefix, default))
    return cases


assert reference("hola", "ho", "x") == "hola"
assert reference("hola", "az", "x") == "x"
