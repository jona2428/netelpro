"""Tarea RLVR (curriculum gcd): divisiones enteras por 2 hasta quedar < 2."""
from __future__ import annotations

import random

TASK_ID = "halving_steps"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero positivo n y devuelva cuántas "
    "veces hay que dividirlo entre 2 (división entera) hasta que quede "
    "menor que 2."
)
SIGNATURE = "(defn halving-steps (n) ...)"
FN_NAME = "halving-steps"


def reference(n: int) -> int:
    steps = 0
    while n >= 2:
        n //= 2
        steps += 1
    return steps


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, 1_000_000),) for _ in range(n)]


assert reference(1) == 0
assert reference(2) == 1
assert reference(7) == 2
assert reference(16) == 4