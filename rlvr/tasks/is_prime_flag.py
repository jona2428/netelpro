"""Tarea RLVR: determinar si un entero (>= 2) es primo."""
from __future__ import annotations

import random

TASK_ID = "is_prime_flag"
DESCRIPTION_ES = (
    "Escribí una función que reciba un entero n >= 2 y devuelva true si n "
    "es primo, false si no lo es."
)
SIGNATURE = "(defn is-prime (n) ...)"
FN_NAME = "is-prime"


def reference(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, n):
        if n % d == 0:
            return False
    return True


def gen_inputs(n: int, seed: int) -> list[tuple[int]]:
    rng = random.Random(seed)
    return [(rng.randint(2, 200),) for _ in range(n)]


assert reference(2) is True
assert reference(4) is False
assert reference(17) is True
assert reference(21) is False
