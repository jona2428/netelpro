"""Tarea RLVR: diferencia absoluta entre los largos de dos strings."""
from __future__ import annotations

import random

from rlvr.tasks._helpers import random_word

TASK_ID = "str_len_diff"
DESCRIPTION_ES = (
    "Escribí una función que reciba dos strings a y b, y devuelva la "
    "diferencia absoluta entre sus largos (siempre un entero >= 0)."
)
SIGNATURE = "(defn str-len-diff (a b) ...)"
FN_NAME = "str-len-diff"


def reference(a: str, b: str) -> int:
    return abs(len(a) - len(b))


def gen_inputs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(random_word(rng, 1, 8), random_word(rng, 1, 8)) for _ in range(n)]


assert reference("ab", "abcd") == 2
assert reference("abcd", "ab") == 2
assert reference("x", "x") == 0
