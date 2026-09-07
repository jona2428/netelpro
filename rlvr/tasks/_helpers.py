"""Helpers compartidos para gen_inputs de tareas del corpus RLVR."""
from __future__ import annotations

import random


def random_int_list(
    rng: random.Random, min_len: int, max_len: int, lo: int, hi: int
) -> list[int]:
    length = rng.randint(min_len, max_len)
    return [rng.randint(lo, hi) for _ in range(length)]
