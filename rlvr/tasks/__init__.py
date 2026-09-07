"""Registro de tareas RLVR + split determinístico train/OOD.

Cada módulo bajo `rlvr.tasks` expone el contrato de tarea (ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §3.1):
TASK_ID, DESCRIPTION_ES, SIGNATURE, FN_NAME, reference(*args), gen_inputs(n, seed).
"""
from __future__ import annotations

import importlib
from types import ModuleType

TASK_MODULE_NAMES: list[str] = [
    "double_value",
    "factorial",
    "sum_range",
    "gcd_pair",
    "power_int",
    "sum_of_digits",
    "count_divisors",
    "is_prime_flag",
    "reverse_list",
    "list_length",
    "list_sum",
    "list_max",
    "count_positive",
    "count_negative",
    "list_contains",
    "nth_element",
    "sum_of_squares",
    "concat_strings",
    "string_length",
    "is_prefix",
    "strings_equal",
    "int_to_string",
    "string_to_int",
    "concat_three",
    "label_with_length",
]

REQUIRED_ATTRS: tuple[str, ...] = (
    "TASK_ID",
    "DESCRIPTION_ES",
    "SIGNATURE",
    "FN_NAME",
    "reference",
    "gen_inputs",
)


# ---------------------------------------------------------------------------
# Held-out (OOD) fijo por decisión de diseño (review final 2026-09-07, fix I1).
#
# La spec promete ~20% del corpus como held-out. El hash-split original dio
# 2/25 (8%) -- granularity de 50% por tarea hacía la evaluación §7
# estadísticamente hueca. En vez de crecer el corpus, el split se declara
# EXPLÍCITO: 5 tareas (20%), balanceadas por familia para que el held-out
# mida las tres familias del corpus:
#   - arithmetic: power_int (recursión multiplicativa, no confundible con
#     los few-shots fib/sum-to que viven en TRAIN)
#   - list: nth_element (indexing, distinto del reverse del test-verifier)
#   - string: string_to_int (parseo inverso de int_to_string, TRAIN)
#   - arithmetic: gcd_pair (Euclides, la forma de recursión más "clásica")
#   - list: list_sum (fold manual -- el patrón de composición más común)
#
# Regla de mantenimiento: OOD_TASK_IDS es un CONTRATO, no una sugerencia.
# Añadir tareas al corpus NO las agrega al OOD; cambiar esta lista es una
# decisión de diseño (cambiar el set de evaluación invalida comparaciones
# históricas).
# ---------------------------------------------------------------------------
OOD_TASK_IDS: tuple[str, ...] = (
    "power_int",
    "nth_element",
    "string_to_int",
    "gcd_pair",
    "list_sum",
)


def load_task(name: str) -> ModuleType:
    """Importa un módulo de tarea por nombre y valida que cumpla el contrato."""
    module = importlib.import_module(f"rlvr.tasks.{name}")
    missing = [attr for attr in REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise ValueError(f"tarea '{name}' no implementa: {missing}")
    return module


def load_all_tasks() -> dict[str, ModuleType]:
    """Carga todos los módulos registrados en TASK_MODULE_NAMES."""
    return {name: load_task(name) for name in TASK_MODULE_NAMES}


def split_train_ood(
    task_ids: list[str], ood_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    """Split determinístico train/OOD: held-out EXPLÍCITO por contrato.

    El OOD es OOD_TASK_IDS (fijo por diseño, ~20% del corpus, balanceado por
    familia). Cualquier id presente en task_ids pero NO en OOD_TASK_IDS va
    a train -- la lista explícita es un SUBSET del corpus, no un dominio
    cerrado: ids sintéticos o futuros no listados van a train (el contrato
    OOD solo secuestra ids que existen como tareas registradas).
    Determinístico por construcción: mismo input, mismo output.
    """
    if not 0.0 < ood_fraction < 1.0:
        raise ValueError("ood_fraction debe estar entre 0 y 1 (exclusivo)")
    ood_set = set(OOD_TASK_IDS)
    train = [tid for tid in sorted(task_ids) if tid not in ood_set]
    ood = [tid for tid in sorted(task_ids) if tid in ood_set]
    return train, ood


__all__ = [
    "TASK_MODULE_NAMES",
    "REQUIRED_ATTRS",
    "OOD_TASK_IDS",
    "load_task",
    "load_all_tasks",
    "split_train_ood",
]
