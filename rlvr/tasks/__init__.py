"""Registro de tareas RLVR + split determinístico train/OOD.

Cada módulo bajo `rlvr.tasks` expone el contrato de tarea (ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §3.1):
TASK_ID, DESCRIPTION_ES, SIGNATURE, FN_NAME, reference(*args), gen_inputs(n, seed).
"""
from __future__ import annotations

import hashlib
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
    """Split determinístico por hash de TASK_ID -- no aleatorio en cada corrida.

    Un mismo TASK_ID siempre cae del mismo lado, sin importar el orden de la
    lista de entrada ni cuántas veces se llame.
    """
    if not 0.0 < ood_fraction < 1.0:
        raise ValueError("ood_fraction debe estar entre 0 y 1 (exclusivo)")
    threshold = int(ood_fraction * (2**32))
    train: list[str] = []
    ood: list[str] = []
    for task_id in sorted(task_ids):
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big")
        if bucket < threshold:
            ood.append(task_id)
        else:
            train.append(task_id)
    return train, ood


__all__ = [
    "TASK_MODULE_NAMES",
    "REQUIRED_ATTRS",
    "load_task",
    "load_all_tasks",
    "split_train_ood",
]
