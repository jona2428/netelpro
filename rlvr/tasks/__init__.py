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
    # --- corpus growth 2026-09-08 (OOD granularity 20% -> 5%, ver docstring
    # de OOD_TASK_IDS más abajo) ---
    "lcm_pair",
    "collatz_steps",
    "reverse_digits",
    "is_perfect_square",
    "ceil_div",
    "digit_count",
    "max_of_three",
    "sum_even_up_to",
    "product_range",
    "is_palindrome_number",
    "list_min",
    "count_occurrences",
    "is_sorted_ascending",
    "list_index_of",
    "count_zero",
    "list_product",
    "list_all_positive",
    "count_even",
    "append_at_end",
    "list_average_floor",
    "string_repeat_n",
    "longest_of_two",
    "concat_with_separator",
    "is_empty_string",
    "str_len_diff",
    "shorter_of_two",
    "int_pair_to_string",
    "wrap_in_brackets",
    "double_concat",
    "prefix_or_default",
    # --- curriculum gcd 2026-09-08 (lever run #3) ---
    # gcd_pair (OOD) quedó 0/8 en runs #1/#2: exige recursión de DOS
    # argumentos con reorden (gcd-two b (rem a b)), patrón que ninguna tarea
    # train ejercita (las tareas con quot/rem son unarias). Estas 3 enseñan
    # ese patrón con salidas distintas al gcd (transferencia, no memorización
    # del OOD): halving-steps (recursión quot unaria), euclid-steps
    # (reorden+rem) y sub-gcd-steps (reorden bidireccional). TRAIN side
    # only: OOD_TASK_IDS intacto -- regla de mantenimiento del contrato.
    "halving_steps",
    "euclid_steps",
    "sub_gcd_steps",
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
# Held-out (OOD) fijo por decisión de diseño (review final 2026-09-07, fix I1;
# crecido 2026-09-08 para resolver la granularidad de pass@8).
#
# Diseño original (runs #1/#2, 5 tareas): la spec promete ~20% del corpus
# como held-out. El hash-split inicial dio 2/25 (8%) -- granularity de 50%
# por tarea hacía la evaluación §7 estadísticamente hueca. El split se
# declaró EXPLÍCITO: 5 tareas (20%), balanceadas por familia.
#
# Crecimiento 2026-09-08: 5 tareas seguían dando granularidad de 20% por
# tarea (un solo acierto de más movía el pass@8 entero un salto). Se decidió
# CRECER el absoluto de OOD a 20 tareas (granularidad 5%) en vez de mantener
# el ~20% del corpus a rajatabla -- eso hubiera exigido ~100 tareas totales
# (80 train) solo para sostener la proporción, trabajo desproporcionado al
# problema real (resolución estadística, no proporción). El corpus total
# queda en 55 (35 train + 20 OOD, ~36%) -- la convención del 20% se
# releva explícitamente por la misma razón que motivó el fix original.
#
# Balance por familia (7 aritmética / 7 listas / 6 strings):
#   arithmetic: power_int, gcd_pair (run #1/#2, sin cambios), + lcm_pair,
#     collatz_steps, reverse_digits, is_perfect_square, ceil_div
#   list: nth_element, list_sum (run #1/#2, sin cambios), + list_min,
#     count_occurrences, is_sorted_ascending, list_index_of, count_zero
#   string: string_to_int (run #1/#2, sin cambios), + string_repeat_n,
#     longest_of_two, concat_with_separator, is_empty_string, str_len_diff
#
# Regla de mantenimiento: OOD_TASK_IDS es un CONTRATO, no una sugerencia.
# Añadir tareas al corpus NO las agrega al OOD; cambiar esta lista es una
# decisión de diseño (cambiar el set de evaluación invalida comparaciones
# históricas -- por eso las 5 originales de runs #1/#2 se mantienen intactas
# dentro del set de 20, para que ese tramo de la serie siga siendo comparable).
# ---------------------------------------------------------------------------
OOD_TASK_IDS: tuple[str, ...] = (
    "power_int",
    "nth_element",
    "string_to_int",
    "gcd_pair",
    "list_sum",
    "lcm_pair",
    "collatz_steps",
    "reverse_digits",
    "is_perfect_square",
    "ceil_div",
    "list_min",
    "count_occurrences",
    "is_sorted_ascending",
    "list_index_of",
    "count_zero",
    "string_repeat_n",
    "longest_of_two",
    "concat_with_separator",
    "is_empty_string",
    "str_len_diff",
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
