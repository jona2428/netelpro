"""Contract tests para rlvr.tasks: registro de tareas, split determinístico
train/OOD, y que cada tarea cumpla su contrato. Ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §3."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlvr.tasks import (  # noqa: E402
    REQUIRED_ATTRS,
    TASK_MODULE_NAMES,
    load_all_tasks,
    load_task,
    split_train_ood,
)


def test_split_is_deterministic():
    ids = [f"task_{i}" for i in range(50)]
    train1, ood1 = split_train_ood(ids)
    train2, ood2 = split_train_ood(ids)
    assert train1 == train2
    assert ood1 == ood2


def test_split_covers_every_id_exactly_once():
    ids = [f"task_{i}" for i in range(200)]
    train, ood = split_train_ood(ids, ood_fraction=0.2)
    assert sorted(train + ood) == sorted(ids)
    assert set(train).isdisjoint(ood)


def test_split_fraction_is_roughly_respected():
    """Fix I1 + crecimiento 2026-09-08: el OOD es el contrato explícito de
    20 tareas (granularidad 5% de pass@8, corpus total 55).

    El hash-split original daba 2/25 (8%); el split explícito de 5 tareas
    (fix I1) daba 20% pero con granularidad de 20% por tarea -- 15 tareas
    más se agregaron para bajar la granularidad a 5% sin escalar el corpus
    a ~100 tareas por sostener la proporción del 20% (ver docstring de
    OOD_TASK_IDS). El ratio ya no se acota a ~20%: es un efecto secundario
    del tamaño del corpus, no un invariante que el split deba mantener.
    """
    ids = TASK_MODULE_NAMES
    train, ood = split_train_ood(ids, ood_fraction=0.2)
    assert ood == sorted(
        [
            "gcd_pair",
            "list_sum",
            "nth_element",
            "power_int",
            "string_to_int",
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
        ]
    )
    assert len(ood) == 20
    ratio = len(ood) / len(ids)
    assert 0.3 <= ratio <= 0.4


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_out_of_range_fraction(bad_fraction):
    with pytest.raises(ValueError):
        split_train_ood(["a", "b"], ood_fraction=bad_fraction)


def test_load_task_raises_on_module_missing_required_attrs(monkeypatch):
    fake_module = types.ModuleType("rlvr.tasks._fake_incomplete")
    fake_module.TASK_ID = "fake"  # falta DESCRIPTION_ES, SIGNATURE, FN_NAME, reference, gen_inputs

    def fake_import(name):
        assert name == "rlvr.tasks._fake_incomplete"
        return fake_module

    monkeypatch.setattr("rlvr.tasks.importlib.import_module", fake_import)
    with pytest.raises(ValueError, match="no implementa"):
        load_task("_fake_incomplete")


def test_load_all_tasks_returns_every_registered_module():
    tasks = load_all_tasks()
    assert set(tasks.keys()) == set(TASK_MODULE_NAMES)
    for name, module in tasks.items():
        for attr in REQUIRED_ATTRS:
            assert hasattr(module, attr), f"{name} le falta {attr}"


def test_task_ids_are_unique():
    tasks = load_all_tasks()
    ids = [m.TASK_ID for m in tasks.values()]
    assert len(ids) == len(set(ids)), "TASK_ID duplicado en el corpus"


@pytest.mark.parametrize("task_name", TASK_MODULE_NAMES)
def test_task_gen_inputs_are_consistent_with_reference(task_name):
    module = load_task(task_name)
    inputs = module.gen_inputs(8, seed=1)
    assert len(inputs) == 8
    for args in inputs:
        assert isinstance(args, tuple)
        module.reference(*args)  # no debe lanzar
