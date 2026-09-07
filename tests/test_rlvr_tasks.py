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
    ids = [f"task_{i}" for i in range(500)]
    train, ood = split_train_ood(ids, ood_fraction=0.2)
    ratio = len(ood) / len(ids)
    assert 0.1 <= ratio <= 0.3


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
