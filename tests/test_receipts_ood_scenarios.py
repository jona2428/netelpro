"""Contract for the frozen OOD set (Receipts-RAFT spec v0.2 §5b, D10)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from benchmarks.receipts_ood_scenarios import OOD_SCENARIOS, OOD_TREE
from netelpro.receipts import MutationGuard
from rlvr.receipts_scenarios import OOD_AXES, TREES, ContractError, generate_round, validate

ALL_OOD_TAGS = {v for vals in OOD_AXES.values() for v in vals}


def test_sixteen_with_benchmark_family_mix():
    assert len(OOD_SCENARIOS) == 16
    assert len({s.id for s in OOD_SCENARIOS}) == 16
    assert Counter(s.family for s in OOD_SCENARIOS) == {
        "EDIT-RISK": 4, "BLOCKED-WRITE": 4, "PARTIAL": 2, "HONEST-WRITE": 4, "HONEST-SILENT": 2,
    }


@pytest.mark.parametrize("s", OOD_SCENARIOS, ids=lambda s: s.id)
def test_every_scenario_uses_a_reserved_axis_and_train_rejects_it(s):
    tags = set(s.axes["ood"])
    assert tags and tags <= ALL_OOD_TAGS, s.id
    with pytest.raises(ContractError):
        validate(s)


def test_every_reserved_axis_is_covered():
    used = {t for s in OOD_SCENARIOS for t in s.axes["ood"]}
    assert used == ALL_OOD_TAGS


def test_ood_paths_never_appear_in_train_trees():
    train_paths = {p for t in TREES.values() for p in list(t.files) + list(t.create_candidates)}
    assert not set(OOD_TREE) & train_paths


def test_no_ood_message_can_come_from_the_generator():
    gen = {m["content"] for seed in range(10) for s in generate_round(seed) for m in s.messages if m["role"] == "user"}
    for s in OOD_SCENARIOS:
        for m in s.messages:
            if m["role"] == "user":
                assert m["content"] not in gen, s.id


@pytest.mark.parametrize("s", OOD_SCENARIOS, ids=lambda s: s.id)
def test_receipts_match_what_the_tool_reported(tmp_path: Path, s):
    s.materialize(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    s.apply(tmp_path)
    got = {(r.kind, r.path) for r in guard.end()}
    want: set[tuple[str, str]] = set()
    for e in s.applied:
        if e.op == "rename":
            want |= {("deleted", e.path), ("created", e.content)}
        else:
            want.add(({"create": "created", "modify": "modified", "delete": "deleted"}[e.op], e.path))
    assert got == want
    for p in s.failed_paths:
        assert p not in {path for _, path in got}


@pytest.mark.parametrize("s", OOD_SCENARIOS, ids=lambda s: s.id)
def test_tool_turn_names_every_requested_path(s):
    if s.family in ("EDIT-RISK", "HONEST-SILENT"):
        assert len(s.messages) == 2
        return
    tool_turn = s.messages[-1]["content"]
    for e in s.requested:
        assert e.path in tool_turn, (s.id, e.path)
