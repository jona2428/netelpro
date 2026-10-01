"""Contract tests for the Receipts-RAFT procedural scenario generator.

Spec: docs/superpowers/specs/2026-10-01-receipts-raft-design.md, §3 (D5, D6)
and §8 step 1. What these pin down:

  - determinism by seed (D5): same round seed, byte-identical scenarios;
  - family coverage: 12 per family, 60 per round, five families;
  - the held-out contract (D7): no train scenario reuses a benchmark user
    message or any path the 16 benchmark scenarios touch;
  - the OOD contract (D6): no reserved axis value (English, ENOSPC/timeout,
    unified diff, rename, dotfiles, 4-level paths) ever reaches train;
  - effects are real: applying a scenario to a seeded workspace produces
    exactly the receipts the scenario says landed, and nothing for the
    paths the tool output reported as failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.receipts_qwen_live_bench import SCENARIOS as BENCH_SCENARIOS
from netelpro.receipts import MutationGuard
from rlvr.receipts_scenarios import (
    BENCH_PATHS,
    ERROR_TEMPLATES,
    FAMILIES,
    MAX_TRAIN_DEPTH,
    OOD_AXES,
    PER_FAMILY,
    TRAIN_ERRORS,
    TRAIN_FORMATS,
    TRAIN_OPS,
    TREES,
    ContractError,
    GenScenario,
    generate_round,
    validate,
)

ROUND_SEEDS = range(20)


@pytest.fixture(scope="module")
def many_rounds() -> list[GenScenario]:
    out: list[GenScenario] = []
    for seed in ROUND_SEEDS:
        out.extend(generate_round(seed))
    return out


def _user_texts(s: GenScenario) -> list[str]:
    return [m["content"] for m in s.messages if m["role"] == "user"]


def _all_text(s: GenScenario) -> str:
    return "\n".join(m["content"] for m in s.messages)


# --- determinism (D5) -------------------------------------------------------


def test_same_seed_same_round():
    a = generate_round(7)
    b = generate_round(7)
    assert [s.to_json() for s in a] == [s.to_json() for s in b]


def test_different_seed_different_prompts():
    a = {_all_text(s) for s in generate_round(1)}
    b = {_all_text(s) for s in generate_round(2)}
    # Fresh seed per round so the accumulated pool does not repeat prompts.
    assert len(a & b) <= 3


def test_to_json_roundtrips():
    s = generate_round(3)[0]
    assert GenScenario.from_json(json.loads(json.dumps(s.to_json()))) == s


# --- coverage ---------------------------------------------------------------


def test_round_shape():
    r = generate_round(0)
    assert len(r) == PER_FAMILY * len(FAMILIES) == 60
    for fam in FAMILIES:
        assert sum(s.family == fam for s in r) == PER_FAMILY
    assert len({s.id for s in r}) == len(r)


def test_twelve_trees_two_to_six_files():
    assert len(TREES) == 12
    for name, tree in TREES.items():
        assert 2 <= len(tree.files) <= 6, name


def test_all_train_axes_get_used(many_rounds):
    assert {s.axes["format"] for s in many_rounds if s.axes["format"]} == set(TRAIN_FORMATS)
    assert {e for s in many_rounds for e in s.axes["errors"]} == set(TRAIN_ERRORS)
    assert {e.op for s in many_rounds for e in s.requested} == set(TRAIN_OPS)
    assert {s.tree for s in many_rounds} == set(TREES)
    assert {len(s.requested) for s in many_rounds if s.family != "HONEST-SILENT"} == {1, 2, 3}


@pytest.mark.parametrize("family", ["EDIT-RISK", "BLOCKED-WRITE", "PARTIAL", "HONEST-WRITE", "HONEST-SILENT"])
def test_family_semantics(many_rounds, family):
    for s in (x for x in many_rounds if x.family == family):
        req = {e.path for e in s.requested}
        landed = {e.path for e in s.applied}
        failed = set(s.failed_paths)
        if family == "EDIT-RISK":
            assert req and not landed and not failed
            assert len(s.messages) == 2  # system + user, never a tool result
        elif family == "BLOCKED-WRITE":
            assert req and not landed and failed == req
        elif family == "PARTIAL":
            assert len(req) >= 2 and landed and failed
            assert landed | failed == req and not (landed & failed)
        elif family == "HONEST-WRITE":
            assert req and landed == req and not failed
        else:  # HONEST-SILENT
            assert not req and not landed and not failed
            assert s.mentioned_paths and len(s.messages) == 2
        # D13 needs the paths the request names; every request names them.
        for p in s.mentioned_paths:
            assert p in _user_texts(s)[0]


def test_tool_output_names_every_requested_path(many_rounds):
    for s in many_rounds:
        if s.family in ("BLOCKED-WRITE", "PARTIAL", "HONEST-WRITE"):
            tool_turn = _user_texts(s)[-1]
            for e in s.requested:
                assert e.path in tool_turn, (s.id, e.path)


# --- held-out contract (D7) -------------------------------------------------


def test_bench_paths_cover_the_benchmark():
    for b in BENCH_SCENARIOS:
        text = "\n".join(m["content"] for m in b.messages if m["role"] == "user")
        for p in ("src/app.py", "config/settings.py", "README.md", "tests/test_app.py", "src/utils.py"):
            if p in text:
                assert p in BENCH_PATHS


def test_no_train_tree_uses_a_benchmark_path():
    for name, tree in TREES.items():
        for p in list(tree.files) + list(tree.create_candidates):
            assert p not in BENCH_PATHS, (name, p)


def test_no_train_scenario_matches_the_benchmark(many_rounds):
    bench_user = {m["content"] for b in BENCH_SCENARIOS for m in b.messages if m["role"] == "user"}
    for s in many_rounds:
        for t in _user_texts(s):
            assert t not in bench_user, s.id
        for p in s.mentioned_paths:
            assert p not in BENCH_PATHS, (s.id, p)


# --- OOD contract (D6) ------------------------------------------------------


def test_ood_axes_are_disjoint_from_train():
    assert not set(OOD_AXES["errors"]) & set(TRAIN_ERRORS)
    assert not set(OOD_AXES["formats"]) & set(TRAIN_FORMATS)
    assert not set(OOD_AXES["ops"]) & set(TRAIN_OPS)
    assert OOD_AXES["langs"] == ("en",)
    for e in OOD_AXES["errors"]:
        assert e in ERROR_TEMPLATES  # step 4 builds the OOD set from these


def test_no_ood_axis_reaches_train(many_rounds):
    ood_markers = ["ENOSPC", "no space left", "timed out", "timeout", "@@ ", "--- a/", "+++ b/", "rename", "renombr"]
    for s in many_rounds:
        text = _all_text(s)
        for marker in ood_markers:
            assert marker.lower() not in text.lower(), (s.id, marker)
        assert s.axes["lang"] == "es"
        for p in s.mentioned_paths + [e.path for e in s.requested] + [p for p, _ in s.files]:
            parts = p.split("/")
            assert len(parts) <= MAX_TRAIN_DEPTH, (s.id, p)
            assert not any(part.startswith(".") for part in parts), (s.id, p)


def test_validate_rejects_ood_leaks():
    s = generate_round(0)[0]
    bad = s.replace_axes(errors=("ENOSPC",))
    with pytest.raises(ContractError):
        validate(bad)
    bad = s.replace_axes(format="unified_diff")
    with pytest.raises(ContractError):
        validate(bad)
    bad = s.replace_axes(lang="en")
    with pytest.raises(ContractError):
        validate(bad)


# --- effects are real (D5) --------------------------------------------------


def test_receipts_match_applied_effects(tmp_path: Path, many_rounds):
    kind_of = {"create": "created", "modify": "modified", "delete": "deleted"}
    for i, s in enumerate(many_rounds[:300]):
        root = tmp_path / f"w{i}"
        root.mkdir()
        s.materialize(root)
        guard = MutationGuard(root)
        guard.begin()
        s.apply(root)
        receipts = guard.end()
        got = {(r.kind, r.path) for r in receipts}
        want = {(kind_of[e.op], e.path) for e in s.applied}
        assert got == want, s.id
        for p in s.failed_paths:
            assert p not in {r.path for r in receipts}


def test_materialize_matches_tree(tmp_path: Path):
    s = generate_round(4)[5]
    s.materialize(tmp_path)
    for p, content in s.files:
        assert (tmp_path / p).read_text(encoding="utf-8") == content
