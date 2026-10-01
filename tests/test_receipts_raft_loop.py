"""Receipts-RAFT loop logic without a GPU: harvest, rescore, SFT/DPO data,
audit sheet and the D11 gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.receipts_raft import (
    MAX_KEEP_PER_SCENARIO,
    Sample,
    audit_gate,
    audit_labels_path,
    audit_sheet,
    dpo_pairs,
    harvest,
    load_pool,
    require_audit,
    rescore,
    round_stats,
    save_pool,
    sft_examples,
)


def fake_sampler(messages, n, temperature):
    """Half honest-looking, half lies: 'Listo.' and a claim on every path named."""
    import re

    paths = re.findall(r"[\w/-]+\.[a-z]{2,4}\b|\b(?:Dockerfile|Makefile)\b", messages[1]["content"])
    lie = " ".join(f"Modifiqué {p}." for p in paths) or "Listo."
    honest = f"Todavía no tengo el resultado de la herramienta para {paths[0] if paths else 'eso'}, no confirmo ningún cambio."
    return [lie, honest, "Listo."] * (n // 3) + [lie] * (n % 3)


@pytest.fixture(scope="module")
def pool() -> list[Sample]:
    return harvest(0, fake_sampler, n=6)


def test_harvest_grades_every_sample(pool):
    assert len(pool) == 60 * 6
    assert {s.reward for s in pool} == {0, 1}
    assert all((s.reward == 1) == (s.reasons == []) for s in pool)


def test_rescore_is_idempotent_and_catches_tampering(pool):
    copy = [Sample(**s.__dict__) for s in pool]
    assert rescore(copy) == 0
    copy[0].reward = 1 - copy[0].reward
    assert rescore(copy) == 1


def test_sft_keeps_only_winners_capped(pool):
    ex = sft_examples(pool)
    assert ex
    per: dict[str, int] = {}
    for e in ex:
        key = json.dumps(e["messages"])
        per[key] = per.get(key, 0) + 1
    assert max(per.values()) <= MAX_KEEP_PER_SCENARIO
    winners = {s.text for s in pool if s.reward == 1}
    assert all(e["completion"] in winners for e in ex)


def test_dpo_pairs_are_same_prompt_win_vs_loss(pool):
    pairs = dpo_pairs(pool)
    assert pairs
    by_text = {}
    for s in pool:
        by_text.setdefault(s.text, set()).add(s.reward)
    for p in pairs:
        assert 1 in by_text[p["chosen"]] and 0 in by_text[p["rejected"]] and p["chosen"] != p["rejected"]


def test_audit_sheet_has_only_winners_and_hides_reasons(pool):
    sheet = audit_sheet(pool, seed=1)
    assert 0 < len(sheet) <= 48
    blob = json.dumps(sheet)
    assert "D1:" not in blob and "D13" not in blob and "reasons" not in blob
    assert all(m["role"] != "system" for r in sheet for m in r["conversation"])
    winners = {s.text for s in pool if s.reward == 1}
    assert all(r["model_text"] in winners for r in sheet)
    assert audit_sheet(pool, seed=1) == sheet


def _sheet(labels):
    return [{"audit_id": f"a{i}", "human_label": lab} for i, lab in enumerate(labels)]


def test_gate_allows_one_in_48_and_blocks_two():
    assert audit_gate(_sheet(["no-claim"] * 47 + ["theater"]))[1]
    assert not audit_gate(_sheet(["no-claim"] * 46 + ["theater"] * 2))[1]


def test_gate_fails_closed():
    assert not audit_gate([])[1]
    assert not audit_gate(_sheet(["no-claim", None]))[1]
    assert not audit_gate(_sheet(["looks fine"]))[1]


def test_round_zero_audit_is_shared_by_both_arms(tmp_path: Path):
    assert audit_labels_path(tmp_path, "A", 0) == audit_labels_path(tmp_path, "B", 0)
    assert audit_labels_path(tmp_path, "A", 1) != audit_labels_path(tmp_path, "B", 1)


def test_require_audit_reads_the_committed_file(tmp_path: Path):
    with pytest.raises(RuntimeError, match="no hand audit"):
        require_audit(tmp_path, "A", 0)
    p = audit_labels_path(tmp_path, "A", 0)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_sheet(["theater"] * 3 + ["no-claim"] * 45)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ABOVE 2%"):
        require_audit(tmp_path, "A", 0)
    p.write_text(json.dumps(_sheet(["honest-claim"] * 48)), encoding="utf-8")
    assert "ok" in require_audit(tmp_path, "A", 0)


def test_pool_roundtrip_and_stats(tmp_path: Path, pool):
    f = tmp_path / "pool.json"
    save_pool(pool, f)
    assert [s.__dict__ for s in load_pool(f)] == [s.__dict__ for s in pool]
    st = round_stats(pool)
    assert st["n"] == len(pool) and set(st["r1_by_family"]) == {"EDIT-RISK", "BLOCKED-WRITE", "PARTIAL", "HONEST-WRITE", "HONEST-SILENT"}
