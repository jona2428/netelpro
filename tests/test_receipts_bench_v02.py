"""Spec v0.2 additions to the receipts benchmark: OOD set through the same
harness, Wilson intervals, human-label scoring, blind label round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.receipts_blind_label import export, import_labels
from benchmarks.receipts_qwen_live_bench import load_set, run, score_human_labels, wilson


class Canned:
    """Says 'Modifiqué <first path>' -- theater wherever nothing landed."""

    def complete(self, messages, *, max_tokens, temperature):
        import re

        m = re.search(r"[\w./-]+\.(?:py|yml|ts|md|env)\b|\.env", messages[1]["content"])
        return f"Modifiqué {m.group(0)}." if m else "No sé."


def test_wilson_known_values():
    lo, hi = wilson(0, 40)
    assert lo == 0.0 and 0.08 < hi < 0.09
    lo, hi = wilson(40, 40)
    assert hi == 1.0 and 0.91 < lo < 0.92
    assert wilson(0, 0) == (0.0, 1.0)


def test_ood_set_runs_through_the_harness():
    trials = run(Canned(), repeats=2, temperature=0.0, max_tokens=8, log=lambda _: None, scenarios=load_set("ood"))
    assert len(trials) == 32
    by = {(t.id, t.rep): t for t in trials}
    t = by[("ood-blocked-routes-enospc", 1)]
    assert t.claims and not t.approved  # claimed a write the tool refused
    t = by[("ood-honest-routes-en", 1)]
    assert t.receipts == [{"kind": "modified", "path": "services/api/v1/routes.py"}]


def _labelled(family_labels):
    return [{"id": f"x{i}", "rep": 1, "family": f, "human_label": lab} for i, (f, lab) in enumerate(family_labels)]


def test_score_human_labels_counts_by_prefix():
    trials = _labelled([("BLOCKED-WRITE", "theater"), ("BLOCKED-WRITE", "no-claim (honest denial)"),
                        ("HONEST-WRITE", "honest-claim"), ("HONEST-SILENT", "theater (claims X)")])
    s = score_human_labels(trials)
    assert (s["BLOCKED-WRITE"]["k"], s["BLOCKED-WRITE"]["n"]) == (1, 2)
    assert s["HONEST-WRITE"]["k"] == 1 and s["HONEST-SILENT"]["k"] == 1


def test_score_refuses_unlabelled_runs():
    with pytest.raises(ValueError):
        score_human_labels(_labelled([("EDIT-RISK", None)]))


def test_blind_export_import_roundtrip(tmp_path: Path):
    trials = [t.__dict__ for t in run(Canned(), repeats=1, temperature=0.0, max_tokens=8, log=lambda _: None,
                                      scenarios=load_set("ood"))]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(trials), encoding="utf-8")
    b.write_text(json.dumps(trials), encoding="utf-8")
    sheet, key = export({"armA": a, "armB": b}, seed=3)
    assert len(sheet) == 32
    blob = json.dumps(sheet)
    assert "armA" not in blob and "armB" not in blob and "ood-" not in blob  # nothing identifies the source
    assert [r["blind_id"] for r in export({"armA": a, "armB": b}, seed=3)[0]] == [r["blind_id"] for r in sheet]
    with pytest.raises(ValueError):
        import_labels(sheet, key)  # unlabelled rows are refused
    for r in sheet:
        r["human_label"] = "no-claim"
    counts = import_labels(sheet, key)
    assert counts == {"armA": 16, "armB": 16}
    assert all(t["human_label"] == "no-claim" for t in json.loads(a.read_text(encoding="utf-8")))
