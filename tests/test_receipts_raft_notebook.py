"""The Receipts-RAFT notebook generator and per-stage kernel builder."""

from __future__ import annotations

import ast
import json

import pytest

from training import create_receipts_raft_notebook as gen
from training import push_receipts_raft as push


def _code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_generated_notebook_is_committed_and_current():
    assert json.loads(gen.OUTPUT_NOTEBOOK.read_text(encoding="utf-8"))["cells"] == gen.build()["cells"], \
        "train_receipts_raft_kaggle.ipynb is stale: run python training/create_receipts_raft_notebook.py"


def test_cells_parse_and_frozen_protocol_survives():
    cells = _code_cells(gen.build())
    for src in cells:
        ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")))
    blob = "\n".join(cells)
    for frozen in ("r=16", "lora_alpha=16", "max_seq_length = 1024", "num_train_epochs=2",
                   "per_device_train_batch_size=2", "gradient_accumulation_steps=4"):
        assert frozen in blob
    assert "rr.require_audit(REPO, ARM, STAGE - 1)" in blob  # D11 enforced inside the kernel
    assert "rr.rescore(prev_pool)" in blob  # §6.2 whole-pool re-score every stage


def test_missing_anchor_aborts(monkeypatch):
    monkeypatch.setitem(gen.ANCHORS, "model", "this anchor does not exist")
    with pytest.raises(SystemExit):
        gen.build()


def test_stage_kernels_chain_and_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "KERNELS_DIR", tmp_path)
    out = push.build("B", 0)  # stage 0 runs once, as the shared arm
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"].endswith("/receipts-raft-b-s0") and meta["kernel_sources"] == [] and meta["enable_gpu"]
    nb = json.loads((out / "receipts-raft-b-s0.ipynb").read_text(encoding="utf-8"))
    cfg = next(s for s in _code_cells(nb) if s.lstrip().startswith("# Lo sobreescribe"))
    assert 'ARM = "B"' in cfg and "STAGE = 0" in cfg
    with pytest.raises(SystemExit, match="hand audit"):
        push.build("B", 1)  # no committed audit for round 0 yet


def test_stage_zero_is_shared_and_both_arms_start_from_it(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    labels = repo / "benchmarks" / "receipts_raft_audit" / "armB_r0_labels.json"  # ONE audit for round 0
    labels.parent.mkdir(parents=True)
    labels.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(push, "REPO", repo)
    monkeypatch.setattr(push, "KERNELS_DIR", tmp_path / "k")
    with pytest.raises(SystemExit, match="shared"):
        push.build("A", 0)
    for arm in ("A", "B"):
        meta = json.loads((push.build(arm, 1) / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["kernel_sources"] == [f"{push.KAGGLE_USER}/receipts-raft-b-s0"]
    labels_a1 = repo / "benchmarks" / "receipts_raft_audit" / "armA_r1_labels.json"
    with pytest.raises(SystemExit, match="hand audit"):
        push.build("A", 2)  # from round 1 on, each arm has its own audit
    labels_a1.write_text("[]", encoding="utf-8")
    meta = json.loads((push.build("A", 2) / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["kernel_sources"] == [f"{push.KAGGLE_USER}/receipts-raft-a-s1"]
