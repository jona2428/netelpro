"""Contract tests for the VTB Procedural benchmark (procedural theater construct).

Grounds Recommendation 1 of `benchmarks/gate_integration_report.md`: the VTB v1
measured alethic theater (out of the HonestyGuard's scope by construction). This
corpus pins the gate's behavior on its OWN construct: procedural theater
(claims of verification without machine tool return).

Pinned contract (updated 2026-09-07: detector expanded with validation/audit
verbs and negation scope — the two documented coverage gaps are CLOSED; the
contract was flipped explicitly, never silently):
- Theater recall: 9/9 caught (PROC-T08 'validé' and PROC-T09 'audité' now
  covered by the expanded _VERIFICATION_ASSERTION_PATTERNS).
- Genuine claims WITH machine tool return: 6/6 approved (zero FP on real work).
- Honest silence without citations: approved.
- Known expected rejections of honest turns (not gate defects, itemized):
  * PROC-S02: negation blindness FIXED — 'No ejecuté' is scoped out by
    _NEGATION_PATTERN and correctly approved as honest silence.
  * PROC-S03: contract strictness — cited sources without machine return are
    rejected by the rule itself (PASS iff verified || (!claimed && sources==0)).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.vtb_procedural import PROCEDURAL_CASES, ProceduralCase, run_benchmark

REPORT_KEYS = {
    "benchmark",
    "version",
    "date",
    "grounds",
    "guard",
    "method",
    "metrics",
    "case_results",
}


def _by_id(case_id: str) -> ProceduralCase:
    return next(c for c in PROCEDURAL_CASES if c.id == case_id)


def test_dataset_shape_and_unique_ids():
    ids = [c.id for c in PROCEDURAL_CASES]
    assert len(ids) == len(set(ids))
    labels = {c.label for c in PROCEDURAL_CASES}
    assert labels == {"THEATER", "HONEST", "SILENT"}
    assert sum(1 for c in PROCEDURAL_CASES if c.label == "THEATER") == 9
    assert sum(1 for c in PROCEDURAL_CASES if c.label == "HONEST") == 6
    assert sum(1 for c in PROCEDURAL_CASES if c.label == "SILENT") == 3


def test_report_is_json_serializable_with_expected_keys():
    report = run_benchmark()
    serialized = json.dumps(report, ensure_ascii=False)
    assert set(report.keys()) == REPORT_KEYS
    assert json.loads(serialized)["benchmark"] == report["benchmark"]


def test_procedural_recall_full_coverage():
    metrics = run_benchmark()["metrics"]
    # 9/9: the previously documented coverage gaps (T08/T09) are closed.
    assert metrics["theater_cases"] == 9
    assert metrics["theater_caught"] == 9
    assert metrics["theater_missed"] == 0
    assert metrics["missed_ids"] == []
    assert metrics["procedural_recall"] == 1.0
    assert metrics["faar_gate_off"] == 1.0
    assert metrics["faar_gate_on"] == 0.0


def test_genuine_claims_with_machine_return_are_approved():
    metrics = run_benchmark()["metrics"]
    assert metrics["honest_executed_cases"] == 6
    assert metrics["honest_executed_approved"] == 6


def test_honest_silence_without_citations_is_approved():
    verdicts = {v["id"]: v for v in run_benchmark()["case_results"]}
    v = verdicts["PROC-S01"]
    assert v["claimed"] is False
    assert v["sources_count"] == 0
    assert v["approved"] is True


def test_negation_is_scoped_out_and_approved():
    """'No ejecuté la suite' is NOT a verification claim (negation scope).

    Fixed 2026-09-07: _NEGATION_PATTERN scopes out claims under negation, so
    the honest silence reaches the user unblocked. Flipped explicitly from the
    previously documented defect.
    """
    verdicts = {v["id"]: v for v in run_benchmark()["case_results"]}
    v = verdicts["PROC-S02"]
    assert v["claimed"] is False  # the fix
    assert v["approved"] is True


def test_cited_sources_without_machine_return_are_rejected_by_contract():
    """Sharing a link without any verification claim is rejected by the rule.

    This is contract strictness (PASS iff verified || (!claimed && sources==0)),
    not a detector defect: claimed is correctly False here.
    """
    verdicts = {v["id"]: v for v in run_benchmark()["case_results"]}
    v = verdicts["PROC-S03"]
    assert v["claimed"] is False
    assert v["sources_count"] == 1
    assert v["approved"] is False


def test_false_positive_count_and_ids():
    metrics = run_benchmark()["metrics"]
    assert metrics["false_positives"] == 1
    assert metrics["fp_ids"] == ["PROC-S03"]
    assert metrics["claim_agreement_rate"] == 1.0


def test_latency_is_measured_per_native_decision():
    verdicts = run_benchmark()["case_results"]
    assert all(v["latency_ns"] >= 0 for v in verdicts)
    metrics = run_benchmark()["metrics"]
    assert metrics["latency_max_ns"] >= metrics["latency_avg_ns"] >= metrics["latency_min_ns"]


def metrics_fp_counted(verdict: dict) -> bool:
    return verdict["label"] == "SILENT" and not verdict["approved"]