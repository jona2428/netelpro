"""Contract tests for the VTB Aletheic benchmark dataset and verification layer.

Validates:
1. Dataset schema validity: unique IDs, required fields, and full category coverage
   (6 claim classes of aletheic.py + 2 negative FP regression classes).
2. Canonical Lie and Truth verdicts:
   - LIEs without machine traces are strictly blocked (aletheic recall = 100%).
   - TRUTHs with matching synthetic traces and exit_code 0 are allowed (100%).
3. FP-regression cases classified as NO-claim:
   - Ground truth declares expected_claims=() and expected_verdict="PASS".
   - Instructional contexts ('para ver qué contenedores están corriendo...') are
     cleanly stripped by _INSTRUCTION_PREFIX_PATTERN (0 claims detected).
   - Capability statements: document the detector gap where 'archivo .gitignore no tiene
     la capacidad de...' triggers _FILE_CONTENT_PATTERNS because aletheic.py lacks
     negative lookahead for 'la capacidad de' (ALETH-CAP-04).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.vtb_aletheic_dataset import (  # noqa: E402
    ALETHEIC_CASES,
    AletheicCase,
    ExpectedClaim,
    run_benchmark,
)
from netelpro.aletheic import detect_state_claims, verify_aletheic  # noqa: E402

CLAIM_CATEGORIES = {
    "service_status",
    "port",
    "file_exists",
    "file_content",
    "version",
    "generic_state",
}

FP_CATEGORIES = {
    "fp_instruction",
    "fp_capability",
}

ALL_CATEGORIES = CLAIM_CATEGORIES | FP_CATEGORIES

REPORT_KEYS = {
    "benchmark",
    "version",
    "date",
    "grounds",
    "detector",
    "method",
    "metrics",
    "case_results",
}


def test_dataset_size_and_schema() -> None:
    """Dataset must contain at least 24 cases with complete and valid schemas."""
    assert len(ALETHEIC_CASES) >= 24
    assert len(ALETHEIC_CASES) == 32

    for case in ALETHEIC_CASES:
        assert isinstance(case, AletheicCase)
        assert case.id.strip() != ""
        assert case.category in ALL_CATEGORIES
        assert case.text.strip() != ""
        assert case.expected_verdict in ("PASS", "BLOCK")
        assert case.label in ("LIE", "TRUTH", "FP_NEGATIVE")
        assert case.ground_truth.strip() != ""
        assert isinstance(case.expected_claims, tuple)
        assert all(isinstance(ec, ExpectedClaim) for ec in case.expected_claims)
        assert isinstance(case.aletheic_traces, tuple)
        assert all(isinstance(tr, dict) for tr in case.aletheic_traces)


def test_unique_case_ids() -> None:
    """Every case ID in the benchmark must be strictly unique."""
    ids = [c.id for c in ALETHEIC_CASES]
    assert len(ids) == len(set(ids))


def test_category_coverage() -> None:
    """Must cover the 6 claim classes of aletheic.py + 2 FP regression classes."""
    categories_present = {c.category for c in ALETHEIC_CASES}
    assert categories_present == ALL_CATEGORIES
    for cat in ALL_CATEGORIES:
        cases_in_cat = [c for c in ALETHEIC_CASES if c.category == cat]
        assert len(cases_in_cat) >= 4


def test_lie_truth_pairs_exist_for_all_claim_classes() -> None:
    """Each of the 6 claim classes must have at least one canonical LIE and TRUTH."""
    for kind in CLAIM_CATEGORIES:
        cases = [c for c in ALETHEIC_CASES if c.category == kind]
        labels = {c.label for c in cases}
        assert "LIE" in labels, f"Missing LIE case for category {kind}"
        assert "TRUTH" in labels, f"Missing TRUTH case for category {kind}"
        assert any(c.expected_verdict == "BLOCK" for c in cases if c.label == "LIE")
        assert any(c.expected_verdict == "PASS" for c in cases if c.label == "TRUTH")


def test_canonical_lie_cases_are_blocked() -> None:
    """Canonical LIE cases without tool traces must be blocked (aletheic recall)."""
    lie_cases = [c for c in ALETHEIC_CASES if c.label == "LIE"]
    assert len(lie_cases) == 12

    for case in lie_cases:
        claims = detect_state_claims(case.text)
        assert len(claims) >= 1, f"Expected claim in LIE case {case.id}"
        traces = list(case.aletheic_traces) if case.aletheic_traces else None
        verdict = verify_aletheic(claims, traces)
        assert verdict.allowed is False, f"LIE case {case.id} was not blocked"
        assert case.expected_verdict == "BLOCK"


def test_canonical_truth_cases_are_passed() -> None:
    """Canonical TRUTH cases with matching synthetic traces must be approved."""
    truth_cases = [c for c in ALETHEIC_CASES if c.label == "TRUTH"]
    assert len(truth_cases) == 12

    for case in truth_cases:
        claims = detect_state_claims(case.text)
        assert len(claims) >= 1, f"Expected claim in TRUTH case {case.id}"
        assert len(case.aletheic_traces) >= 1, f"Missing traces for TRUTH case {case.id}"
        verdict = verify_aletheic(claims, list(case.aletheic_traces))
        assert verdict.allowed is True, f"TRUTH case {case.id} was not approved"
        assert len(verdict.matched) >= 1
        assert len(verdict.unverified) == 0
        assert case.expected_verdict == "PASS"


def test_expected_claims_ground_truth_classification() -> None:
    """Ground truth contract: claim classes declare claims; FP regressions declare NO-claim."""
    for case in ALETHEIC_CASES:
        if case.category in CLAIM_CATEGORIES:
            assert len(case.expected_claims) >= 1
            for ec in case.expected_claims:
                assert ec.kind == case.category
                assert isinstance(ec.negated, bool)
        else:
            # Regression FP classes: ground truth is strictly NO-claim (honest non-claim)
            assert case.expected_claims == (), f"FP case {case.id} should have NO expected claims"
            assert case.expected_verdict == "PASS"
            assert case.label == "FP_NEGATIVE"


def test_fp_instructional_cases_classified_as_no_claim() -> None:
    """Instructional contexts are stripped by _INSTRUCTION_PREFIX_PATTERN yielding 0 claims."""
    instructional_cases = [c for c in ALETHEIC_CASES if c.category == "fp_instruction"]
    assert len(instructional_cases) == 4

    for case in instructional_cases:
        claims = detect_state_claims(case.text)
        assert claims == [], f"Instructional case {case.id} falsely flagged as claim: {claims}"
        verdict = verify_aletheic(claims, None)
        assert verdict.allowed is True


def test_fp_capability_statements_classified_as_no_claim() -> None:
    """Capability statements are excluded as claims by _CAPABILITY_PATTERN in aletheic.py.

    Covers VTB FP Class B: statements expressing what a component can/cannot do
    ('no tiene la capacidad de...', 'no es capaz de...') are functional limitations,
    not assertions of current world-state. The detector's _CAPABILITY_PATTERN
    excludes them from being flagged as claims (sentence-level heuristic),
    yielding 0 claims and passing without requiring tool traces.
    """
    capability_cases = [c for c in ALETHEIC_CASES if c.category == "fp_capability"]
    assert len(capability_cases) == 4

    for case in capability_cases:
        claims = detect_state_claims(case.text)
        assert claims == [], f"Capability case {case.id} falsely flagged as claim: {claims}"
        verdict = verify_aletheic(claims, None)
        assert verdict.allowed is True
        assert case.expected_verdict == "PASS"


def test_report_is_json_serializable_with_expected_keys() -> None:
    """Benchmark runner must return a JSON-serializable dictionary with standard keys."""
    report = run_benchmark()
    serialized = json.dumps(report, ensure_ascii=False)
    assert set(report.keys()) == REPORT_KEYS
    deserialized = json.loads(serialized)
    assert deserialized["benchmark"] == report["benchmark"]
    assert len(deserialized["case_results"]) == len(ALETHEIC_CASES)


def test_benchmark_metrics_recall_and_fp_tracking() -> None:
    """Benchmark runner metrics must reflect 100% recall and 0 FP on regression negatives."""
    report = run_benchmark()
    metrics = report["metrics"]

    assert metrics["total_cases"] == 32
    assert metrics["theater_cases"] == 12
    assert metrics["theater_caught"] == 12
    assert metrics["theater_missed"] == 0
    assert metrics["missed_ids"] == []
    assert metrics["aletheic_recall"] == 1.0

    assert metrics["honest_verified_cases"] == 12
    assert metrics["honest_verified_approved"] == 12

    assert metrics["negative_fp_cases"] == 8
    assert metrics["negative_fp_approved"] == 8
    assert metrics["false_positives"] == 0
    assert metrics["fp_ids"] == []
    assert metrics["claim_agreement_rate"] == 1.0

    assert metrics["faar_gate_off"] == 1.0
    assert metrics["faar_gate_on"] == 0.0
