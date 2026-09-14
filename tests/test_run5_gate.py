"""Tests for benchmarks/run5_gate.py and benchmarks/train_qlora_dora_run5_kaggle.ipynb.

Uses the repo's real Netelpro compiler on 3 mandatory samples:
1. One valid truth-table (must pass)
2. One invalid truth-table (must fail)
3. One empty output (must fail)
Also validates extract_block, gate_report, and nbformat JSON validity of the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from benchmarks.run5_gate import (
    CompileVerdict,
    compile_verdict,
    extract_block,
    extract_cases,
    gate_report,
)

# 1. Valid sample: canonically formatted netelpro truth-table contract and cases
VALID_SAMPLE = """```netelpro
(truth-table filter-rule
  (a-ok : (Int 0 1))
  (b-ok : (Int 0 1))
  (c-ok : (Int 0 1))
  ((0 0 0) -> 0)
  ((0 0 1) -> 0)
  ((0 1 0) -> 0)
  ((0 1 1) -> 0)
  ((1 0 0) -> 0)
  ((1 0 1) -> 0)
  ((1 1 0) -> 0)
  ((1 1 1) -> 1)
  ((_ _ _) -> 0))
```
```netelpro-cases
(1,1,1) -> 1
(0,1,1) -> 0
(1,1,0) -> 0
(1,0,1) -> 0
```"""

# 2. Invalid sample: malformed syntax (uncovered combinations and missing default row)
INVALID_SAMPLE = """```netelpro
(truth-table filter-rule
  (a-ok : (Int 0 1))
  (b-ok : (Int 0 1))
  ((1 1) -> 1)
)
```"""

# 3. Empty sample
EMPTY_SAMPLE = ""


def test_valid_truth_table_sample_passes():
    """Real compiler on sample 1: valid truth-table must compile and pass."""
    verdict = compile_verdict(VALID_SAMPLE)
    assert verdict.ok is True
    assert bool(verdict) is True
    assert verdict == True  # noqa: E712
    assert verdict.error is None
    assert verdict.contract is not None
    assert "truth-table filter-rule" in verdict.contract
    assert verdict.rule_filter is not None
    # Real execution check via rule_filter
    assert verdict.rule_filter.decide(1, 1, 1) is True
    assert verdict.rule_filter.decide(0, 0, 0) is False


def test_invalid_truth_table_sample_fails():
    """Real compiler on sample 2: invalid syntax must fail fail-closed."""
    verdict = compile_verdict(INVALID_SAMPLE)
    assert verdict.ok is False
    assert bool(verdict) is False
    assert verdict == False  # noqa: E712
    assert verdict.error is not None
    assert "rejected" in verdict.error.lower() or "error" in verdict.error.lower()


def test_empty_output_sample_fails():
    """Real compiler on sample 3: empty output must fail cleanly."""
    verdict = compile_verdict(EMPTY_SAMPLE)
    assert verdict.ok is False
    assert bool(verdict) is False
    assert verdict == False  # noqa: E712
    assert verdict.error is not None
    assert "empty" in verdict.error.lower()

    # Also test whitespace-only
    verdict_ws = compile_verdict("   \n\t  ")
    assert verdict_ws.ok is False


def test_extract_block():
    """Test extracting netelpro and netelpro-cases blocks."""
    contract = extract_block(VALID_SAMPLE, fence="netelpro")
    assert contract is not None
    assert contract.startswith("(truth-table filter-rule")
    assert contract.endswith("((_ _ _) -> 0))")

    cases_block = extract_block(VALID_SAMPLE, fence="netelpro-cases")
    assert cases_block is not None
    assert "(1,1,1) -> 1" in cases_block

    # Test raw code without fences
    raw_code = "(truth-table filter-rule (a : (Int 0 1)) ((0) -> 0) ((1) -> 1) ((_) -> 0))"
    extracted_raw = extract_block(raw_code, fence="netelpro")
    assert extracted_raw == raw_code

    # Non-existent fence
    assert extract_block(VALID_SAMPLE, fence="nonexistent") is None


def test_extract_cases():
    """Test parsing case lines into tuples."""
    cases = extract_cases(VALID_SAMPLE)
    assert cases is not None
    assert len(cases) == 4
    assert cases[0] == ((1, 1, 1), 1)
    assert cases[1] == ((0, 1, 1), 0)


def test_gate_report_statistics():
    """Test gate_report aggregates verdicts correctly."""
    v_pass = compile_verdict(VALID_SAMPLE)
    v_fail = compile_verdict(INVALID_SAMPLE)
    v_empty = compile_verdict(EMPTY_SAMPLE)

    rep = gate_report([v_pass, v_fail, v_empty])
    assert rep["total"] == 3
    assert rep["passed"] == 1
    assert rep["failed"] == 2
    assert pytest.approx(rep["compile_rate"], 0.01) == 1.0 / 3.0
    assert rep["passed_gate"] is True
    assert "RUN #5 GATE REPORT" in rep["report_text"]
    assert "1/3" in rep["report_text"]


def test_gate_report_zero_compile_rate_raises():
    """Test gate_report raises visible GATE_FAILED error if compile rate is 0."""
    v_fail = compile_verdict(INVALID_SAMPLE)
    v_empty = compile_verdict(EMPTY_SAMPLE)

    # Without raise_on_zero
    rep = gate_report([v_fail, v_empty], raise_on_zero=False)
    assert rep["compile_rate"] == 0.0
    assert rep["passed_gate"] is False

    # With raise_on_zero=True
    with pytest.raises(RuntimeError, match="GATE_FAILED: Compile rate is 0.0%"):
        gate_report([v_fail, v_empty], raise_on_zero=True)


def test_run5_notebook_is_valid_nbformat_json():
    """Validate that benchmarks/train_qlora_dora_run5_kaggle.ipynb is valid nbformat JSON."""
    repo_root = Path(__file__).resolve().parents[1]
    nb_path = repo_root / "benchmarks" / "train_qlora_dora_run5_kaggle.ipynb"
    assert nb_path.exists(), f"Notebook does not exist at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Validate standard nbformat v4 structure
    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    assert nb.get("nbformat_minor") in (2, 4, 5)
    assert "cells" in nb, "Notebook missing 'cells' key"
    assert "metadata" in nb, "Notebook missing 'metadata' key"
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 10

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    markdown_cells = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
    assert len(code_cells) >= 5
    assert len(markdown_cells) >= 5

    # Check that all code cells have proper structure and NEVER use !pip
    all_code = []
    for idx, cell in enumerate(code_cells):
        assert "execution_count" in cell
        assert "outputs" in cell
        assert isinstance(cell["outputs"], list)
        assert "source" in cell
        src_lines = cell["source"]
        for line in src_lines:
            assert not line.strip().startswith("!pip"), (
                f"Found forbidden !pip in code cell: {line.strip()}"
            )
        all_code.append("".join(src_lines))

    combined_code = "\n".join(all_code)

    # Validate required specifications in notebook code
    assert "bitsandbytes==0.50.2" in combined_code
    assert "peft==0.14.0" in combined_code
    assert "transformers==4.48.3" in combined_code
    assert "trl==0.15.2" in combined_code
    assert "datasets==3.2.0" in combined_code
    assert "use_dora=True" in combined_code
    assert "r=128" in combined_code
    assert "lora_alpha=256" in combined_code
    assert "lora_dropout=0.05" in combined_code
    assert "HONESTY_SYSTEM_PROMPT" in combined_code
    assert "train_mix.jsonl" in combined_code
    assert "heldout.jsonl" in combined_code
    assert "GATE_FAILED" in combined_code
    assert "q4_k_m" in combined_code
    assert "HF_TOKEN" in combined_code
