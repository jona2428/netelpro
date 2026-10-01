"""examples/receipts_demo.py must stay honest: it runs, passes, exits 0,
and its printed tally agrees with the PASS/FAIL marks it emitted. Same
contract as tests/test_gate_demo.py, for the same reason: a demo that
grades itself wrongly is worse than one that fails loudly.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "examples" / "receipts_demo.py"

pytestmark = pytest.mark.skipif(not DEMO.exists(), reason="receipts_demo.py not present")


@pytest.fixture(scope="module")
def run() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def test_receipts_demo_runs_and_passes(run: subprocess.CompletedProcess[str]) -> None:
    assert run.returncode == 0, f"exit {run.returncode}\n--- stdout ---\n{run.stdout}\n--- stderr ---\n{run.stderr}"
    assert "checks passed" in run.stdout
    assert "FAILED" not in run.stdout
    assert "[FAIL]" not in run.stdout


def test_receipts_demo_tally_is_consistent(run: subprocess.CompletedProcess[str]) -> None:
    passed = len(re.findall(r"\[PASS\]", run.stdout))
    failed = len(re.findall(r"\[FAIL\]", run.stdout))
    tally = re.search(r"(\d+)/(\d+) checks passed", run.stdout)
    assert tally, run.stdout
    assert int(tally.group(1)) == passed == int(tally.group(2))
    assert failed == 0


def test_receipts_demo_shows_the_evidence(run: subprocess.CompletedProcess[str]) -> None:
    """The four facts the demo exists to show, asserted on the output."""
    out = run.stdout
    assert "config/settings.py REJECTED" in out
    assert "sha256 unchanged" in out
    assert "chain" in out and "refused" in out
    assert "rows checked: 40" in out


def test_receipts_demo_output_is_plain_when_piped(run: subprocess.CompletedProcess[str]) -> None:
    assert "\033[" not in run.stdout
