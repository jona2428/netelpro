"""The demo must stay honest: it runs, it passes, and it exits 0.

`examples/gate_demo.py` is the front door -- the thing a stranger runs
first. If it silently breaks (a renamed rule, a changed Gate API, a
verdict that stops matching its own policy), the first impression is a
lie. So it is tested like any other artifact, not left to manual runs.

Two properties matter and are checked separately:

1. The demo completes with exit code 0 -- every internal check passed.
2. The exit code is not hardcoded: a demo whose checks all pass while
   printing FAIL would be the exact failure mode this file exists to
   prevent, so the PASS/FAIL tally in stdout is cross-checked against
   the exit code.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "examples" / "gate_demo.py"

pytestmark = pytest.mark.skipif(
    not DEMO.exists(),
    reason="gate_demo.py not present",
)


def _run_demo() -> subprocess.CompletedProcess[str]:
    """Run the demo in a clean subprocess, exactly as a stranger would."""
    return subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def test_gate_demo_runs_and_passes() -> None:
    """The demo must exit 0: all 12 of its own checks passed."""
    result = _run_demo()
    assert result.returncode == 0, (
        f"gate_demo.py exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "checks passed" in result.stdout
    assert "FAILED" not in result.stdout


def test_gate_demo_reports_zero_failures() -> None:
    """No check in the demo may report FAIL, whatever the exit code says."""
    result = _run_demo()
    assert "[FAIL]" not in result.stdout, result.stdout


def test_gate_demo_tally_is_consistent() -> None:
    """The printed tally must agree with the number of PASS marks emitted.

    Guards the specific dishonesty of printing 'N/N checks passed' while
    individual checks printed FAIL -- a demo that grades itself wrongly
    is worse than one that fails loudly.
    """
    result = _run_demo()
    stdout = result.stdout

    passed_marks = len(re.findall(r"\[PASS\]", stdout))
    failed_marks = len(re.findall(r"\[FAIL\]", stdout))

    tally = re.search(r"(\d+)/(\d+) checks passed", stdout)
    assert tally, f"no tally line found in output:\n{stdout}"

    reported_passed = int(tally.group(1))
    reported_total = int(tally.group(2))

    assert reported_total == passed_marks + failed_marks, (
        f"tally says {reported_total} total but "
        f"{passed_marks} PASS + {failed_marks} FAIL marks were printed"
    )
    assert reported_passed == passed_marks
    assert failed_marks == 0


def test_gate_demo_proves_the_four_claims() -> None:
    """The four load-bearing claims of the demo must be visible in its output.

    Each corresponds to a section of the demo; if a section is gutted the
    demo would still exit 0 while proving nothing, so the evidence strings
    are asserted directly.
    """
    result = _run_demo()
    stdout = result.stdout

    # 2. Compiled to native machine code.
    assert "native entry point at 0x" in stdout
    assert "declared sorry holes: 0" in stdout
    # 4. Fail-closed on every failure mode, with a reason.
    assert "fails closed on: missing file" in stdout
    assert "fails closed on: unparseable rule" in stdout
    assert "fails closed on: type misuse" in stdout
    # 5. Real measured throughput.
    assert "decisions per second" in stdout
    # 6. Differential parity native vs interpreter.
    assert "mismatches:    0" in stdout


def test_gate_demo_needs_no_optional_dependencies() -> None:
    """The demo must not import torch, transformers, or hit the network.

    Its whole point is that a stranger can run it with nothing but the
    netelpro package. A torch import would silently make that false on
    machines without a GPU stack.
    """
    source = DEMO.read_text(encoding="utf-8")
    packaged = (REPO_ROOT / "netelpro" / "gate_demo.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "import transformers", "import requests", "urllib.request"):
        for label, body in (("examples/gate_demo.py", source), ("netelpro/gate_demo.py", packaged)):
            assert forbidden not in body, f"{label} must stay dependency-free: found {forbidden!r}"
