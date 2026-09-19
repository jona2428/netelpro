"""One command. Thirty seconds. No GPU, no model, no training, no network.

This is the Netelpro gate as a standalone product: an approval policy
written in lines a human can read, compiled to native machine code,
deciding in microseconds, and refusing to fail open when anything at
all goes wrong.

Nothing here is a mockup. The rule is compiled by the real compiler
(`netelpro.rule_filter.RuleFilter` -> LLVM via llvmlite) and every
decision below is executed by machine code reached through ctypes.
The differential check at the end runs the same cases through the
reference interpreter and compares, so you do not have to trust the
native path on its own.

Run:
    python examples/gate_demo.py

Exit code 0 means every check in this file passed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netelpro.gate import Gate, check_file  # noqa: E402

RULE_PATH = Path(__file__).resolve().parent / "gates" / "expense_approval.sl"

WIDTH = 74

# Colour only when a human is actually looking. Piping the output to a
# file, to `grep`, or to a test harness must yield plain text -- escape
# codes in captured output are noise, and they make the output
# unassertable. Respects the NO_COLOR convention too.
_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str) -> str:
    return code if _COLOR else ""


BOLD = _c("\033[1m")
DIM = _c("\033[2m")
GREEN = _c("\033[92m")
RED = _c("\033[91m")
YELLOW = _c("\033[93m")
CYAN = _c("\033[96m")
RESET = _c("\033[0m")

_checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool) -> None:
    """Record one verifiable assertion of this demo."""
    _checks.append((label, ok))
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {label}")


def section(n: int, title: str) -> None:
    print()
    print(f"{BOLD}{CYAN}{n}. {title}{RESET}")
    print(f"{DIM}{'-' * WIDTH}{RESET}")


def main() -> int:
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    print(f"{BOLD}  NETELPRO GATE — a policy that cannot fail open{RESET}")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")

    # ---------------------------------------------------------------- 1
    section(1, "THE RULE — a human reads it")
    source = RULE_PATH.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    )
    print(f"  {DIM}File:{RESET} {RULE_PATH.name}")
    print()
    for line in body.splitlines():
        print(f"    {line}")
    print()
    print(f"  {DIM}That is the entire policy. No weights, no prompts, no tuning.{RESET}")
    check("rule file is non-empty and parseable as a filter-rule", "defn filter-rule" in source)

    # ---------------------------------------------------------------- 2
    section(2, "COMPILED — to native machine code, once, at startup")
    t0 = time.perf_counter()
    gate = Gate(RULE_PATH)
    compile_ms = (time.perf_counter() - t0) * 1000.0

    entry = getattr(gate._filter, "_resolve_entry_address", None)
    entry_addr = entry() if callable(entry) else None

    print(f"  compiled in {compile_ms:.1f} ms")
    if entry_addr:
        print(f"  native entry point at {entry_addr:#x}  {DIM}(machine code, not a Python function){RESET}")
    holes = gate.manifest()
    print(f"  declared sorry holes: {len(holes)}  {DIM}(an undeclared hole is a compile error){RESET}")
    check("rule compiled without raising", True)
    check("no sorry holes: the rule is complete, nothing is stubbed out", not holes)

    # ---------------------------------------------------------------- 3
    section(3, "DECIDES — real verdicts on real inputs")
    print(f"  {DIM}{'amount':>8}  {'manager':>7}  {'emergency':>9}   verdict{RESET}")
    cases: list[tuple[tuple[int, int, int], bool]] = [
        ((500, 1, 0), True),
        ((501, 1, 0), False),
        ((50, 0, 1), True),
        ((51, 0, 1), False),
        ((500, 0, 0), False),
        ((0, 1, 0), True),
    ]
    all_ok = True
    for args, expected in cases:
        allow, reason = gate.check(*args)
        ok = (allow == expected) and reason is None
        all_ok = all_ok and ok
        mark = f"{GREEN}ALLOW{RESET}" if allow else f"{RED}DENY {RESET}"
        note = "" if reason is None else f"  {YELLOW}({reason}){RESET}"
        print(f"  {args[0]:>8}  {args[1]:>7}  {args[2]:>9}   {mark}{note}")
    check("every verdict matches the policy documented in the rule file", all_ok)
    check(
        "reason is None exactly when the rule decided (not on any failure)",
        all(gate.check(*a)[1] is None for a, _ in cases),
    )

    # ---------------------------------------------------------------- 4
    section(4, "FAILS CLOSED — garbage in, DENY out, with the reason")
    print(f"  {DIM}The important column is the verdict. A gate that fails open is worse{RESET}")
    print(f"  {DIM}than no gate, because you would still believe you were protected.{RESET}")
    print()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        broken = td_path / "broken.sl"
        broken.write_text("(defn filter-rule (a) ((this is not netelpro))", encoding="utf-8")

        failures: list[tuple[str, tuple[bool, str | None]]] = [
            ("missing file", check_file(td_path / "nope.sl", 1, 1, 0)),
            ("unparseable rule", check_file(broken, 1, 1, 0)),
            ("wrong arity (2 args to a 3-arg rule)", gate.check(500, 1)),
            ("wrong arity (5 args to a 3-arg rule)", gate.check(1, 1, 1, 1, 1)),
            # `amount` is declared plain `Int` (i64), so -1 is a legal value
            # and the rule decides it (ALLOW: -1 <= 500 AND manager == 1).
            # That is a policy outcome, not a gate failure -- so the real
            # boundary case to prove here is a TYPE violation: a string
            # where the boundary law requires an i64.
            ("type misuse (string where Int is required)", gate.check("not-an-int", 1, 0)),
        ]

        for label, (allow, reason) in failures:
            denied = (allow is False) and (reason is not None)
            mark = f"{RED}DENY {RESET}" if allow is False else f"{GREEN}ALLOW{RESET}"
            print(f"  {mark} {label}")
            print(f"         {DIM}{reason}{RESET}")
            check(f"fails closed on: {label}", denied)

    # ---------------------------------------------------------------- 5
    section(5, "SPEED — the number you can verify yourself")
    N = 200_000
    gate.check(500, 1, 0)  # warm up
    t0 = time.perf_counter()
    for _ in range(N):
        gate.check(500, 1, 0)
    elapsed = time.perf_counter() - t0
    per_sec = N / elapsed
    per_call_us = (elapsed / N) * 1e6

    print(f"  {BOLD}{per_sec:,.0f}{RESET} decisions per second")
    print(f"  {per_call_us:.2f} microseconds per decision")
    print(f"  {DIM}{N:,} calls, measured just now on this machine,{RESET}")
    print(f"  {DIM}including the Python->ctypes call overhead.{RESET}")
    check("throughput exceeds 100k decisions/second", per_sec > 100_000)

    # ---------------------------------------------------------------- 6
    section(6, "PROVEN — native machine code vs reference interpreter")
    print(f"  {DIM}Same rule, same inputs, two independent execution paths.{RESET}")
    print(f"  {DIM}Any disagreement is reported. An empty list is agreement.{RESET}")
    print()
    mismatches = gate.verify(cases)
    print(f"  cases checked: {len(cases)}")
    print(f"  mismatches:    {len(mismatches)}")
    if mismatches:
        for m in mismatches:
            print(f"    {RED}{m}{RESET}")
    check("native and interpreter agree on every case", not mismatches)

    # ---------------------------------------------------------------- verdict
    failed = [label for label, ok in _checks if not ok]
    print()
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    if failed:
        print(f"{BOLD}{RED}  {len(failed)} of {len(_checks)} checks FAILED{RESET}")
        for label in failed:
            print(f"    - {label}")
        print(f"{BOLD}{'=' * WIDTH}{RESET}")
        return 1
    print(f"{BOLD}{GREEN}  {len(_checks)}/{len(_checks)} checks passed{RESET}")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    print()
    print(f"  {DIM}Every number above came from compiled machine code reached through{RESET}")
    print(f"  {DIM}ctypes on this machine, in this run. No simulation, no mock, no{RESET}")
    print(f"  {DIM}model, no GPU, no network. Read the source and verify it yourself.{RESET}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
