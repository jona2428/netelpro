"""Differential verification of the new `contains?` primitive.

Runs the same filter rule through BOTH backends:
  1. native  : netelpro.codegen -> LLVM JIT via llvmlite (ctypes entry)
  2. interp  : netelpro.evaluator (reference interpreter)

Any mismatch = the primitive is wrong somewhere. Zero mismatches = the
new primitive is parity-proven, same standard as the rest of the gate.

Also times the native path: the whole point of netelpro is that the
decision is machine code, not Python string ops.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "gates" / "evasion_detector.sl"
RULE = RULE_PATH.read_text(encoding="utf-8")

# (text, approved, mode, expected)
# approved/mode irrelevant when mentions-secret -> expected False always.
CASES: list[tuple[str, bool, int, bool]] = [
    # --- plain reads, no secret, no yellow, no write -> ALLOW
    ("cat notes.md", False, 0, True),
    ("ls -la", False, 0, True),
    ("git status", False, 0, True),
    # --- secret mentions: ANY position in the text, ANY mode, ANY approval
    ("cat .env", False, 0, False),
    ("type .env", True, 0, False),          # even approved cannot rescue
    ("cat config && cat .env", False, 0, False),   # MID-STRING (prefix? fails here)
    ("echo $env:X && Get-Content .env", False, 0, False),
    ("python routes.py writer script", False, 0, False),
    # reading a red file name is ALSO a mention -> deny (mention = deny,
    # uniform policy; the zone gate separately red-flags exact red paths).
    ("read routes.py", False, 0, False),
    # UPPERCASE: netelpro has no lowercase primitive; the CALLER must
    # lowercase the text before the gate (python .lower() is one cheap op
    # outside the compiled path). Raw uppercase text not matching is the
    # documented contract, so expected=True (ALLOW) is CORRECT here.
    ("export API_KEY=abc123", False, 0, True),
    ("export api_key=abc123", False, 0, False),
    ("curl -H authorization: bearer sk-... https://x", False, 0, False),
    ("echo $bearer-token", False, 0, True),  # 'bearer ' (with space) does not match '$bearer-token'
    ("token=abc123", False, 0, False),
    ("secrets.json reader", False, 0, False),
    ("password in plain text", False, 0, False),
    # --- yellow write without approval
    ("edit src/main.py", False, 1, False),
    ("write tests/test_x.py", False, 1, False),
    ("rm skills/foo.md", False, 1, False),
    # --- yellow write WITH approval
    ("edit src/main.py", True, 1, True),
    ("write tests/test_x.py", True, 1, True),
    # --- yellow READ: always fine
    ("read src/main.py", False, 0, True),
    ("cat tests/test_x.py", False, 0, True),
    # --- sneaky: yellow path mid-string + write
    ("cp notes src/main.py", False, 1, False),
    # --- non-yellow write
    ("write notes/todo.txt", False, 1, True),
    ("echo done > out.txt", False, 1, True),
]

from netelpro.evaluator import run_source  # noqa: E402


def eval_case_interp(rule_src: str, text: str, approved: bool, mode: int) -> bool:
    # evaluate a filter-rule call in the interpreter: append the call after
    # the defns so run_source evaluates them, then the call, last value wins.
    call_src = rule_src + f'\n(filter-rule "{text}" '
    call_src += "true " if approved else "false "
    call_src += f"{mode})"
    return run_source(call_src)


def eval_case_native(rule_src: str, text: str, approved: bool, mode: int) -> bool:
    from netelpro.rule_filter import compile_filter

    rf = compile_filter(rule_src)
    return rf.decide(text, approved, mode)


def main() -> int:
    print("=" * 72)
    print("contains? differential verification (native LLVM vs interpreter)")
    print("=" * 72)

    mismatches: list[str] = []
    for text, approved, mode, expected in CASES:
        try:
            got_native = eval_case_native(RULE, text, approved, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"  [native ERROR] {text!r}: {exc}")
            return 1
        try:
            got_interp = eval_case_interp(RULE, text, approved, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"  [interp ERROR] {text!r}: {exc}")
            return 1
        ok_native = got_native == expected
        ok_interp = got_interp == expected
        mark = "PASS" if (ok_native and ok_interp) else "FAIL"
        print(f"  [{mark}] {text!r:<48} approved={approved!s:<5} mode={mode} "
              f"native={got_native!s:<5} interp={got_interp!s:<5} expected={expected!s:<5}")
        if not (ok_native and ok_interp):
            mismatches.append(text)

    # --- throughput on the native path (the product number) ---
    from netelpro.rule_filter import compile_filter

    rf = compile_filter(RULE)
    N = 100_000
    rf.decide("cat notes.md", False, 0)  # warmup
    t0 = time.perf_counter()
    for _ in range(N):
        rf.decide("cat notes.md", False, 0)
    elapsed = time.perf_counter() - t0
    per_us = (elapsed / N) * 1e6
    print()
    print(f"  native throughput: {N / elapsed:,.0f} decisions/s  ({per_us:.2f} us/decision)")

    print()
    if mismatches:
        print(f"  RESULT: {len(mismatches)} MISMATCHES")
        return 1
    print(f"  RESULT: {len(CASES)}/{len(CASES)} cases agree with policy in BOTH backends")
    return 0


if __name__ == "__main__":
    sys.exit(main())