"""Run #5 Gate Evaluator: Real Netelpro truth-table compilation & case arbitration.

Extracts netelpro syntax and cases blocks from model outputs, executes the real
LLVM/host compiler (compile_filter), and reports verification statistics.
Used by benchmarks/train_qlora_dora_run5_kaggle.ipynb to gate model export.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure netelpro is importable if running in a repo clone or subfolder
try:
    from netelpro.rule_filter import RuleFilterError, compile_filter
except ImportError:
    for cand in [
        Path("/kaggle/working/netelpro"),
        Path("netelpro"),
        Path(__file__).resolve().parents[1] if "__file__" in globals() else None,
        Path.cwd() / "netelpro",
    ]:
        if cand and cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            if (cand / "src").exists() and str(cand / "src") not in sys.path:
                sys.path.insert(0, str(cand / "src"))
            break
    from netelpro.rule_filter import RuleFilterError, compile_filter

CASE_LINE_RE = re.compile(r"^\s*\(\s*([01][01,\s]*)\s*\)\s*->\s*([01]|true|false)\s*$")


def extract_block(text: str, fence: str = "netelpro") -> Optional[str]:
    """Extract fenced code block from text (e.g. ```netelpro or ```netelpro-cases).

    If fences are present, returns the last matching block stripped.
    If no fences are present in text, and fence is 'netelpro', checks if the text
    itself is raw Lisp/Netelpro code starting with '(' and returns it stripped.
    Returns None if text is empty or no block could be extracted.
    """
    if not text or not text.strip():
        return None

    pattern = rf"```{re.escape(fence)}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    pattern_loose = rf"```{re.escape(fence)}\s*(.*?)```"
    matches_loose = re.findall(pattern_loose, text, re.DOTALL)
    if matches_loose:
        return matches_loose[-1].strip()

    # Raw code fallback: if text doesn't contain markdown fences
    if "```" not in text:
        stripped = text.strip()
        if fence == "netelpro" and stripped.startswith("("):
            return stripped

    return None


def extract_cases(text: str) -> Optional[List[Tuple[Tuple[int, ...], int]]]:
    """Extract cases from ```netelpro-cases``` block or raw case lines.

    Returns list of ((arg0, arg1, ...), expected_result).
    """
    cases_block = extract_block(text, fence="netelpro-cases")
    source_lines = (cases_block if cases_block is not None else text).splitlines()
    cases: List[Tuple[Tuple[int, ...], int]] = []
    for line in source_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        m = CASE_LINE_RE.match(line_clean)
        if m:
            args = tuple(int(t) for t in re.split(r"[,\s]+", m.group(1).strip()) if t)
            exp = 1 if m.group(2).lower() in ("1", "true") else 0
            cases.append((args, exp))
    return cases if cases else None


@dataclass
class CompileVerdict:
    """Verdict of compiling and verifying a netelpro truth-table generation."""

    ok: bool
    error: Optional[str] = None
    contract: Optional[str] = None
    cases: Optional[List[Tuple[Tuple[int, ...], int]]] = None
    mismatches: Optional[List[Any]] = None
    rule_filter: Any = None

    def __bool__(self) -> bool:
        return self.ok

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self.ok == other
        return super().__eq__(other)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


def compile_verdict(
    text_or_contract: str,
    cases: Optional[List[Tuple[Tuple[int, ...], int]]] = None,
    verify_cases: bool = False,
) -> CompileVerdict:
    """Compile contract and optionally verify cases with the real Netelpro compiler.

    Fail-closed: Any syntax error, bad types, missing default row, or engine error
    results in ok=False.

    Args:
        text_or_contract: Raw model generation or extracted contract string.
        cases: Optional list of test cases ((args), expected). If None and verify_cases
               is True, attempts to extract cases from ```netelpro-cases``` block.
        verify_cases: If True, also checks that test cases match the compiled rule.
                      Defaults to False (pure compilation check), but automatically
                      activates if explicit cases are provided or if netelpro-cases
                      fence is detected in text.

    Returns:
        CompileVerdict instance.
    """
    if not text_or_contract or not text_or_contract.strip():
        return CompileVerdict(ok=False, error="empty output", contract=None)

    # If markdown fences present, extract the netelpro block
    if "```netelpro" in text_or_contract:
        contract = extract_block(text_or_contract, fence="netelpro")
        if not contract:
            return CompileVerdict(
                ok=False, error="unclosed or empty ```netelpro``` block", contract=None
            )
    elif "```" in text_or_contract:
        contract = extract_block(text_or_contract, fence="lisp") or extract_block(
            text_or_contract, fence=""
        )
        if not contract:
            contract = text_or_contract.strip()
    else:
        contract = text_or_contract.strip()

    # Attempt compilation with real Netelpro compiler
    try:
        rf = compile_filter(contract)
    except RuleFilterError as exc:
        return CompileVerdict(
            ok=False, error=f"compilation rejected: {exc}", contract=contract
        )
    except Exception as exc:
        return CompileVerdict(
            ok=False, error=f"compiler error: {exc!r}", contract=contract
        )

    # Check cases if provided or present
    extracted_cases = cases
    if extracted_cases is None and ("```netelpro-cases" in text_or_contract or verify_cases):
        extracted_cases = extract_cases(text_or_contract)

    mismatches: List[Any] = []
    if extracted_cases:
        try:
            mismatches = rf.verify_int(extracted_cases)
            if mismatches:
                return CompileVerdict(
                    ok=False,
                    error=f"{len(mismatches)} case mismatches",
                    contract=contract,
                    cases=extracted_cases,
                    mismatches=mismatches,
                    rule_filter=rf,
                )
        except Exception as exc:
            return CompileVerdict(
                ok=False,
                error=f"case verification failed: {exc!r}",
                contract=contract,
                cases=extracted_cases,
                mismatches=[],
                rule_filter=rf,
            )

    return CompileVerdict(
        ok=True,
        error=None,
        contract=contract,
        cases=extracted_cases,
        mismatches=[],
        rule_filter=rf,
    )


def gate_report(
    results: List[Union[CompileVerdict, Dict[str, Any], str, bool]],
    min_rate: float = 0.0,
    raise_on_zero: bool = False,
) -> Dict[str, Any]:
    """Generate summary report and compile rate from gate sample verdicts.

    Args:
        results: List of CompileVerdict instances, dicts with 'ok', or raw strings to evaluate.
        min_rate: Minimum compile rate required to pass (0.0 means > 0).
        raise_on_zero: If True and compile_rate == 0.0, raises RuntimeError("GATE_FAILED: ...").

    Returns:
        Dict with keys:
            total: int
            passed: int
            failed: int
            compile_rate: float (0.0 to 1.0)
            compile_rate_pct: float (0.0 to 100.0)
            passed_gate: bool
            per_sample: list of dicts
            report_text: formatted string summary
    """
    verdicts: List[CompileVerdict] = []
    for item in results:
        if isinstance(item, CompileVerdict):
            verdicts.append(item)
        elif isinstance(item, dict):
            ok = bool(item.get("ok", False))
            err = item.get("error")
            verdicts.append(
                CompileVerdict(ok=ok, error=err, contract=item.get("contract"))
            )
        elif isinstance(item, bool):
            verdicts.append(CompileVerdict(ok=item))
        elif isinstance(item, str):
            verdicts.append(compile_verdict(item))
        else:
            verdicts.append(CompileVerdict(ok=bool(item)))

    total = len(verdicts)
    passed = sum(1 for v in verdicts if v.ok)
    failed = total - passed
    compile_rate = (passed / total) if total > 0 else 0.0
    compile_rate_pct = compile_rate * 100.0
    passed_gate = compile_rate > min_rate if min_rate == 0.0 else compile_rate >= min_rate

    lines = [
        "=" * 65,
        "🚪 RUN #5 GATE REPORT — NETELPRO TRUTH-TABLE GRAMMAR COMPILE RATE",
        "=" * 65,
        f"Total Samples: {total} | Passed: {passed} | Failed: {failed}",
        f"Compile Rate:  {compile_rate_pct:.1f}% ({passed}/{total})",
        f"Gate Status:   {'PASSED ✅' if passed_gate else 'FAILED ❌'}",
        "-" * 65,
        f"{'Sample':<8} {'Status':<10} {'Details / Error'}",
        "-" * 65,
    ]
    per_sample_info = []
    for idx, v in enumerate(verdicts, 1):
        status_str = "PASS ✅" if v.ok else "FAIL ❌"
        detail_str = "Compiled successfully" if v.ok else (v.error or "Unknown error")
        lines.append(f"{idx:<8} {status_str:<10} {detail_str[:45]}")
        per_sample_info.append({
            "sample_index": idx,
            "ok": v.ok,
            "error": v.error,
            "contract": v.contract,
        })
    lines.append("=" * 65)
    report_text = "\n".join(lines)

    if raise_on_zero and compile_rate == 0.0:
        raise RuntimeError(
            f"GATE_FAILED: Compile rate is 0.0% ({passed}/{total}). "
            "Model failed to inject Netelpro grammar. Halting before export."
        )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "compile_rate": compile_rate,
        "compile_rate_pct": compile_rate_pct,
        "passed_gate": passed_gate,
        "per_sample": per_sample_info,
        "report_text": report_text,
    }
