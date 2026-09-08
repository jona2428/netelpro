"""Generic compiled gate: fail-closed action approval for any host.

Extracts the Neuromancer guard pattern (verification_guard.py, zone
policy, shutdown guard) as a standalone product, decoupled from any
agent framework. A `.sl` `filter-rule` is compiled to native code via
RuleFilter; every failure mode that is not a real decision DENIES with
an explicit reason (fail-closed contract).

Strict surface (raises):
- Gate(rule_path): load + compile; GateError on unreadable file or
  compile failure (coords preserved from RuleFilterError).
- gate.decide(*args): raw compiled decision; rule errors propagate as
  RuleFilterError for hosts that prefer strict mode.
- open_gate(rule_path): strict constructor alias.

Fail-closed surface (never raises):
- gate.check(*args) -> (allow, reason): reason is None exactly when the
  compiled rule decided (True or False); any string reason is a failure
  and always pairs with allow=False (missing file, compile error, arity
  or type misuse, runtime error, sorry hole, non-bool return).
- check_file(rule_path, *args): one-shot fail-closed including gate
  construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from netelpro.rule_filter import RuleFilter, RuleFilterError

__all__ = ["Gate", "GateError", "check_file", "open_gate"]


class GateError(Exception):
    """The gate could not be constructed (unreadable rule file or compile failure)."""


def _compile(source: str, defn_name: str, origin: Path) -> RuleFilter:
    try:
        return RuleFilter(source, defn_name)
    except RuleFilterError as e:
        raise GateError(f"rule compile failed: {origin}: {e}") from e


class Gate:
    """Fail-closed approval gate over a compiled .sl filter-rule.

    Thin bridge: holds a RuleFilter (compiled native rule) and exposes
    the strict (decide) and fail-closed (check) surfaces. Knows nothing
    about Neuromancer, repo layouts, singletons, or logging.
    """

    def __init__(self, rule_path: str | Path, defn_name: str = "filter-rule") -> None:
        self.rule_path = Path(rule_path)
        self.defn_name = defn_name
        try:
            source = self.rule_path.read_text(encoding="utf-8")
        except OSError as e:
            raise GateError(f"rule file not readable: {self.rule_path}: {e}") from e
        self._filter = _compile(source, defn_name, self.rule_path)

    def decide(self, *args: Any) -> bool:
        """Raw compiled decision. Rule errors (RuleFilterError) propagate."""
        return self._filter.decide(*args)

    def check(self, *args: Any) -> tuple[bool, str | None]:
        """Fail-closed decision: (allow, reason). Never raises.

        Contract: (True, None) and (False, None) are real decisions by
        the compiled rule; (False, reason) with reason non-None is a
        failure -- the gate never fails open and never fails silently.
        """
        try:
            allowed = self._filter.decide(*args)
        except RuleFilterError as e:
            return False, f"gate misuse: {e}"
        except Exception as e:
            # Broad catch IS the contract here: fail-closed with a
            # declared reason, not a silent suppress (see module doc).
            return False, f"gate denied by failure ({type(e).__name__}): {e}"
        if type(allowed) is not bool:
            return False, f"gate denied: rule returned non-bool ({type(allowed).__name__})"
        return allowed, None

    def manifest(self) -> list[str]:
        """Declared sorry holes of the rule, as human-readable diagnostics."""
        return self._filter.manifest()

    def verify(self, cases: Any) -> Any:
        """Differential native-vs-interpreter check; delegates to RuleFilter."""
        return self._filter.verify(cases)


def open_gate(rule_path: str | Path, defn_name: str = "filter-rule") -> Gate:
    """Open a Gate in strict mode: raises GateError on any failure."""
    return Gate(rule_path, defn_name)


def check_file(
    rule_path: str | Path,
    *args: Any,
    defn_name: str = "filter-rule",
) -> tuple[bool, str | None]:
    """One-shot fail-closed check of a rule file: never raises."""
    try:
        gate = Gate(rule_path, defn_name)
    except GateError as e:
        return False, f"gate unavailable: {e}"
    return gate.check(*args)