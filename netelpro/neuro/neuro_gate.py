"""Netelpro NeuroGate: High-level neuro-symbolic contract validation for LLMs.

Enforces deterministic compile-checked invariances over LLM tool-calling,
structured actions, and discrete parameters under a fail-closed contract.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from netelpro.gate import Gate, GateError


class NetelproNeuroGate:
    """Neuro-symbolic decision gate with formal fail-closed guarantees."""

    def __init__(
        self,
        rule_path: str | Path,
        defn_name: str = "filter-rule",
        fallback_action: Any = None,
    ) -> None:
        self.rule_path = Path(rule_path)
        self.gate = Gate(self.rule_path, defn_name=defn_name)
        self.fallback_action = fallback_action
        self.history: list[dict[str, Any]] = []

    def verify(
        self,
        *rule_args: Any,
        action_payload: Any = None,
    ) -> tuple[bool, Any, str | None, float]:
        """Verifies candidate parameters against the compiled rule.

        Returns:
            (allow: bool, action: Any, reason: str | None, latency_us: float)
        """
        t0 = time.perf_counter_ns()
        allow, reason = self.gate.check(*rule_args)
        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0

        if allow:
            effective_action = action_payload
        else:
            effective_action = self.fallback_action

        audit_entry = {
            "args": list(rule_args),
            "allow": allow,
            "reason": reason,
            "action": effective_action,
            "latency_us": latency_us,
        }
        self.history.append(audit_entry)

        return allow, effective_action, reason, latency_us

    def last_audit(self) -> dict[str, Any] | None:
        return self.history[-1] if self.history else None
