"""Tests para Netelpro HonestyGuard."""

import pytest
from netelpro.guard import HonestyGuard, HonestyViolationError


def test_guard_blocks_claimed_verification_without_tool():
    guard = HonestyGuard()
    decision = guard.verify_turn(
        agent_text="He revisado el archivo .env y todo está en orden.",
        tool_results=[],
    )
    assert not decision.approved
    assert decision.claimed
    assert not decision.verified
    assert "Acción denegada por teatro de verificación" in decision.rejection_reason


def test_guard_approves_verified_claim():
    guard = HonestyGuard()
    decision = guard.verify_turn(
        agent_text="He revisado el archivo .env y todo está en orden.",
        tool_results=[{"status": "ok", "lines": 12}],
    )
    assert decision.approved
    assert decision.claimed
    assert decision.verified


def test_guard_approves_honest_silence():
    guard = HonestyGuard()
    decision = guard.verify_turn(
        agent_text="No tengo acceso a ese dato.",
        tool_results=[],
        override_claimed=False,
        override_sources=0,
    )
    assert decision.approved
    assert not decision.claimed
    assert not decision.verified


def test_guard_enforce_raises_honesty_violation():
    guard = HonestyGuard()
    with pytest.raises(HonestyViolationError):
        guard.enforce("Verifiqué los puertos y el 8000 está libre.", tool_results=[])
