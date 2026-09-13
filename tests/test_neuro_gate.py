"""Tests for NetelproNeuroGate high-level LLM decision interceptor."""

from __future__ import annotations

from pathlib import Path
import pytest

from netelpro.neuro import NetelproNeuroGate

_ROBOT_RULE = Path(__file__).resolve().parents[1] / "examples" / "gates" / "robot_interlock.sl"


def test_neuro_gate_robot_interlock_approved():
    gate = NetelproNeuroGate(
        rule_path=_ROBOT_RULE,
        fallback_action={"status": "HALT", "commanded_speed": 0},
    )

    action = {"status": "MOVE", "commanded_speed": 100}
    # door_open=0, speed=100, estop=0 -> ALLOW
    allow, effective_action, reason, latency_us = gate.verify(0, 100, 0, action_payload=action)

    assert allow is True
    assert effective_action["status"] == "MOVE"
    assert reason is None
    assert latency_us > 0.0


def test_neuro_gate_robot_interlock_denied_fail_closed():
    gate = NetelproNeuroGate(
        rule_path=_ROBOT_RULE,
        fallback_action={"status": "HALT", "commanded_speed": 0},
    )

    # LLM hallucinates an unsafe move command during emergency stop
    unsafe_action = {"status": "MOVE", "commanded_speed": 500}
    # door_open=0, speed=500, estop=1 -> DENY
    allow, effective_action, reason, latency_us = gate.verify(0, 500, 1, action_payload=unsafe_action)

    assert allow is False
    assert effective_action["status"] == "HALT"
    assert effective_action["commanded_speed"] == 0
    assert reason is None  # clean compiled decision
    assert latency_us > 0.0
