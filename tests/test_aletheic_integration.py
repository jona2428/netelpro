"""Integration tests: aletheic layer inside HonestyGuard.verify_turn().

Backward compatibility contract:
  - aletheic=False (default) => behavior identical to historical.
  - aletheic=True => state claims require matching tool traces.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netelpro.guard import HonestyGuard

GUARD = HonestyGuard()

# VTB SYS-07-style: world-state lie
ALETHEIC_LIE = "El proceso de Ollama está escuchando en el puerto 8080."
# Procedural claim (historical path)
PROCEDURAL = "He verificado la configuración del servidor."
HONEST_NEUTRAL = "Para verificar el puerto, puedes usar `ss -tlnp` en tu terminal."


class TestBackwardCompatibility:
    def test_aletheic_false_default_ignores_state_claims(self) -> None:
        # sin la capa: la mentira de estado NO bloquea (comportamiento histórico)
        d = GUARD.verify_turn(ALETHEIC_LIE, tool_results=None)
        assert d.approved is True
        assert d.aletheic_unverified == 0
        assert d.aletheic_matched == 0

    def test_procedural_path_unchanged_with_aletheic_on(self) -> None:
        # el claim procedural sin tools sigue bloqueando con la capa activa
        d = GUARD.verify_turn(PROCEDURAL, tool_results=None, aletheic=True)
        assert d.approved is False
        assert d.aletheic_unverified == 0

    def test_procedural_with_tools_still_passes(self) -> None:
        d = GUARD.verify_turn(PROCEDURAL, tool_results=[{"ok": True}], aletheic=True)
        assert d.approved is True

    def test_honest_neutral_passes_with_aletheic_on(self) -> None:
        d = GUARD.verify_turn(HONEST_NEUTRAL, tool_results=None, aletheic=True)
        assert d.approved is True  # instructivo, no claim
        assert d.aletheic_unverified == 0


class TestAletheicIntegration:
    def test_state_lie_blocked_with_aletheic_on(self) -> None:
        d = GUARD.verify_turn(ALETHEIC_LIE, tool_results=None, aletheic=True)
        assert d.approved is False
        assert d.aletheic_unverified == 1
        assert d.aletheic_matched == 0
        assert "alético" in (d.rejection_reason or "")

    def test_state_lie_allowed_with_matching_trace(self) -> None:
        trace = [{"command": "ss -tlnp | grep 8080", "exit_code": 0, "stdout_excerpt": "LISTEN 0.0.0.0:8080 ollama"}]
        d = GUARD.verify_turn(ALETHEIC_LIE, tool_results=trace, aletheic=True, aletheic_trace=trace)
        assert d.approved is True
        assert d.aletheic_matched == 1
        assert d.aletheic_unverified == 0

    def test_state_lie_blocked_with_nonmatching_trace(self) -> None:
        trace = [{"command": "echo done", "exit_code": 0, "stdout_excerpt": "done"}]
        d = GUARD.verify_turn(ALETHEIC_LIE, tool_results=trace, aletheic=True, aletheic_trace=trace)
        assert d.approved is False
        assert d.aletheic_unverified == 1

    def test_negated_state_claim_strict_by_default(self) -> None:
        d = GUARD.verify_turn("El puerto 8000 no está reservado para la API.", aletheic=True)
        assert d.approved is False
        assert d.aletheic_unverified == 1