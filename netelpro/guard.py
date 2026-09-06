"""Netelpro Honesty Guard - Universal Epistemic Gate for LLM Agents.

Compiles Netelpro rules natively via LLVM and enforces that claims of
verification in agent turns are strictly backed by verifiable tool execution.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from netelpro.rule_filter import RuleFilter

# Patrones semánticos comunes que indican una aserción de verificación empírica
_VERIFICATION_ASSERTION_PATTERNS = [
    re.compile(r"\b(he\s+revisado|revisé|verifiqué|comprobé|inspeccioné|confirmado|ejecuté|corrí|pasé|analicé|medí)\b", re.IGNORECASE),
    re.compile(r"\b(i\s+(have\s+)?(verified|checked|inspected|confirmed|tested|analyzed|executed|ran))\b", re.IGNORECASE),
    re.compile(r"\b(tests\s+pasaron|compiló\s+con\s+0|cero\s+errores|todo\s+está\s+operativo)\b", re.IGNORECASE),
]

# Patrones para contar fuentes o referencias citadas en el texto
_CITATION_PATTERNS = [
    re.compile(r"\[\d+\]"),
    re.compile(r"\((fuente|source|ref):[^)]+\)", re.IGNORECASE),
    re.compile(r"https?://\S+"),
]

# Regla de producción por defecto: La Fiscalía de Reportes
DEFAULT_VERIFICATION_RULE = """
; Netelpro v0.6 -- La Fiscalía de Reportes
(defn filter-rule (claimed verified sources)
  (if verified
      true
      (if (not claimed)
          (== sources 0)
          false)))
"""


@dataclass(frozen=True)
class GuardDecision:
    """Resultado de la evaluación de la regla Netelpro compilada."""
    approved: bool
    claimed: bool
    verified: bool
    sources_count: int
    latency_ns: int
    rule_name: str
    rejection_reason: str | None = None


class HonestyViolationError(Exception):
    """Excepción lanzada cuando un agente intenta cometer Teatro de Verificación."""
    pass


class HonestyGuard:
    """Guardián nativo de honestidad epistémica respaldado por Netelpro LLVM."""

    def __init__(self, rule_source: str = DEFAULT_VERIFICATION_RULE, name: str = "verification_rule") -> None:
        self.name = name
        self.rule_source = rule_source
        self._filter = RuleFilter(rule_source)

    @classmethod
    def from_file(cls, path: str | Path) -> HonestyGuard:
        p = Path(path)
        content = p.read_text("utf-8")
        return cls(rule_source=content, name=p.stem)

    def detect_claims(self, text: str) -> bool:
        """Detecta si el texto del turno del agente afirma haber realizado una verificación."""
        for pattern in _VERIFICATION_ASSERTION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def count_citations(self, text: str) -> int:
        """Cuenta la cantidad de citas o fuentes externas reclamadas en el texto."""
        count = 0
        for pattern in _CITATION_PATTERNS:
            count += len(pattern.findall(text))
        return count

    def verify_turn(
        self,
        agent_text: str,
        tool_results: Sequence[Any] | None = None,
        override_claimed: bool | None = None,
        override_sources: int | None = None,
    ) -> GuardDecision:
        """Audita el turno del agente a través del binario nativo LLVM.

        Args:
            agent_text: Texto emitido por el modelo.
            tool_results: Lista de resultados de herramientas devueltos por el sistema.
            override_claimed: Opcional, forzar flag claimed.
            override_sources: Opcional, forzar conteo de fuentes.

        Returns:
            GuardDecision con el veredicto del compilador nativo y métricas.
        """
        claimed = override_claimed if override_claimed is not None else self.detect_claims(agent_text)
        has_verified_tool = bool(tool_results and len(tool_results) > 0)
        sources_count = override_sources if override_sources is not None else self.count_citations(agent_text)

        t0 = time.perf_counter_ns()
        # Invocación directa a código máquina nativo vía ctypes
        passed = bool(self._filter.decide(claimed, has_verified_tool, sources_count))
        t1 = time.perf_counter_ns()

        reason = None
        if not passed:
            reason = (
                f"Acción denegada por teatro de verificación: el texto afirma verificación "
                f"(claimed={claimed}), pero la máquina no registró retorno de herramientas "
                f"(verified={has_verified_tool}) con {sources_count} fuentes citadas."
            )

        return GuardDecision(
            approved=passed,
            claimed=claimed,
            verified=has_verified_tool,
            sources_count=sources_count,
            latency_ns=t1 - t0,
            rule_name=self.name,
            rejection_reason=reason,
        )

    def enforce(
        self,
        agent_text: str,
        tool_results: Sequence[Any] | None = None,
    ) -> str:
        """Valida y devuelve el texto; lanza HonestyViolationError si es rechazado."""
        decision = self.verify_turn(agent_text, tool_results)
        if not decision.approved:
            raise HonestyViolationError(decision.rejection_reason)
        return agent_text
