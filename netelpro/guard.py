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
    re.compile(r"\b(he\s+revisado|he\s+verificado|he\s+comprobado|he\s+inspeccionado|he\s+ejecutado|he\s+analizado|revisé|verifiqué|comprobé|inspeccioné|confirmé|confirmo|confirmado|ejecuté|corrí|pasé|analicé|medí|validé|audité|escaneé|testeé)\b", re.IGNORECASE),
    re.compile(r"\b(i\s+(have\s+)?(verified|checked|inspected|confirmed|tested|analyzed|executed|ran|validated|audited|scanned))\b", re.IGNORECASE),
    re.compile(r"\b((el|la)\s+)?(escaneo\s+confirmó|análisis\s+confirmó)\b", re.IGNORECASE),
    re.compile(r"\b(tests\s+pasaron|compiló\s+con\s+0|cero\s+errores|todo\s+está\s+operativo)\b", re.IGNORECASE),
]

# Negaciones verbales y léxicas (ES: no|nunca|jamás|tampoco|ninguna; EN:
# not|never + cualquier contracción n't). Nota: \bn't era código muerto — no
# existe límite de palabra entre consonante y apóstrofe; \w+n't sí matchea.
_NEGATION_PATTERN = re.compile(
    r"\b(no|nunca|jamás|tampoco|ninguna|not|never)\b|\w+n't\b",
    re.IGNORECASE,
)

# Frontera de cláusula/contraste: si entre la negación y el verbo hay
# puntuación de frontera o un conector de contraste, el alcance de la negación
# termina ahí y no alcanza al verbo ("No, verifiqué X", "no revisé logs,
# pero verifiqué el puerto"). Cura los falsos negativos por 'no' discursivo.
_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"[,;:.!?¿¡]|\b(pero|sino|aunque|but|however|although)\b",
    re.IGNORECASE,
)

# Un match dentro de un span interrogativo no es un claim: la negación ahí es
# pragmática ("¿Crees que ejecuté algo?"), no léxica.
_QUESTION_SPAN_PATTERN = re.compile(r"¿[^?]*\?")

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
    aletheic_unverified: int = 0
    aletheic_matched: int = 0


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
        """Detecta si el texto del turno del agenta afirma haber realizado una verificación.

        Ronda 2 del detector de negación:
        - Ventana de 40 chars con alcance sintáctico: una negación no alcanza
          al verbo si entre ambos hay frontera de cláusula o conector de
          contraste ("No, verifiqué X" SÍ es claim).
        - Un match dentro de un span interrogativo no es claim
          ("¿Crees que ejecuté algo?").
        - Léxico ES ampliado (jamás/tampoco/ninguna) y contracciones EN
          completas vía \\w+n't (don't, can't, didn't, ...).
        """
        for pattern in _VERIFICATION_ASSERTION_PATTERNS:
            for m in pattern.finditer(text):
                if self._negation_scopes_over(text, m.start()):
                    continue  # negado: no es un claim
                if any(
                    q.start() <= m.start() < q.end()
                    for q in _QUESTION_SPAN_PATTERN.finditer(text)
                ):
                    continue  # pregunta retórica: no es un claim
                return True
        return False

    def _negation_scopes_over(self, text: str, verb_start: int) -> bool:
        """True si una negación en la ventana previa (40 chars) alcanza al verbo.

        La negación no cuenta cuando entre ella y el verbo hay frontera de
        cláusula o conector de contraste: su alcance termina ahí.
        """
        prefix = text[max(0, verb_start - 40) : verb_start]
        for neg in _NEGATION_PATTERN.finditer(prefix):
            if _CLAUSE_BOUNDARY_PATTERN.search(prefix[neg.end() :]):
                continue  # negación de otra cláusula o discursiva
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
        aletheic: bool = False,
        aletheic_trace: list[dict] | None = None,
    ) -> GuardDecision:
        """Audita el turno del agente a través del binario nativo LLVM.

        Args:
            agent_text: Texto emitido por el modelo.
            tool_results: Lista de resultados de herramientas devueltos por el sistema.
            override_claimed: Opcional, forzar flag claimed.
            override_sources: Opcional, forzar conteo de fuentes.
            aletheic: Activa la capa de teatro alético (aseveraciones de estado
                del mundo sin evidencia: "Ollama está escuchando en el 8080").
                Opt-in: con False el comportamiento es idéntico al histórico
                (solo claims procedurales).
            aletheic_trace: Traces de ejecución para verificar claims aléticos
                (lista de {'command', 'exit_code', 'stdout_excerpt'}). Requerido
                si aletheic=True y el texto contiene state claims.

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

        # Capa alética (opt-in): aseveraciones de estado del mundo requieren
        # trace de herramienta que las respalde (ver netelpro/aletheic.py).
        aletheic_unverified = 0
        aletheic_matched = 0
        if aletheic:
            from netelpro.aletheic import detect_state_claims, verify_aletheic

            state_claims = detect_state_claims(agent_text)
            verdict = verify_aletheic(state_claims, aletheic_trace)
            aletheic_unverified = len(verdict.unverified)
            aletheic_matched = len(verdict.matched)
            if not verdict.allowed:
                passed = False

        reason = None
        if not passed and aletheic and aletheic_unverified > 0:
            reason = (
                f"Acción denegada por teatro alético: {aletheic_unverified} aseveración(es) de "
                f"estado del mundo sin trace de verificación "
                f"(aletheic_matched={aletheic_matched}, claimed={claimed}, verified={has_verified_tool})."
            )
        elif not passed:
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
            aletheic_unverified=aletheic_unverified,
            aletheic_matched=aletheic_matched,
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
