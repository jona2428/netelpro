"""Capa de Articulación Natural de Netelpro (Anti-Robotic Natural Language Articulator).

Resuelve el dilema fundamental: ¿cómo evitar que un sistema puramente lógico
y determinista suene como un autómata rígido de los años 80?

Arquitectura:
- Núcleo (Netelpro / Binario): Provee la VERDAD matemática indiscutible (0 alucinaciones).
- Articulador (Área de Broca): Traduce el veredicto y la memoria a lenguaje natural
  humano, cálido, contextual y claro.
"""

from __future__ import annotations



class DiscreteArticulator:
    """Articula decisiones lógicas y recuerdos binarios en lenguaje natural fluido."""

    def __init__(self, tone: str = "amigable_experto") -> None:
        self.tone: str = tone

    def articulate_decision(
        self,
        domain_name: str,
        signals: dict[str, int],
        verdict: int,
        rule_name: str,
        law_description: str | None = None,
    ) -> str:
        """Formula una explicación humana y natural sobre la decisión tomada."""
        active = [k for k, v in signals.items() if v == 1]
        inactive = [k for k, v in signals.items() if v == 0]

        verdict_str = "AUTORIZADO / ACTIVO" if verdict == 1 else "RECHAZADO / BLOQUEADO (Fail-Closed)"

        thought = (
            f"<thought>\n"
            f"Análisis de estado en dominio '{domain_name}':\n"
            f"- Señales en ALTO (1): {', '.join(active) if active else 'ninguna'}\n"
            f"- Señales en BAJO (0): {', '.join(inactive) if inactive else 'ninguna'}\n"
            f"- Ley gobernante: {law_description or 'Contrato determinista Netelpro'}\n"
            f"- Veredicto nativo LLVM: {verdict_str}\n"
            f"</thought>"
        )

        # Construcción de respuesta conversacional natural (sin sonar a máquina)
        if verdict == 1:
            natural = (
                f"Todo en orden con **{domain_name}**. Tras evaluar las condiciones del sistema, "
                f"veo que las variables críticas ({', '.join(active)}) cumplen los requisitos de seguridad. "
                f"Por ende, la operación fue **aprobada y autorizada** de acuerdo a la regla `{rule_name}`."
            )
        else:
            reasons = []
            if inactive:
                reasons.append(f"las señales {', '.join(inactive)} no están en el estado seguro requerido")
            natural = (
                f"Por precaución, la acción en **{domain_name}** fue **bloqueada de forma preventiva**. "
                f"El motor de Netelpro operó bajo el principio *fail-closed*: {'debido a que ' + reasons[0] if reasons else 'la combinación de estados no está permitida'}. "
                f"El sistema permanece en estado seguro hasta que se restablezcan las condiciones."
            )

        return f"{thought}\n\n{natural}"

    def articulate_memory_recall(
        self,
        query: str,
        concept: str,
        content: str,
        similarity_percent: int,
        contract: str | None = None,
    ) -> str:
        """Articula una recuperación asociativa de memoria sin alucinaciones."""
        thought = (
            f"<thought>\n"
            f"Consulta: '{query}'\n"
            f"Memoria binaria recuperada: '{concept}' (Similitud de Hamming: {similarity_percent}%)\n"
            f"Estado: Recuperación directa en CPU sin tensores ni flotantes.\n"
            f"</thought>"
        )

        response = (
            f"Respecto a tu consulta sobre *{query}*, tengo registrado con certeza matemática:\n\n"
            f"{content}\n"
        )

        if contract:
            response += f"\n**Contrato formal asociado:**\n{contract}\n"

        return f"{thought}\n\n{response}"
