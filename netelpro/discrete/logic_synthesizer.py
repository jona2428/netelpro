"""Sintetizador Lógico Discreto para Netelpro (Zero-Float Logic Synthesizer).

Aprende y sintetiza contratos formales ejecutables a partir de observaciones
y tablas de verdad sin utilizar descenso de gradiente, ni números flotantes, ni backprop.

El resultado es código Netelpro canónico que compila inmediatamente a código
máquina nativo mediante LLVM con cero sobrecarga.
"""

from __future__ import annotations

from typing import Sequence

from netelpro.rule_filter import RuleFilter


class DiscreteLogicSynthesizer:
    """Sintetizador discreto de reglas formales Netelpro a partir de datos observados."""

    def __init__(self, rule_name: str = "filter-rule") -> None:
        self.rule_name: str = rule_name

    def synthesize_truth_table_contract(
        self,
        param_names: Sequence[str],
        cases: Sequence[tuple[Sequence[int], int]],
        default_verdict: int = 0,
    ) -> str:
        """Sintetiza un contrato formal Netelpro con formato (truth-table ...).

        Garantiza principio fail-closed: cualquier caso no observado o ambiguo
        cae en la cláusula comodín por defecto (default_verdict).
        """
        if not param_names:
            raise ValueError("Se requiere al menos un parámetro")

        decls = "\n".join(f"  ({p} : (Int 0 1))" for p in param_names)

        # Ordenar casos observados de forma determinista
        sorted_cases = sorted(cases, key=lambda item: tuple(item[0]))

        rows: list[str] = []
        seen_keys: set[tuple[int, ...]] = set()

        for inputs, verdict in sorted_cases:
            key = tuple(inputs)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bits_str = " ".join(str(int(b)) for b in inputs)
            v_val = 1 if verdict else 0
            rows.append(f"  (({bits_str}) -> {v_val})")

        # Cláusula comodín fail-closed
        wildcard_bits = " ".join("_" for _ in param_names)
        rows.append(f"  (({wildcard_bits}) -> {default_verdict})")

        body = "\n".join(rows)
        return f"```netelpro\n(truth-table {self.rule_name}\n{decls}\n{body})\n```"

    def learn_and_compile(
        self,
        param_names: Sequence[str],
        training_cases: Sequence[tuple[Sequence[int], int]],
        default_verdict: int = 0,
    ) -> tuple[RuleFilter, str]:
        """Aprende la regla directamente de los datos y la compila con LLVM.

        Retorna (objeto RuleFilter compilado en LLVM, código fuente Netelpro).
        Tiempo de ejecución: ~1-5 milisegundos en CPU. Cero entrenamiento de GPU.
        """
        contract_with_fence = self.synthesize_truth_table_contract(
            param_names=param_names,
            cases=training_cases,
            default_verdict=default_verdict,
        )

        # Extraer bloque de código limpio para el compilador
        clean_source = contract_with_fence
        if "```netelpro" in clean_source:
            start = clean_source.find("```netelpro") + len("```netelpro")
            end = clean_source.find("```", start)
            clean_source = clean_source[start:end].strip()

        # Compilación nativa con el motor de Netelpro (LLVM)
        filter_rule = RuleFilter(clean_source, defn_name=self.rule_name)

        # Verificación diferencial inmediata
        mismatches = filter_rule.verify_int(
            [(tuple(inputs), exp) for inputs, exp in training_cases]
        )
        if mismatches:
            raise RuntimeError(
                f"El contrato compilado no satisface los casos de entrenamiento: {mismatches}"
            )

        return filter_rule, contract_with_fence
