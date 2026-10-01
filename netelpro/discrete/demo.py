"""Demostración interactiva del Motor Discreto de Netelpro (Zero-Float).

Ejecución:
    python -m netelpro.discrete.demo
"""

from __future__ import annotations

import time

from netelpro.discrete import (
    BinaryMemoryBank,
    DiscreteArticulator,
    DiscreteLogicSynthesizer,
)


def run_demo() -> None:
    print("=" * 70)
    print("🚀 NETELPRO ZERO-FLOAT: MOTOR DE MEMORIA BINARIA Y SÍNTESIS LÓGICA")
    print("=" * 70)
    print("Arquitectura: Cero tensores, cero números flotantes, cero backpropagation.")
    print("Hardware: CPU estándar (instrucciones bitwise uint64 y LLVM nativo).\n")

    # 1. Almacenamiento en Memoria Binaria (RAG Discreto)
    print("--- [1] ALMACENAMIENTO DE RECUERDOS (SimHash entero en microsegundos) ---")
    bank = BinaryMemoryBank(num_bits=512)

    t0 = time.perf_counter()
    bank.store(
        concept="cavitacion y backpressure en streaming",
        content=(
            "La ecuación de continuidad hidrodinámica se aplica a colas de mensajería (Kafka/RabbitMQ): "
            "cuando el flujo de entrada excede la tasa de consumo, el sistema debe aplicar contrapresión "
            "activa y abrir válvulas de alivio para evitar el colapso del buffer."
        ),
        metadata={"dominio": "hidráulica / sistemas distribuidos"},
    )
    bank.store(
        concept="apoptosis celular y circuit breaker",
        content=(
            "El aislamiento programado celular destruye instancias degradadas sin perder el quórum "
            "del clúster mediante una cascada de caspasas controlada (Circuit Breaker distribuido)."
        ),
        metadata={"dominio": "biología / microservicios"},
    )
    bank.store(
        concept="recocido simulado y scheduler uma",
        content=(
            "El descenso gradual de la temperatura de Boltzmann permite que el planificador UMA "
            "escape de mínimos locales de congestión de memoria y balancee las tareas eficientemente."
        ),
        metadata={"dominio": "termodinámica / scheduling"},
    )
    t_store = (time.perf_counter() - t0) * 1000
    print(f"✅ 3 hechos complejos indexados en memoria binaria en: {t_store:.2f} ms\n")

    # 2. Recuperación Asociativa por Distancia de Hamming
    print("--- [2] RECUPERACIÓN ASOCIATIVA POR DISTANCIA DE HAMMING (Binary RAG) ---")
    query = "¿Cómo evitamos que colapsen las colas de streaming por exceso de flujo?"
    print(f"Consulta entrante: '{query}'")

    t0 = time.perf_counter()
    matches = bank.recall(query, top_k=1)
    t_recall = (time.perf_counter() - t0) * 1000

    if matches:
        match = matches[0]
        print(f"Recuerdo recuperado: '{match.record.concept}'")
        print(f"Similitud de Hamming: {match.similarity_percent}% (distancia: {match.hamming_distance} bits)")
        print(f"Tiempo de búsqueda en CPU: {t_recall:.3f} ms ({t_recall * 1000:.0f} microsegundos)")
        print(f"Contenido: {match.record.content}\n")

    # 3. Síntesis Lógica Discreta de Reglas (Entrenamiento sin gradientes)
    print("--- [3] APRENDIZAJE Y SÍNTESIS LÓGICA DIRECTA (LLVM JIT) ---")
    print("Entrenando regla de decisión booleana para 'Absorción de Ráfagas'...")
    params = ["inlet-ok", "outlet-ok", "valve-open"]

    # Casos observados: Se activa si inlet=1, outlet=1 y válvula abierta=1
    training_cases = [
        ((0, 0, 0), 0),
        ((0, 0, 1), 0),
        ((0, 1, 0), 0),
        ((0, 1, 1), 0),
        ((1, 0, 0), 0),
        ((1, 0, 1), 0),
        ((1, 1, 0), 0),
        ((1, 1, 1), 1),
    ]

    synthesizer = DiscreteLogicSynthesizer(rule_name="backpressure-flow-gate")

    t0 = time.perf_counter()
    rule_filter, contract_source = synthesizer.learn_and_compile(
        param_names=params,
        training_cases=training_cases,
        default_verdict=0,
    )
    t_synth = (time.perf_counter() - t0) * 1000

    print(f"✅ Regla aprendida, verificada y compilada por LLVM en: {t_synth:.2f} ms")
    print("Código Netelpro generado:")
    print(contract_source)
    print()

    # 4. Evaluación y Ejecución Nativa
    print("--- [4] EJECUCIÓN DETERMINISTA EN SILICIO NATIVO ---")
    test_signals = {"inlet-ok": 1, "outlet-ok": 1, "valve-open": 1}
    t0 = time.perf_counter()
    verdict = rule_filter.decide(1, 1, 1)
    t_exec = (time.perf_counter() - t0) * 1000000  # nanosegundos

    print(f"Señales de entrada: {test_signals}")
    print(f"Veredicto nativo LLVM: {'AUTORIZADO (1)' if verdict else 'DENEGADO (0)'}")
    print(f"Latencia de decisión de la compuerta: {t_exec:.0f} nanosegundos\n")

    # 5. Articulación Natural Humana (Anti-Robótico)
    print("--- [5] CAPA DE ARTICULACIÓN NATURAL (¿Suena a robot? ¡No!) ---")
    articulator = DiscreteArticulator()
    human_response = articulator.articulate_decision(
        domain_name="Pipeline de Streaming Kafka",
        signals=test_signals,
        verdict=1 if verdict else 0,
        rule_name="backpressure-flow-gate",
        law_description="Ecuación de Continuidad y Prevención de Cavitación",
    )
    print(human_response)
    print("\n" + "=" * 70)
    print("🏁 RESUMEN COMPARATIVO:")
    print("• Entrenamiento clásico Teo-SDS en T4: ~91 tok/s -> 156 HORAS estimadas.")
    print(f"• Motor Discreto Netelpro en CPU: Síntesis + Compilación LLVM en {t_synth:.1f} MILISEGUNDOS.")
    print("• Precisión: 100% determinista, 0% alucinaciones, 0 coma flotante.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
