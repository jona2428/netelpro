"""Pruebas unitarias para el motor Netelpro Discrete (Zero-Float)."""

import time

from netelpro.discrete.bit_vector import BitVector
from netelpro.discrete.binary_memory import BinaryMemoryBank
from netelpro.discrete.logic_synthesizer import DiscreteLogicSynthesizer
from netelpro.discrete.articulator import DiscreteArticulator


def test_bit_vector_algebra():
    """Verifica álgebra de bits pura y distancia de Hamming."""
    v1 = BitVector.random(num_bits=512, seed=42)
    v2 = BitVector.random(num_bits=512, seed=99)

    # Identidad XOR
    v_zero = v1 ^ v1
    assert v_zero.hamming_distance(BitVector(num_bits=512)) == 0

    # Distancia a sí mismo es 0 (similitud 100%)
    assert v1.hamming_distance(v1) == 0
    assert v1.similarity_percent(v1) == 100

    # Distancia ortogonal pseudo-aleatoria debe rondar el 50% (~256 bits)
    dist = v1.hamming_distance(v2)
    assert 200 <= dist <= 312

    # Bundle de mayoría
    bundled = BitVector.bundle([v1, v1, v2])
    assert bundled.hamming_distance(v1) < 40


def test_bit_vector_from_text():
    """Verifica que el SimHash entero sea determinista y capture similitud semántica."""
    t1 = "el flujo laminar en la tuberia hidraulica"
    t2 = "el flujo laminar en la tuberia"
    t3 = "un astronauta cocinando empanadas en jupiter"

    v1 = BitVector.from_text(t1, num_bits=512)
    v2 = BitVector.from_text(t2, num_bits=512)
    v3 = BitVector.from_text(t3, num_bits=512)

    sim_related = v1.similarity_percent(v2)
    sim_unrelated = v1.similarity_percent(v3)

    assert sim_related > sim_unrelated
    assert sim_related >= 60


def test_binary_memory_bank():
    """Verifica almacenamiento asociativo y recuperación rápida (Binary RAG)."""
    bank = BinaryMemoryBank(num_bits=512)

    bank.store(
        concept="cavitacion y backpressure",
        content="La cavitacion en bombas se previene manteniendo el inlet por encima de la presion de vapor.",
        metadata={"dominio": "fluidos"},
    )
    bank.store(
        concept="apoptosis celular",
        content="La apoptosis es la muerte celular programada para preservar la estabilidad del tejido.",
        metadata={"dominio": "biologia"},
    )

    # Consulta semántica cercana
    matches = bank.recall("como prevenir la cavitacion en bombas", top_k=1)
    assert len(matches) == 1
    assert matches[0].record.concept == "cavitacion y backpressure"
    assert "inlet" in matches[0].record.content

    # Latencia pura de búsqueda Hamming: 500 búsquedas en menos de 50 milisegundos en CPU
    q_vec = BitVector.from_text("cavitacion bombas", num_bits=512)
    t0 = time.perf_counter()
    for _ in range(500):
        bank.recall(q_vec, top_k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"Demasiado lento: {elapsed_ms:.1f}ms"


def test_logic_synthesizer_and_llvm_compilation():
    """Verifica la síntesis de reglas desde observaciones y su compilación LLVM nativa."""
    synthesizer = DiscreteLogicSynthesizer(rule_name="filter-rule")

    params = ["inlet-ok", "outlet-ok", "valve-open"]
    # Regla: se activa si inlet-ok=1 y outlet-ok=1
    training_cases = [
        ((0, 0, 0), 0),
        ((0, 0, 1), 0),
        ((0, 1, 0), 0),
        ((0, 1, 1), 0),
        ((1, 0, 0), 0),
        ((1, 0, 1), 0),
        ((1, 1, 0), 1),
        ((1, 1, 1), 1),
    ]

    # Aprender y compilar a LLVM nativo en microsegundos
    t0 = time.perf_counter()
    rule_filter, contract_code = synthesizer.learn_and_compile(
        param_names=params,
        training_cases=training_cases,
        default_verdict=0,
    )
    synthesis_ms = (time.perf_counter() - t0) * 1000

    assert synthesis_ms < 500, f"Síntesis y compilación tardó {synthesis_ms:.2f}ms"
    assert "```netelpro" in contract_code
    assert "(truth-table filter-rule" in contract_code

    # Ejecutar veredicto nativo LLVM en CPU
    res_pos = rule_filter.decide(1, 1, 1)
    assert res_pos is True

    res_neg = rule_filter.decide(1, 0, 1)
    assert res_neg is False

    res_wild = rule_filter.decide(0, 1, 0)
    assert res_wild is False


def test_articulator_natural_output():
    """Verifica que el articulador genere lenguaje fluido sin sonar robótico."""
    articulator = DiscreteArticulator()

    signals = {"inlet-ok": 1, "outlet-ok": 0, "valve-open": 1}
    response = articulator.articulate_decision(
        domain_name="Circuito de Enfriamiento",
        signals=signals,
        verdict=0,
        rule_name="cooling-safety-gate",
        law_description="Conservacion de presion minima",
    )

    assert "<thought>" in response
    assert "</thought>" in response
    assert "bloqueada de forma preventiva" in response
    assert "fail-closed" in response
