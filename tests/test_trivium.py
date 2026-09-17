"""Unit tests for the Netelpro Trivium Engine and Fallacy Gate."""

from __future__ import annotations

from pathlib import Path

from netelpro.gate import Gate
from netelpro.neuro.trivium import NetelproTriviumEngine
from training.data.trivium_corpus import create_trivium_corpus


def test_fallacy_gate_formal_verdicts():
    rule_path = Path(__file__).parent.parent / "netelpro" / "neuro" / "rules" / "fallacy_detector.sl"
    gate = Gate(rule_path)

    # Valid argument (no ad hominem, no dichotomy, no straw man, grounded)
    allow, reason = gate.check(0, 0, 0, 1)
    assert allow is True
    assert reason is None

    # Ad hominem violation
    allow, _ = gate.check(1, 0, 0, 1)
    assert allow is False

    # False dichotomy violation
    allow, _ = gate.check(0, 1, 0, 1)
    assert allow is False

    # Straw man violation
    allow, _ = gate.check(0, 0, 1, 1)
    assert allow is False

    # Ungrounded premise violation
    allow, _ = gate.check(0, 0, 0, 0)
    assert allow is False


def test_trivium_engine_ad_hominem():
    engine = NetelproTriviumEngine()
    record = engine.analyze_argument("No le crean a Juan porque es un ignorante.")
    assert record.is_valid is False
    assert "ad_hominem" in record.fallacies_detected
    assert record.ethos_score < 0.5
    assert record.latency_us > 0


def test_trivium_engine_false_dichotomy():
    engine = NetelproTriviumEngine()
    record = engine.analyze_argument("O estás conmigo o estás en contra de todo.")
    assert record.is_valid is False
    assert "false_dichotomy" in record.fallacies_detected


def test_trivium_engine_valid_grounded_argument():
    engine = NetelproTriviumEngine()
    text = "El proyecto es viable porque la evidencia empírica respalda cada estimación de costos."
    record = engine.analyze_argument(text)
    assert record.is_valid is True
    assert len(record.fallacies_detected) == 0
    assert record.logos_score >= 0.85
    assert "VÁLIDO" in record.formal_proof


def test_trivium_corpus_generation():
    corpus = create_trivium_corpus()
    assert len(corpus) > 20
    for sample in corpus:
        assert "<|user|>" in sample["prompt"]
        assert "<|assistant|>" in sample["prompt"]
        assert "<|thought|>" in sample["prompt"]
        assert "<|endthought|>" in sample["prompt"]


def test_dialectical_prompt_format():
    engine = NetelproTriviumEngine()
    prompt = engine.format_dialectical_prompt("Inteligencia Artificial", "La IA reemplazará a todos los pensadores")
    assert "Inteligencia Artificial" in prompt
    assert "<|thought|>" in prompt
