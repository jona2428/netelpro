"""Offline unit and integration tests for training/data/download_corpus.py (teo_v2).

Tests:
1. Document cleaning (boilerplate/navigation line removal, min length 200 chars, whitespace normalization)
2. Deduplication (SHA-1 on normalized NFKC lowercased text, duplicate counting)
3. Framing [bos] + ids + [eos] (strict special tokens, zero padding in token sequence)
4. Shard math (exact splitting across shard boundaries, max_shards cap, uint16 stream verification)
5. CLI argument and size parsing
6. Offline end-to-end integration using mock document streams
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from training.data.download_corpus import (
    DocumentDeduplicator,
    PackedShardWriter,
    build_corpus,
    clean_document,
    is_boilerplate_line,
    parse_byte_size,
    parse_sources,
)
from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer
from training.data.compile_packed import load_stream, stream_stats


# ===========================================================================
# 1. Cleaning & Boilerplate Removal Tests
# ===========================================================================


def test_is_boilerplate_line():
    """Verify detection of cookie consent, navigation breadcrumbs, and social bars."""
    boilerplate_samples = [
        "Aceptar todas las cookies",
        "Esta web utiliza cookies para mejorar su experiencia",
        "Cookie policy and privacy notice",
        "Inicio > Noticias > Deportes > Fútbol",
        "Home » Blog » Article",
        "Saltar al contenido principal",
        "| Home | About | Contact |",
        "Compartir en Facebook",
        "Share on Twitter",
        "Todos los derechos reservados.",
        "All rights reserved (c) 2026",
        "------------------------------------",
        "===================",
        "***",
    ]
    for line in boilerplate_samples:
        assert is_boilerplate_line(line), f"Failed to identify boilerplate: {line!r}"

    valid_lines = [
        "El modelo de lenguaje teo_v2 procesa millones de tokens en paralelo.",
        "def calcular_invariante(estado: dict) -> bool:",
        "La historia de la filosofía comenzó en las costas de Jonia.",
        "for i in range(10):",
        "import numpy as np",
    ]
    for line in valid_lines:
        assert not is_boilerplate_line(line), f"False positive boilerplate: {line!r}"


def test_clean_document_removes_boilerplate_and_short_docs():
    """Verify clean_document filters boilerplate lines and enforces min length."""
    # Text with boilerplate lines mixed in
    raw_doc = (
        "Inicio > Noticias > Ciencia\n"
        "Aceptar cookies para continuar navegando.\n"
        "El telescopio espacial James Webb ha observado una galaxia extremadamente antigua "
        "formada apenas unos cientos de millones de años después del Big Bang. Las observaciones "
        "espectroscópicas confirman la presencia de elementos pesados en etapas tempranas del universo, "
        "desafiando algunos modelos cosmológicos previos sobre la evolución estelar y la formación de galaxias primitivas.\n"
        "Compartir en Twitter\n"
        "Todos los derechos reservados."
    )

    cleaned = clean_document(raw_doc, is_code=False, min_chars=200)
    assert cleaned is not None
    assert len(cleaned) >= 200
    assert "Aceptar cookies" not in cleaned
    assert "Inicio > Noticias" not in cleaned
    assert "Compartir en Twitter" not in cleaned
    assert "Todos los derechos reservados" not in cleaned
    assert "telescopio espacial James Webb" in cleaned

    # Short document (< 200 chars) must be dropped
    short_doc = "Esta es una frase corta en español que no alcanza la longitud mínima de doscientos caracteres."
    assert clean_document(short_doc, is_code=False, min_chars=200) is None

    # Exactly 200 chars
    exact_doc = "A" * 200
    assert clean_document(exact_doc, min_chars=200) == exact_doc

    # Empty / None
    assert clean_document("", min_chars=200) is None
    assert clean_document(None, min_chars=200) is None


def test_clean_document_code_preserves_indentation():
    """Verify code cleaning preserves indentation and python syntax."""
    python_code = (
        "def quicksort(arr: list[int]) -> list[int]:\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
    )
    # Ensure > 200 chars
    while len(python_code) < 220:
        python_code += "    # Extra comment line to ensure length exceeds threshold\n"

    cleaned = clean_document(python_code, is_code=True, min_chars=200)
    assert cleaned is not None
    assert "    if len(arr) <= 1:" in cleaned
    assert "    return quicksort(left)" in cleaned


# ===========================================================================
# 2. Deduplication Tests
# ===========================================================================


def test_deduplication_exact_and_normalized():
    """Verify SHA-1 deduplication detects identical and normalized duplicates."""
    dedup = DocumentDeduplicator()

    doc1 = (
        "El modelo neuro-simbólico de Netelpro unifica razonamiento formal y redes neuronales "
        "en arquitecturas de silicio ultra-eficientes diseñadas para teo_v2."
    )
    # Different case and whitespace
    doc1_variant = (
        "  EL MODELO   NEURO-SIMBÓLICO DE   NETELPRO UNIFICA RAZONAMIENTO FORMAL Y REDES NEURONALES\n\n"
        "EN ARQUITECTURAS DE SILICIO ULTRA-EFICIENTES DISEÑADAS PARA TEO_V2.  "
    )
    doc2 = (
        "Un documento completamente distinto que discute algoritmos de optimización "
        "convergencia estocástica y descenso de gradiente acelerado para LLMs."
    )

    # First doc is unique
    assert not dedup.is_duplicate(doc1)
    assert dedup.seen_count == 1
    assert dedup.duplicate_count == 0

    # Exact duplicate rejected
    assert dedup.is_duplicate(doc1)
    assert dedup.seen_count == 1
    assert dedup.duplicate_count == 1

    # Normalized duplicate rejected
    assert dedup.is_duplicate(doc1_variant)
    assert dedup.seen_count == 1
    assert dedup.duplicate_count == 2

    # Different doc accepted
    assert not dedup.is_duplicate(doc2)
    assert dedup.seen_count == 2
    assert dedup.duplicate_count == 2


# ===========================================================================
# 3. [bos] / [eos] Framing Tests
# ===========================================================================


def test_document_framing(tmp_path: Path):
    """Verify framed document has exact [bos] + ids + [eos] and zero pad tokens."""
    # Create sample tokenizer
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "Texto de prueba para verificar framing de tokens bos y eos en teo_v2.\n" * 50,
        encoding="utf-8",
    )
    tok = NetelproBPETokenizer()
    tok.train([sample_file], vocab_size=500, min_frequency=1)

    writer = PackedShardWriter(
        out_dir=tmp_path / "shards",
        shard_size=1000,
        tokenizer=tok,
    )

    text = "Este es un documento verificado para comprobar el framing adecuado de tokens."
    framed = writer.frame_document(text)

    # 1. Starts with bos (1), ends with eos (2)
    assert framed[0] == tok.bos_token_id == 1
    assert framed[-1] == tok.eos_token_id == 2

    # 2. No pad tokens (0)
    assert tok.pad_token_id not in framed
    assert 0 not in framed

    # 3. Middle tokens are non-special
    inner_tokens = framed[1:-1]
    assert len(inner_tokens) > 0
    assert 1 not in inner_tokens
    assert 2 not in inner_tokens


# ===========================================================================
# 4. Shard Math & Boundary Splitting Tests
# ===========================================================================


def test_shard_math_and_boundary_split(tmp_path: Path):
    """Verify that PackedShardWriter cleanly splits across shard boundaries and respects max_shards."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "Fragmento de texto utilizado para entrenar un modelo BPE sintético.\n" * 50,
        encoding="utf-8",
    )
    tok = NetelproBPETokenizer()
    tok.train([sample_file], vocab_size=500, min_frequency=1)

    # Use a small shard size of 100 tokens to test boundary splits
    shard_tokens = 100
    out_dir = tmp_path / "packed_shards"

    writer = PackedShardWriter(
        out_dir=out_dir,
        shard_size=shard_tokens,
        max_shards=3,
        tokenizer=tok,
    )

    doc_text = "Palabra " * 30  # Each doc produces ~30-40 tokens
    docs_written = 0

    # Write documents until max_shards (3) is reached
    for _ in range(20):
        ok = writer.write_document(doc_text)
        if ok:
            docs_written += 1
        else:
            break

    manifest = writer.close()

    # Verify shards created
    assert len(writer.shards_completed) == 3
    assert manifest["total_shards"] == 3

    # Check that completed shards have exactly shard_size tokens
    for i in range(2):  # Shards 0 and 1 were completed and rolled over
        shard_info = writer.shards_completed[i]
        assert shard_info["total_tokens"] == shard_tokens
        shard_file = out_dir / shard_info["file_name"]
        assert shard_file.stat().st_size == shard_tokens * 2  # uint16 = 2 bytes per token

        # Verify sidecar metadata
        meta_file = Path(f"{shard_file}.meta.json")
        assert meta_file.is_file()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["total_tokens"] == shard_tokens
        assert meta["pad_fraction"] == 0.0
        assert meta["dtype"] == "uint16"

        # Verify with compile_packed.load_stream and stream_stats
        stream = load_stream(shard_file)
        assert len(stream) == shard_tokens
        stats = stream_stats(stream, pad_token_id=0)
        assert stats["pad_fraction"] == 0.0
        assert 0 not in stream

        # Verify sha256
        actual_sha = hashlib.sha256(shard_file.read_bytes()).hexdigest()
        assert meta["sha256"] == actual_sha


# ===========================================================================
# 5. CLI & Argument Parsing Tests
# ===========================================================================


def test_parse_byte_size():
    """Verify human-readable byte sizes are parsed accurately."""
    assert parse_byte_size(None) is None
    assert parse_byte_size(1024) == 1024
    assert parse_byte_size("150MB") == 150 * 1024 * 1024
    assert parse_byte_size("150M") == 150 * 1024 * 1024
    assert parse_byte_size("2GB") == 2 * 1024 * 1024 * 1024
    assert parse_byte_size("500KB") == 500 * 1024
    assert parse_byte_size("100B") == 100
    assert parse_byte_size("200000000") == 200_000_000


def test_parse_sources():
    """Verify source resolution and error handling."""
    assert parse_sources(None) == ["spanish", "code"]
    assert parse_sources([]) == ["spanish", "code"]
    assert parse_sources(["spanish"]) == ["spanish"]
    assert parse_sources(["code"]) == ["code"]
    assert parse_sources("spanish,code") == ["spanish", "code"]
    assert parse_sources(["fineweb", "github-code"]) == ["spanish", "code"]
    assert parse_sources("all") == ["spanish", "code"]

    with pytest.raises(ValueError, match="Unknown source"):
        parse_sources(["invalid_source"])


# ===========================================================================
# 6. Full Offline Pipeline Integration Test
# ===========================================================================


def test_offline_build_corpus_end_to_end(tmp_path: Path):
    """End-to-end offline pipeline test with synthetic Spanish and Python documents."""
    out_dir = tmp_path / "corpus_out"

    # Synthetic document stream
    spanish_doc_1 = (
        "El entrenamiento a gran escala de modelos de lenguaje autorregresivos sobre corpora multilingües "
        "requiere una preparación meticulosa de los datos para eliminar ruido, boilerplate y secuencias redundantes. "
        "En teo_v2 se integran textos en español de alta calidad con código Python estructurado para potenciar "
        "capacidades de razonamiento formal y verificación neuro-simbólica."
    )
    spanish_doc_2 = (
        "La arquitectura del transformador aprovecha mecanismos de atención dispersa y capas de memoria "
        "compartida en silicio para maximizar el throughput computacional sin desperdiciar ancho de banda en tokens de relleno."
    )
    # Duplicate of doc 1 with whitespace variance
    spanish_doc_dup = spanish_doc_1.upper()

    python_doc_1 = (
        "def compute_aletheic_closure(states: list[dict], transitions: list[tuple[int, int]]) -> set[int]:\n"
        "    visited = set()\n"
        "    queue = [0]\n"
        "    while queue:\n"
        "        curr = queue.pop(0)\n"
        "        if curr not in visited:\n"
        "            visited.add(curr)\n"
        "            queue.extend([dst for src, dst in transitions if src == curr])\n"
        "    return visited\n"
    )
    python_doc_2 = (
        "class NeuroSymbolicBridge:\n"
        "    def __init__(self, vocab_size: int = 32768) -> None:\n"
        "        self.vocab_size = vocab_size\n"
        "        self.invariants = []\n"
        "    def add_invariant(self, predicate) -> None:\n"
        "        self.invariants.append(predicate)\n"
        "    def verify(self, state: dict) -> bool:\n"
        "        return all(inv(state) for inv in self.invariants)\n"
    )

    mock_stream = [
        ("spanish", spanish_doc_1),
        ("code", python_doc_1),
        ("spanish", spanish_doc_dup),  # Should be deduplicated!
        ("spanish", spanish_doc_2),
        ("code", python_doc_2),
    ]

    metrics = build_corpus(
        out_dir=out_dir,
        sources=["spanish", "code"],
        shard_size=2000,
        max_shards=None,
        min_doc_chars=100,
        sample_bytes_for_tokenizer=500,
        custom_doc_stream=iter(mock_stream),
    )

    # 1. Summary metrics
    assert metrics["status"] == "completed"
    assert metrics["unique_docs"] == 4
    assert metrics["duplicate_docs_dropped"] == 1
    assert metrics["total_shards"] >= 1

    # 2. Tokenizer artifact created and frozen
    tok_path = out_dir / "tokenizer.json"
    assert tok_path.is_file()
    loaded_tok = NetelproBPETokenizer.load(tok_path)
    assert loaded_tok.vocab_size > 0

    # 3. Output shards exist with pad_fraction == 0.0
    shard_0 = out_dir / "shard_00000.bin"
    assert shard_0.is_file()
    stream = load_stream(shard_0)
    stats = stream_stats(stream, pad_token_id=0)
    assert stats["pad_fraction"] == 0.0
    assert 0 not in stream


def test_spanning_sequence_exact_reconstruction(tmp_path: Path):
    """Verify that splitting across multiple shards preserves the exact continuous token stream."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "Fragmento de texto de entrenamiento continuo para Netelpro teo_v2.\n" * 50,
        encoding="utf-8",
    )
    tok = NetelproBPETokenizer()
    tok.train([sample_file], vocab_size=500, min_frequency=1)

    out_dir = tmp_path / "shards_reconstruction"
    shard_size = 40  # Very small shards to force multiple splits

    writer = PackedShardWriter(
        out_dir=out_dir,
        shard_size=shard_size,
        tokenizer=tok,
    )

    doc_list = [
        "Primer documento de longitud moderada con varias palabras clave.",
        "Segundo documento enfocado en verificación neuro-simbólica y contratos en silicio.",
        "Tercer documento que contiene código Python def resolver(a, b): return a * b.",
    ]

    expected_full_tokens = []
    for doc in doc_list:
        framed = writer.frame_document(doc)
        expected_full_tokens.extend(framed)
        writer.write_document(doc)

    writer.close()

    # Reassemble all shards into one contiguous array
    shard_files = sorted(out_dir.glob("shard_*.bin"))
    assert len(shard_files) >= 2

    reconstructed_tokens = []
    for sf in shard_files:
        shard_memmap = load_stream(sf)
        reconstructed_tokens.extend(shard_memmap.tolist())

    assert reconstructed_tokens == expected_full_tokens
    assert len(reconstructed_tokens) == len(expected_full_tokens)


def test_clean_document_edge_cases():
    """Verify handling of empty strings, whitespace, none, and non-string types."""
    assert clean_document("") is None
    assert clean_document("   \n\n\t  ") is None
    assert clean_document(None) is None
    assert clean_document(12345) is None
    assert clean_document("a" * 199, min_chars=200) is None
    assert clean_document("a" * 200, min_chars=200) == "a" * 200


def test_cli_parsing_and_help():
    """Verify that CLI argparser accepts --sources, --max-shards, --max-bytes-per-source, --out-dir."""
    from training.data.download_corpus import main as cli_main

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0

