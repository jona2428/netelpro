"""Tests for the packed token stream compiler and loader utilities (teo_v2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from training.data.compile_packed import (
    FallbackTokenizer,
    compile_corpus_packed,
    get_batch,
    load_stream,
    load_tokenizer,
    main as cli_main,
    stream_stats,
)


@pytest.fixture
def active_tokenizer():
    """Returns active tokenizer or FallbackTokenizer if BPE is not available yet."""
    return load_tokenizer()


def test_end_to_end(tmp_path: Path, active_tokenizer):
    """Tmp dir with 3 .txt docs -> compile -> meta.json exists with correct total_tokens, pad_fraction == 0.0."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    doc1 = (
        "El compilador binario emite flujos compactados contiguos uint16 sin relleno "
        "para maximizar la densidad de información y evitar bucles de padding en la inferencia."
    )
    doc2 = (
        "La arquitectura neuro-simbólica de Netelpro integra razonamiento formal, "
        "contratos rigurosos y ejecución en silicio a frecuencias ultra-rápidas."
    )
    doc3 = (
        "Teo v2 escala el entrenamiento autorregresivo sobre ventanas continuas empaquetadas "
        "sin desperdicio computacional, emulando la eficiencia de nanoGPT."
    )

    docs = [doc1, doc2, doc3]
    for i, doc in enumerate(docs, 1):
        assert len(doc) >= 80, f"Doc {i} must have >= 80 chars for default min_doc_chars"
        (corpus_dir / f"doc_{i}.txt").write_text(doc, encoding="utf-8")

    out_bin = tmp_path / "compiled.bin"
    meta = compile_corpus_packed(
        input_path=corpus_dir,
        out_bin=out_bin,
        tokenizer=active_tokenizer,
        min_doc_chars=80,
    )

    # 1. Output files exist
    assert out_bin.exists()
    meta_json_path = Path(f"{out_bin}.meta.json")
    alt_meta_json_path = out_bin.with_suffix(".meta.json")
    assert meta_json_path.exists() or alt_meta_json_path.exists()

    # 2. Meta matches expected content
    with open(meta_json_path if meta_json_path.exists() else alt_meta_json_path, "r", encoding="utf-8") as f:
        meta_loaded = json.load(f)

    assert meta_loaded["total_docs"] == 3
    assert meta_loaded["dtype"] == "uint16"

    # Calculate exact total tokens: [bos] + encode(doc) + [eos] per document
    bos_id = getattr(active_tokenizer, "bos_token_id", 1)
    eos_id = getattr(active_tokenizer, "eos_token_id", 2)
    pad_id = getattr(active_tokenizer, "pad_token_id", 0)

    expected_total_tokens = 0
    for doc in docs:
        try:
            raw = active_tokenizer.encode(doc, add_special_tokens=False)
        except TypeError:
            raw = active_tokenizer.encode(doc)
        if len(raw) > 0 and raw[0] == bos_id:
            raw = raw[1:]
        if len(raw) > 0 and raw[-1] == eos_id:
            raw = raw[:-1]
        if pad_id is not None:
            raw = [t for t in raw if t != pad_id]
        expected_total_tokens += 1 + len(raw) + 1  # bos + encoded + eos

    assert meta_loaded["total_tokens"] == expected_total_tokens
    assert out_bin.stat().st_size == expected_total_tokens * 2

    # Verify sha256
    file_sha256 = hashlib.sha256(out_bin.read_bytes()).hexdigest()
    assert meta_loaded["sha256"] == file_sha256

    # 3. Stream stats: pad_fraction must be exactly 0.0
    stream = load_stream(out_bin)
    assert len(stream) == expected_total_tokens
    stats = stream_stats(stream, pad_token_id=pad_id)
    assert stats["pad_fraction"] == 0.0
    assert stats["total_tokens"] == expected_total_tokens
    assert pad_id not in stream


def test_get_batch():
    """Verify shapes, targets == inputs shifted by 1 on known sequence, and offsets within bounds."""
    # Known sequence: 0, 1, 2, ..., 999
    stream = np.arange(1000, dtype=np.uint16)
    block_size = 32
    batch_size = 8
    rng = np.random.default_rng(12345)

    inputs, targets = get_batch(
        stream=stream,
        block_size=block_size,
        batch_size=batch_size,
        generator=rng,
    )

    # 1. Shapes correct
    assert inputs.shape == (batch_size, block_size)
    assert targets.shape == (batch_size, block_size)

    # 2. Dtypes correct (int64)
    assert inputs.dtype == np.int64
    assert targets.dtype == np.int64

    # 3. Targets == inputs shifted by 1
    # On a consecutive integer sequence stream[i] = i, so targets == inputs + 1
    np.testing.assert_array_equal(targets, inputs + 1)
    # General shift-by-1 property: targets[:, :-1] matches inputs[:, 1:]
    np.testing.assert_array_equal(targets[:, :-1], inputs[:, 1:])

    # 4. Offsets within bounds
    for b in range(batch_size):
        offset = inputs[b, 0]
        assert 0 <= offset <= len(stream) - block_size - 1
        assert offset + block_size + 1 <= len(stream)
        np.testing.assert_array_equal(inputs[b], stream[offset : offset + block_size])
        np.testing.assert_array_equal(targets[b], stream[offset + 1 : offset + block_size + 1])

    # 5. Length verification
    short_stream = np.arange(10, dtype=np.uint16)
    with pytest.raises(ValueError):
        get_batch(short_stream, block_size=10, batch_size=2)


def test_round_trip(tmp_path: Path, active_tokenizer):
    """Decode of the stream's first document segment starts with the expected doc text."""
    doc_text = (
        "Netelpro neuro-symbolic reasoning engine compiles packed token streams "
        "for teo_v2 training without pad tokens to preserve silicon compute bandwidth."
    )
    doc_file = tmp_path / "first_doc.txt"
    doc_file.write_text(doc_text, encoding="utf-8")

    out_bin = tmp_path / "roundtrip.bin"
    compile_corpus_packed(
        input_path=doc_file,
        out_bin=out_bin,
        tokenizer=active_tokenizer,
        min_doc_chars=80,
    )

    stream = load_stream(out_bin)
    bos_id = getattr(active_tokenizer, "bos_token_id", 1)
    eos_id = getattr(active_tokenizer, "eos_token_id", 2)

    assert stream[0] == bos_id
    # Find first EOS token
    eos_indices = np.where(stream == eos_id)[0]
    assert len(eos_indices) >= 1
    first_eos_idx = int(eos_indices[0])

    # Decode first document segment (including or excluding special tokens)
    first_doc_tokens = stream[: first_eos_idx + 1]
    decoded_text = active_tokenizer.decode(first_doc_tokens, skip_special_tokens=True)

    # Decoded text should match original doc
    assert decoded_text.strip().startswith(doc_text.strip()[:40])
    assert decoded_text.strip() == doc_text.strip()


def test_chunking_synthetic_stream(tmp_path: Path, active_tokenizer):
    """Synthetic ~200k-token stream compiles fine in chunks and meta sha256 matches the actual file."""
    # Generate ~200k tokens by repeating distinct sentences across multiple documents
    sentence = (
        "El sistema neuronal de Teo v2 ejecuta grafos computacionales optimizados sobre memoria compartida "
        "con precisión matemática y contratos aletheicos en silicio. "
    )
    # Each sentence is ~150 chars (~30-40 tokens). We generate documents with repeated text.
    corpus_dir = tmp_path / "synthetic_corpus"
    corpus_dir.mkdir()

    num_docs = 100
    repeats_per_doc = 50
    for i in range(num_docs):
        doc_content = f"Documento sintético #{i}: " + (sentence * repeats_per_doc)
        (corpus_dir / f"syn_{i:04d}.txt").write_text(doc_content, encoding="utf-8")

    out_bin = tmp_path / "synthetic.bin"

    # Use chunk_tokens=50_000 so ~200k tokens forces multiple periodic flushes
    meta = compile_corpus_packed(
        input_path=corpus_dir,
        out_bin=out_bin,
        tokenizer=active_tokenizer,
        chunk_tokens=50_000,
        min_doc_chars=80,
    )

    assert out_bin.exists()
    total_tokens = meta["total_tokens"]
    # Check that stream is around ~200k tokens
    assert total_tokens >= 180_000, f"Expected ~200k tokens, got {total_tokens}"

    # Verify file size matches tokens * 2 bytes
    actual_bytes = out_bin.stat().st_size
    assert actual_bytes == total_tokens * 2

    # Verify sha256 matches actual file
    actual_sha256 = hashlib.sha256(out_bin.read_bytes()).hexdigest()
    assert meta["sha256"] == actual_sha256

    # Verify zero padding
    stream = load_stream(out_bin)
    pad_id = getattr(active_tokenizer, "pad_token_id", 0)
    stats = stream_stats(stream, pad_token_id=pad_id)
    assert stats["pad_fraction"] == 0.0
    assert stats["total_tokens"] == total_tokens
    assert len(stats["top10_unique_tokens"]) <= 10


def test_get_batch_reproducibility():
    """Same generator seed produces identical batches; different seeds produce different batches."""
    stream = np.arange(5000, dtype=np.uint16)
    block_size = 64
    batch_size = 16

    seed = 42042
    rng1 = np.random.default_rng(seed)
    x1, y1 = get_batch(stream, block_size=block_size, batch_size=batch_size, generator=rng1)

    rng2 = np.random.default_rng(seed)
    x2, y2 = get_batch(stream, block_size=block_size, batch_size=batch_size, generator=rng2)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)

    # Different seed
    rng3 = np.random.default_rng(99999)
    x3, y3 = get_batch(stream, block_size=block_size, batch_size=batch_size, generator=rng3)

    assert not np.array_equal(x1, x3)


def test_jsonl_input(tmp_path: Path, active_tokenizer):
    """Verifies compilation handles .jsonl with 'text' and 'prompt' fields."""
    jsonl_file = tmp_path / "data.jsonl"
    lines = [
        {"prompt": "Primer documento en formato jsonl con prompt suficientemente largo para superar los ochenta caracteres requeridos."},
        {"text": "Segundo documento en formato jsonl usando la clave text explícitamente y con más de ochenta caracteres en total."},
        {"prompt": "Tercer documento en jsonl para comprobar la robustez y lectura correcta del compilador sin padding alguno."},
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")

    out_bin = tmp_path / "jsonl_stream.bin"
    meta = compile_corpus_packed(
        input_path=jsonl_file,
        out_bin=out_bin,
        tokenizer=active_tokenizer,
        min_doc_chars=80,
    )

    assert meta["total_docs"] == 3
    stream = load_stream(out_bin)
    assert len(stream) == meta["total_tokens"]
    stats = stream_stats(stream)
    assert stats["pad_fraction"] == 0.0


def test_cli_execution(tmp_path: Path):
    """Verifies that CLI entrypoint executes cleanly via arguments."""
    corpus_file = tmp_path / "cli_sample.txt"
    corpus_file.write_text(
        "Prueba de compilación ejecutando la interfaz de línea de comandos del compilador packed para Teo v2.",
        encoding="utf-8",
    )
    out_bin = tmp_path / "cli_out.bin"

    cli_main([
        "--input", str(corpus_file),
        "--out", str(out_bin),
        "--min-doc-chars", "40",
    ])

    assert out_bin.exists()
    meta_file = Path(f"{out_bin}.meta.json")
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["total_docs"] == 1
    assert meta["dtype"] == "uint16"
