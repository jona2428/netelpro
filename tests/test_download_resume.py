"""Tests for crash/interrupt-safe resumable corpus downloading in teo_v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer
from training.data.compile_packed import load_stream, stream_stats
from training.data.download_corpus import (
    DocumentDeduplicator,
    PackedShardWriter,
    build_corpus,
)


@pytest.fixture
def trained_tokenizer(tmp_path_factory) -> NetelproBPETokenizer:
    """Pre-trains a small deterministic NetelproBPETokenizer for resume tests."""
    tok_dir = tmp_path_factory.mktemp("test_tok")
    sample_file = tok_dir / "sample.txt"
    sample_text = (
        "El entrenamiento autoregresivo en Teo v2 optimiza secuencias compactadas uint16.\n"
        "def compute_hash(data: bytes) -> str:\n"
        "    return hashlib.sha256(data).hexdigest()\n"
        "Clase de prueba con suficiente variedad sintáctica para generar un vocabulario mínimo BPE.\n"
    ) * 40
    sample_file.write_text(sample_text, encoding="utf-8")

    tok = NetelproBPETokenizer()
    tok.train([sample_file], vocab_size=300, min_frequency=1)
    return tok


def _generate_doc(topic: str, index: int, repeat: int = 10) -> str:
    """Generates deterministic long document exceeding min_doc_chars."""
    sentence = (
        f"Documento de prueba sobre el tema {topic} con identificador {index:04d} "
        f"para validar la continuidad del empaquetado de shards en teo_v2 sin padding. "
    )
    return sentence * repeat


def test_download_resume_multi_run(tmp_path: Path, trained_tokenizer: NetelproBPETokenizer):
    """Verifies that download_corpus seamlessly resumes from previously completed shards:

    (a) First shard file is completely unchanged (identical SHA-256).
    (b) New shard starts at index 1 and contains framed tokens of the new documents.
    (c) Global meta.json manifest lists both shards and reflects combined totals.
    (d) Duplicate document across runs is skipped and not written twice.
    """
    out_dir = tmp_path / "corpus_resume_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer in out_dir so both runs use the identical frozen artifact
    tok_path = out_dir / "tokenizer.json"
    trained_tokenizer.save(tok_path)

    shard_size = 2000  # Small shard size (2000 tokens = 4000 bytes)

    doc_shared = (
        "Documento compartido entre ejecuciones: Este texto aparece en la primera ejecución y "
        "también en la segunda ejecución para comprobar que el sistema de deduplicación persistente "
        "mediante SHA-1 de texto normalizado evita escribirlo dos veces en los fragmentos binarios."
    ) * 3

    # Stream 1: doc_shared + multiple docs to fill shard 0 (max_shards=1)
    stream_1 = [("spanish", doc_shared)]
    for i in range(15):
        stream_1.append(("spanish", _generate_doc("run1_spanish", i, repeat=6)))
        stream_1.append(("code", f"# Python code snippet {i}\ndef process_batch_{i}():\n    return {i} * 42\n" * 10))

    metrics_run1 = build_corpus(
        out_dir=out_dir,
        shard_size=shard_size,
        max_shards=1,
        min_doc_chars=80,
        custom_doc_stream=iter(stream_1),
        tokenizer_path=tok_path,
    )

    # Verify Run 1 finished exactly 1 shard
    assert metrics_run1["status"] == "completed"
    assert metrics_run1["total_shards"] == 1
    assert metrics_run1["total_tokens"] == shard_size

    shard_0_path = out_dir / "shard_00000.bin"
    assert shard_0_path.is_file()
    assert shard_0_path.stat().st_size == shard_size * 2

    shard_0_sidecar = out_dir / "shard_00000.bin.meta.json"
    assert shard_0_sidecar.is_file()
    with open(shard_0_sidecar, "r", encoding="utf-8") as f:
        meta_shard_0 = json.load(f)
    assert meta_shard_0["total_tokens"] == shard_size
    assert meta_shard_0["shard_index"] == 0

    # Record SHA-256 of shard 0 after run 1
    sha256_run1_shard0 = hashlib.sha256(shard_0_path.read_bytes()).hexdigest()
    assert meta_shard_0["sha256"] == sha256_run1_shard0

    # Verify dedup_state.json exists
    dedup_file = out_dir / "dedup_state.json"
    assert dedup_file.is_file()

    # Stream 2: contains doc_shared (duplicate from run 1) + brand new documents
    stream_2 = [("spanish", doc_shared)]  # Must be dropped by deduplicator
    for j in range(15):
        stream_2.append(("spanish", _generate_doc("run2_spanish_new", j, repeat=6)))
        stream_2.append(("code", f"# Python code snippet run 2 - {j}\ndef execute_{j}():\n    pass\n" * 10))

    # Run 2: resume into the same out_dir, target max_shards=2
    metrics_run2 = build_corpus(
        out_dir=out_dir,
        shard_size=shard_size,
        max_shards=2,
        min_doc_chars=80,
        custom_doc_stream=iter(stream_2),
        tokenizer_path=tok_path,
    )

    assert metrics_run2["status"] == "completed"

    # (a) Assert first shard file unchanged (same SHA-256 as before)
    sha256_run2_shard0 = hashlib.sha256(shard_0_path.read_bytes()).hexdigest()
    assert sha256_run2_shard0 == sha256_run1_shard0

    # (b) Assert new shard starts at index 1 and contains framed tokens of the new docs
    shard_1_path = out_dir / "shard_00001.bin"
    assert shard_1_path.is_file(), "shard_00001.bin must exist after resume"
    assert shard_1_path.stat().st_size == shard_size * 2

    shard_1_sidecar = out_dir / "shard_00001.bin.meta.json"
    assert shard_1_sidecar.is_file()
    with open(shard_1_sidecar, "r", encoding="utf-8") as f:
        meta_shard_1 = json.load(f)
    assert meta_shard_1["shard_index"] == 1
    assert meta_shard_1["total_tokens"] == shard_size

    # Verify shard 1 stream structure: zero pad tokens, proper uint16 format
    stream_1 = load_stream(shard_1_path)
    assert len(stream_1) == shard_size
    stats_1 = stream_stats(stream_1, pad_token_id=0)
    assert stats_1["pad_fraction"] == 0.0

    # (c) Assert global meta.json lists both shards and cumulative metrics
    manifest_path = out_dir / "meta.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        global_manifest = json.load(f)

    assert global_manifest["total_shards"] == 2
    assert global_manifest["total_tokens"] == shard_size * 2
    assert len(global_manifest["shards"]) == 2
    assert global_manifest["shards"][0]["shard_index"] == 0
    assert global_manifest["shards"][0]["sha256"] == sha256_run1_shard0
    assert global_manifest["shards"][1]["shard_index"] == 1
    assert global_manifest["shards"][1]["file_name"] == "shard_00001.bin"

    # (d) Assert duplicate doc across runs is NOT written twice
    assert metrics_run2["duplicate_docs_dropped"] >= 1

    # Verify the shared doc tokens are not written as a new document in shard 1
    shared_tokens = trained_tokenizer.encode(doc_shared, add_special_tokens=False)
    # Check that shard 1 does not contain the exact full framed sequence of doc_shared
    shared_framed = np.array([trained_tokenizer.bos_token_id] + shared_tokens + [trained_tokenizer.eos_token_id], dtype=np.uint16)
    # A simple sub-array check
    shard_1_arr = np.array(stream_1, dtype=np.uint16)
    # The shared framed document cannot occur in shard 1
    framed_bytes = shared_framed.tobytes()
    shard_1_bytes = shard_1_arr.tobytes()
    assert framed_bytes not in shard_1_bytes, "Duplicate document framed tokens found in shard 1!"


def test_resume_handles_partial_and_corrupted_shards(tmp_path: Path, trained_tokenizer: NetelproBPETokenizer):
    """Verifies that partial/interrupted shards are renamed to <name>.partial and recompiled from scratch."""
    out_dir = tmp_path / "corrupt_resume_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_path = out_dir / "tokenizer.json"
    trained_tokenizer.save(tok_path)

    shard_size = 1000

    # 1. Create a valid shard 0
    shard_0_path = out_dir / "shard_00000.bin"
    dummy_data_0 = np.ones(shard_size, dtype="<u2") * 42
    shard_0_path.write_bytes(dummy_data_0.tobytes())

    meta_0 = {
        "shard_index": 0,
        "file_name": "shard_00000.bin",
        "total_tokens": shard_size,
        "total_docs": 5,
        "vocab_size": 300,
        "tokenizer_type": "NetelproBPETokenizer",
        "sha256": hashlib.sha256(dummy_data_0.tobytes()).hexdigest(),
        "created_at": "2026-09-14T00:00:00Z",
        "dtype": "uint16",
        "pad_fraction": 0.0,
    }
    with open(f"{shard_0_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_0, f, indent=2)

    # 2. Create an incomplete/corrupted shard 1 (size mismatch and missing sidecar)
    shard_1_corrupt = out_dir / "shard_00001.bin"
    shard_1_corrupt.write_bytes(b"\x00\x2a" * 150)  # Only 150 tokens instead of 1000

    # Initialize writer with resume=True
    writer = PackedShardWriter(
        out_dir=out_dir,
        shard_size=shard_size,
        tokenizer=trained_tokenizer,
        resume=True,
    )

    # Valid completed count is 1 (only shard 0)
    assert len(writer.shards_completed) == 1
    assert writer.current_shard_idx == 1
    assert writer.total_tokens == shard_size

    # The corrupt file must have been renamed to .partial (not deleted)
    partial_file = out_dir / "shard_00001.bin.partial"
    assert partial_file.is_file()
    assert partial_file.stat().st_size == 300
    assert not shard_1_corrupt.exists()


def test_document_deduplicator_persistence(tmp_path: Path):
    """Verifies that DocumentDeduplicator save_state and load_state round-trip correctly."""
    dedup1 = DocumentDeduplicator()
    doc_a = "Texto representativo número uno para prueba de deduplicador persistente."
    doc_b = "Texto representativo número dos con variaciones para verificar hashing."

    assert not dedup1.is_duplicate(doc_a)
    assert not dedup1.is_duplicate(doc_b)
    assert dedup1.is_duplicate(doc_a)
    assert dedup1.seen_count == 2
    assert dedup1.duplicate_count == 1

    state_path = tmp_path / "dedup_state.json"
    dedup1.save_state(state_path)
    assert state_path.is_file()

    # Load into a fresh deduplicator
    dedup2 = DocumentDeduplicator()
    assert dedup2.seen_count == 0
    dedup2.load_state(state_path)
    assert dedup2.seen_count == 2

    # doc_a and doc_b should immediately be flagged as duplicates
    assert dedup2.is_duplicate(doc_a)
    assert dedup2.is_duplicate(doc_b)
    assert dedup2.duplicate_count == 2

    # New doc should not be duplicate
    doc_c = "Texto nuevo nunca antes visto en el sistema de prueba de deduplicación."
    assert not dedup2.is_duplicate(doc_c)
    assert dedup2.seen_count == 3


def test_no_resume_archiving(tmp_path: Path, trained_tokenizer: NetelproBPETokenizer):
    """Verifies that resume=False archives existing shards and starts compiling from shard 0."""
    out_dir = tmp_path / "no_resume_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = 500
    shard_0_path = out_dir / "shard_00000.bin"
    shard_0_path.write_bytes(b"\x01\x00" * shard_size)

    meta_0 = {
        "shard_index": 0,
        "file_name": "shard_00000.bin",
        "total_tokens": shard_size,
        "total_docs": 1,
        "vocab_size": 300,
        "tokenizer_type": "NetelproBPETokenizer",
        "sha256": hashlib.sha256(shard_0_path.read_bytes()).hexdigest(),
        "created_at": "2026-09-14T00:00:00Z",
        "dtype": "uint16",
        "pad_fraction": 0.0,
    }
    with open(f"{shard_0_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_0, f, indent=2)

    # Initialize writer with resume=False
    writer = PackedShardWriter(
        out_dir=out_dir,
        shard_size=shard_size,
        tokenizer=trained_tokenizer,
        resume=False,
    )

    # Must restart from shard 0
    assert writer.current_shard_idx == 0
    assert len(writer.shards_completed) == 0
    assert not shard_0_path.exists()

    # Archived directory must exist with the old shard
    archive_dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("archive_")]
    assert len(archive_dirs) == 1
    assert (archive_dirs[0] / "shard_00000.bin").is_file()

