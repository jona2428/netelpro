"""Unit tests for Netelpro Mega Mini architecture, Binary Dataset Pipeline, and Conversational Memory."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from netelpro.neuro.tokenizer import NetelproTokenizer
from netelpro.neuro.transformer import NetelproTransformerConfig
from netelpro.neuro.minillm import NetelproMiniLLM
from netelpro.neuro.ste import HAS_TORCH
from training.data.compile_binary import compile_corpus_to_bin
from training.train_mega_llm import load_binary_dataset
from examples.mini_llm_chat import build_conversation_prompt

if HAS_TORCH:
    import torch


def test_binary_compilation_and_loading(tmp_path: Path):
    """Test compiling jsonl samples to pure uint16 binary file and loading as PyTorch tensors."""
    jsonl_file = tmp_path / "test_samples.jsonl"
    bin_file = tmp_path / "test_samples.bin"

    samples = [
        {"prompt": "<|user|>\nHola\n<|assistant|>\nSoy Netelpro en silicio binario.<|eos|>"},
        {"prompt": "<|user|>\nCódigo\n<|assistant|>\ndef test(): pass<|eos|>"},
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    block_size = 64
    n_samples, b_size, n_bytes = compile_corpus_to_bin(jsonl_file, bin_file, block_size=block_size)

    assert n_samples == 2
    assert b_size == 64
    # Each sample is (block_size + 1) * 2 bytes (uint16)
    expected_bytes = 2 * (block_size + 1) * 2
    assert n_bytes == expected_bytes
    assert bin_file.exists()

    if HAS_TORCH:
        inputs, targets = load_binary_dataset(bin_file, num_samples=2, block_size=block_size)
        assert inputs.shape == (2, block_size)
        assert targets.shape == (2, block_size)
        # Shift invariance: targets should be inputs shifted by 1 position
        assert torch.all(inputs[:, 1:] == targets[:, :-1])


def test_conversation_memory_sliding_window():
    """Test that multi-turn conversational memory trims older turns when exceeding block_size."""
    tok = NetelproTokenizer()
    history = [
        ("Turno 1: pregunta de historia antigua", "Respuesta 1 sobre historia antigua"),
        ("Turno 2: pregunta de matemáticas", "Respuesta 2 sobre matemáticas"),
        ("Turno 3: pregunta de física cuántica", "Respuesta 3 sobre física cuántica"),
    ]
    new_msg = "¿Cuál es la síntesis de todo?"

    # Small block_size should force trimming Turno 1 and Turno 2 while preserving Turno 3
    prompt_small = build_conversation_prompt(history, new_msg, tok, block_size=100)
    assert new_msg in prompt_small
    assert "Turno 3" in prompt_small
    assert "Turno 1" not in prompt_small

    # Large block_size should preserve all turns
    prompt_large = build_conversation_prompt(history, new_msg, tok, block_size=512)
    assert "Turno 1" in prompt_large
    assert "Turno 2" in prompt_large
    assert "Turno 3" in prompt_large
    assert new_msg in prompt_large


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_mega_architecture_scaled_topology():
    """Verify high-capacity scaled architecture with 6 layers, 8 heads, and 256 embedding dimension."""
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=128,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.0,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)
    param_count = sum(p.numel() for p in model.parameters())

    assert cfg.n_layer == 6
    assert cfg.n_head == 8
    assert cfg.n_embd == 256
    # Scaled capacity: > 2.5 million parameters
    assert param_count > 2_500_000
