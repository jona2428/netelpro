"""Unit tests for Netelpro Writeable Binary Memory Bank (NetelproBinaryMemoryBank)."""

from __future__ import annotations

from pathlib import Path
import pytest

from netelpro.neuro.memory import NetelproBinaryMemoryBank
from netelpro.neuro.minillm import NetelproMiniLLM
from netelpro.neuro.tokenizer import NetelproTokenizer
from netelpro.neuro.transformer import NetelproTransformerConfig
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_memory_bank_write_and_retrieve(tmp_path: Path):
    mem_file = tmp_path / "test_live_memory.bin"
    bank = NetelproBinaryMemoryBank(num_slots=16, dim=64, memory_file=mem_file)

    # Write a test concept vector
    v1 = torch.randn(64)
    slot_0 = bank.write_memory("Jona es el creador de Netelpro", v1)
    assert slot_0 == 0
    assert bank.slot_masks[0].item() is True
    assert mem_file.exists()

    # Retrieve with exact same vector
    results = bank.retrieve_relevant_memories(v1, top_k=1, threshold=0.5)
    assert len(results) == 1
    assert results[0][0] == 0  # slot 0
    assert results[0][1] > 0.99  # cosine similarity ~ 1.0
    assert results[0][2]["concept"] == "Jona es el creador de Netelpro"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_memory_bank_persistence_reload(tmp_path: Path):
    mem_file = tmp_path / "persistent_memory.bin"
    bank1 = NetelproBinaryMemoryBank(num_slots=8, dim=32, memory_file=mem_file)

    v = torch.randn(32)
    bank1.write_memory("Regla de silicio: corte fail-closed en microsegundos", v)
    assert mem_file.exists()

    # Load in new instance
    bank2 = NetelproBinaryMemoryBank(num_slots=8, dim=32, memory_file=mem_file)
    active = bank2.list_memories()
    assert len(active) == 1
    assert active[0]["concept"] == "Regla de silicio: corte fail-closed en microsegundos"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_minillm_remember_and_recall():
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)

    # Remember in live silicio bank
    slot = model.remember("Python no debe generar errores off-by-one")
    assert slot >= 0

    mems = model.list_memories()
    assert len(mems) == 1
    assert mems[0]["concept"] == "Python no debe generar errores off-by-one"

    # Recall
    recalled = model.recall("Python off-by-one", top_k=1)
    assert len(recalled) == 1
    assert "Python" in recalled[0][2]["concept"]
