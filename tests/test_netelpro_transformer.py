"""Tests for the Netelpro Nano-Transformer Architecture."""

from __future__ import annotations

import pytest

from netelpro.neuro.transformer import (
    NetelproTransformer,
    NetelproTransformerConfig,
)
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


@pytest.fixture
def small_config() -> NetelproTransformerConfig:
    return NetelproTransformerConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_transformer_initialization(small_config):
    model = NetelproTransformer(small_config)
    assert len(model.blocks) == 2
    assert model.lm_head.out_features == 32
    assert model.wte.weight.shape == (32, 16)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_transformer_forward_shapes(small_config):
    model = NetelproTransformer(small_config)
    model.eval()

    idx = torch.randint(0, 32, (2, 8))  # batch=2, seq_len=8
    logits, loss, cert = model(idx)

    assert logits.shape == (2, 8, 32)
    assert loss is None
    assert cert is not None
    assert len(cert.records) == 2  # 2 blocks
    assert cert.total_latency_us > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_transformer_backward_and_ste_gradients(small_config):
    model = NetelproTransformer(small_config)
    model.train()

    idx = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss, cert = model(idx, targets=targets)
    assert loss is not None
    assert loss.item() > 0

    loss.backward()

    # Verify gradients flowed back into token embeddings and Netelpro formal MLP layers
    assert model.wte.weight.grad is not None
    assert torch.norm(model.wte.weight.grad) > 0

    mlp_layer_weight = model.blocks[0].mlp.c_fc.linear.weight
    assert mlp_layer_weight.grad is not None
    assert torch.norm(mlp_layer_weight.grad) > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_transformer_generate(small_config):
    model = NetelproTransformer(small_config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=4, temperature=0.8)

    assert out.shape == (1, 7)  # 3 + 4 = 7
    # Original prompt prefix preserved
    assert out[0, :3].tolist() == [1, 2, 3]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_transformer_formal_inhibition(small_config):
    model = NetelproTransformer(small_config)
    model.eval()

    idx = torch.randint(0, 32, (1, 4))

    # Forward with safe control flag
    _, _, cert_safe = model(idx, control_flags=1)
    assert cert_safe.records[0].active_neurons > 0

    # Forward with cut/inhibit control flag
    _, _, cert_inhibit = model(idx, control_flags=0)
    # Under safety_state=0 all neurons governed by action_boundary/activation_guard must cut to 0
    assert cert_inhibit.records[0].active_neurons == 0
    assert cert_inhibit.records[0].suppressed_neurons == cert_inhibit.records[0].total_neurons
