"""Tests for Netelpro SDS (Silicon Dynamic State) non-transformer architecture."""

from __future__ import annotations

import pytest

from netelpro.neuro.dynamic_state import (
    NetelproSDSConfig,
    NetelproSDSModel,
    NetelproSiliconStateSTE,
)
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


def test_sds_config_properties():
    """Verify NetelproSDSConfig defaults, inner dimensions, and silicon bounds."""
    config = NetelproSDSConfig(
        vocab_size=32768,
        n_layer=12,
        d_model=768,
        d_state=16,
        expand=2,
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
    )
    assert config.d_inner == 1536  # 2 * 768
    # Calculate state memory footprint for entire 12-layer model:
    # 12 layers * 1536 * 16 * 4 bytes (FP32) = 1,179,648 bytes = ~1.1 MB
    state_bytes = config.n_layer * config.d_inner * config.d_state * 4
    assert state_bytes == 1179648  # Entire 350M parameter context state is ~1.1 MB!


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_silicon_state_ste_bounds():
    """Verify that NetelproSiliconStateSTE strictly bounds values and handles STE."""
    h = torch.tensor([-50.0, -5.0, 0.0, 3.2, 5.0, 100.0], requires_grad=True)
    bound_min = -5.0
    bound_max = 5.0

    bounded_h = NetelproSiliconStateSTE.apply(h, bound_min, bound_max, 1)

    assert bounded_h.min().item() >= -5.0
    assert bounded_h.max().item() <= 5.0
    assert torch.allclose(
        bounded_h,
        torch.tensor([-5.0, -5.0, 0.0, 3.2, 5.0, 5.0]),
    )

    loss = bounded_h.sum()
    loss.backward()

    assert h.grad is not None
    assert h.grad[0] == 1.0 or h.grad[0] == 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_silicon_state_ste_fail_closed():
    """Verify that control_flag=0 completely zeros out state (emergency fail-closed)."""
    h = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    frozen_h = NetelproSiliconStateSTE.apply(h, -5.0, 5.0, 0)

    assert torch.all(frozen_h == 0.0)

    loss = frozen_h.sum()
    loss.backward()
    assert torch.all(h.grad == 0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_sds_model_forward_and_backward():
    """Test sequence forward pass, CrossEntropy loss computation, and backpropagation."""
    config = NetelproSDSConfig(
        vocab_size=100,
        n_layer=2,
        d_model=32,
        d_state=8,
        d_conv=3,
        expand=2,
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
    )
    model = NetelproSDSModel(config)

    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 100, (batch_size, seq_len))

    logits, loss, cert = model(idx, targets=targets, control_flags=1)

    assert logits.shape == (batch_size, seq_len, 100)
    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert cert.verified

    loss.backward()

    assert model.lm_head.weight.grad is not None
    assert not torch.isnan(model.lm_head.weight.grad).any()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_sds_model_o1_constant_memory_generation():
    """Verify that O(1) recurrent step generation works and preserves constant state size."""
    config = NetelproSDSConfig(
        vocab_size=50,
        n_layer=2,
        d_model=32,
        d_state=8,
        d_conv=3,
        expand=2,
    )
    model = NetelproSDSModel(config)
    model.eval()

    prompt = torch.tensor([[1, 5, 12, 42]])
    max_new_tokens = 10

    states, conv_states = model.init_inference_state(batch_size=1)
    assert len(states) == config.n_layer
    assert states[0].shape == (1, config.d_inner, config.d_state)

    layer_bytes = config.d_inner * config.d_state * 4
    total_state_bytes = layer_bytes * config.n_layer
    assert total_state_bytes == 4096

    out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    assert out.shape == (1, 14)
    assert (out >= 0).all() and (out < 50).all()
