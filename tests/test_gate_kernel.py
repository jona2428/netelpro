"""Tests for the fused gate+lm_head kernel (docs/GATE_KERNEL_FUSION_SPEC.md).

CPU-only: exercises the reference path and the wiring (gated_lm_head's
dispatch, NetelproTransformer.gated_forward / generate(gate=...)). The
actual Triton kernel launch needs CUDA and is only exercised in
benchmarks/gate_kernel_fusion_kaggle.ipynb -- not here, and not claimed
here.
"""

from __future__ import annotations

import pytest

from netelpro.neuro.gate_kernel import (
    DOT_KERNEL_TOUCHED_TILE_THRESHOLD,
    _touched_tile_fraction,
    gated_lm_head,
    gated_lm_head_reference,
)
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch

    from netelpro.neuro.stream import NetelproStreamProcessor
    from netelpro.neuro.transformer import (
        NetelproTransformer,
        NetelproTransformerConfig,
    )


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


def _toy(vocab_size=32, n_embd=8):
    torch.manual_seed(0)
    x = torch.randn(n_embd)
    weight = torch.randn(vocab_size, n_embd)
    return x, weight


def test_reference_matches_native_kernel_contiguous_range():
    """gated_lm_head_reference must agree with NetelproVectorKernel.filter_logits_tensor
    on the same inputs -- it's the same rule (action_boundary.sl), computed
    two different ways."""
    from pathlib import Path

    from netelpro.neuro.native_kernel import NetelproVectorKernel

    x, weight = _toy()
    rule_path = Path("netelpro/neuro/rules/action_boundary.sl")
    vk = NetelproVectorKernel(rule_path)

    allowed_min, allowed_max, safety_state = 5, 20, 1
    logits = torch.nn.functional.linear(x, weight).unsqueeze(0)
    expected = vk.filter_logits_tensor(logits, allowed_min, allowed_max, safety_state).squeeze(0)

    actual = gated_lm_head_reference(x, weight, allowed_min, allowed_max, safety_state)
    assert torch.allclose(actual, expected)


def test_reference_full_freeze():
    x, weight = _toy()
    out = gated_lm_head_reference(x, weight, allowed_min=0, allowed_max=1000000, safety_state=0)
    assert torch.all(out == float("-inf"))


def test_reference_boundary_inclusive():
    x, weight = _toy(vocab_size=10)
    out = gated_lm_head_reference(x, weight, allowed_min=3, allowed_max=6, safety_state=1)
    for t in range(10):
        if 3 <= t <= 6:
            assert out[t] != float("-inf")
        else:
            assert out[t] == float("-inf")


def test_gated_lm_head_dispatches_to_reference_off_cuda():
    """On a non-CUDA tensor (this CPU test box), gated_lm_head must fall
    back to the reference path regardless of HAS_TRITON -- never attempt a
    GPU kernel launch on a CPU tensor."""
    x, weight = _toy()
    out = gated_lm_head(x, weight, allowed_min=5, allowed_max=20, safety_state=1)
    expected = gated_lm_head_reference(x, weight, allowed_min=5, allowed_max=20, safety_state=1)
    assert torch.equal(out, expected)


def test_touched_tile_fraction_full_range_is_one():
    assert _touched_tile_fraction(0, 32767, vocab_size=32768, block_n=256) == 1.0


def test_touched_tile_fraction_narrow_range_is_small():
    # [5000, 5050] sits entirely inside tile 19 (19*256=4864 .. 20*256-1=5119)
    # -- exactly one touched tile, unlike a range that straddles a tile
    # boundary (which would touch two).
    frac = _touched_tile_fraction(5000, 5050, vocab_size=32768, block_n=256)
    total_tiles = -(-32768 // 256)
    assert frac == pytest.approx(1 / total_tiles)


def test_touched_tile_fraction_broad_range_matches_pilot_measurement():
    """The exact ranges benchmarked in GATE_KERNEL_FUSION_SPEC.md Section 9
    -- confirms the dispatch threshold's own justification (broad ~91%
    touched, measured slower than baseline) is reproducible math, not a
    one-off number typed into the spec by hand."""
    frac = _touched_tile_fraction(100, 30000, vocab_size=32768, block_n=256)
    assert frac > DOT_KERNEL_TOUCHED_TILE_THRESHOLD  # broad range: dispatch must NOT pick the kernel


def test_touched_tile_fraction_empty_range_is_zero():
    assert _touched_tile_fraction(100, 50, vocab_size=32768, block_n=256) == 0.0


def test_gated_forward_matches_unfused_forward():
    """NetelproTransformer.gated_forward's last-position logits must match
    forward()'s last-position logits after the same masking is applied --
    the fusion changes where the work happens, not the result."""
    from pathlib import Path

    from netelpro.neuro.native_kernel import NetelproVectorKernel

    config = NetelproTransformerConfig(
        vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16
    )
    model = NetelproTransformer(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 5))
    allowed_min, allowed_max, safety_state = 10, 50, 1

    fused = model.gated_forward(idx, allowed_min, allowed_max, safety_state)

    logits, _, _ = model(idx)
    vk = NetelproVectorKernel(Path("netelpro/neuro/rules/action_boundary.sl"))
    unfused = vk.filter_logits_tensor(logits[:, -1, :], allowed_min, allowed_max, safety_state)

    assert torch.allclose(fused, unfused, atol=1e-4)


def test_generate_with_gate_matches_generate_without_gate_greedy():
    """With temperature effectively greedy (argmax via top_k=1) and the same
    seed, generate(gate=...) and generate() must choose the same tokens
    whenever the gate's allowed range doesn't exclude the unfused winner --
    a smoke test that wiring the gate in doesn't change sampling semantics."""
    config = NetelproTransformerConfig(
        vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16
    )
    model = NetelproTransformer(config)
    model.eval()

    sp = NetelproStreamProcessor(allowed_min=0, allowed_max=63, safety_state=1)

    idx = torch.randint(0, config.vocab_size, (1, 3))

    torch.manual_seed(42)
    out_gated = model.generate(idx.clone(), max_new_tokens=4, top_k=1, gate=sp)

    torch.manual_seed(42)
    out_plain = model.generate(idx.clone(), max_new_tokens=4, top_k=1)

    assert torch.equal(out_gated, out_plain)
